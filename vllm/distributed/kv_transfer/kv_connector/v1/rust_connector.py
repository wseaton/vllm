# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Rust-based KV Connector implementation using PyO3

simple in-memory KV cache store implemented in Rust for proof-of-concept.
uses CPU copies only (no GPU-to-GPU transfer yet).
"""
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

try:
    import vllm_nixl  # type: ignore

    NixlAgent = vllm_nixl.NixlAgent
    TcpSideChannel = vllm_nixl.TcpSideChannel
    ConnectorScheduler = vllm_nixl.ConnectorScheduler
    ConnectorWorker = vllm_nixl.ConnectorWorker
    RequestMeta = vllm_nixl.RequestMeta
    NIXL_AVAILABLE = True
except ImportError:
    NIXL_AVAILABLE = False
    NixlAgent = None  # type: ignore
    TcpSideChannel = None  # type: ignore
    ConnectorScheduler = None  # type: ignore
    ConnectorWorker = None  # type: ignore
    RequestMeta = None  # type: ignore

logger = init_logger(__name__)


def _torch_dtype_to_str(dtype: torch.dtype) -> str:
    """convert torch dtype to string for Rust storage"""
    mapping = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.int64: "int64",
        torch.int32: "int32",
    }
    return mapping.get(dtype, str(dtype))


def _str_to_torch_dtype(dtype_str: str) -> torch.dtype:
    """convert string back to torch dtype"""
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int64": torch.int64,
        "int32": torch.int32,
    }
    return mapping.get(dtype_str, torch.float32)


class RustConnectorMetadata(KVConnectorMetadata):
    """
    Metadata for Rust-based KV transfers, matching the structure
    returned by ConnectorScheduler.build_connector_meta()
    """
    def __init__(self, rust_dict: dict[str, Any]):
        """
        Initialize from Rust-generated dictionary.

        Expected structure:
        {
            "reqs_to_recv": {req_id: RequestMeta, ...},
            "reqs_to_save": {req_id: RequestMeta, ...},
            "reqs_to_send": {req_id: expiration_timestamp, ...},
            "reqs_in_batch": [req_id, ...],
            "reqs_not_processed": [req_id, ...]
        }
        """
        self.reqs_to_recv: dict[str, Any] = rust_dict.get("reqs_to_recv", {})
        self.reqs_to_save: dict[str, Any] = rust_dict.get("reqs_to_save", {})
        self.reqs_to_send: dict[str, float] = rust_dict.get("reqs_to_send", {})
        self.reqs_in_batch: set[str] = set(rust_dict.get("reqs_in_batch", []))
        self.reqs_not_processed: set[str] = set(rust_dict.get("reqs_not_processed", []))


class RustKVConnectorV1(KVConnectorBase_V1):
    """
    KV Connector backed by NIXL with Rust implementation.

    Uses NIXL for GPU-to-GPU KV cache transfers.
    Supports side channel for metadata exchange.
    """

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        if not NIXL_AVAILABLE:
            raise RuntimeError(
                "NIXL Rust connector not available. "
                "Build it with: cd vllm/distributed/kv_transfer/kv_connector/v1/rust_connector && maturin develop"
            )

        super().__init__(vllm_config=vllm_config, role=role)

        self._role = role
        self._block_size = vllm_config.cache_config.block_size

        # Get engine and network configuration
        kv_transfer_config = vllm_config.kv_transfer_config
        engine_id = kv_transfer_config.kv_connector_extra_config.get("engine_id", "vllm_engine")
        side_channel_host = kv_transfer_config.kv_connector_extra_config.get("side_channel_host", "localhost")

        # Calculate side channel port based on DP and TP ranks
        # Base port + DP rank * TP size + TP rank
        parallel_config = vllm_config.parallel_config
        base_port = kv_transfer_config.kv_connector_extra_config.get("side_channel_base_port", 45000)
        dp_rank = getattr(parallel_config, 'dp_rank', 0)
        tp_size = parallel_config.tensor_parallel_size
        tp_rank = parallel_config.rank % tp_size
        side_channel_port = base_port + dp_rank * tp_size + tp_rank

        if role == KVConnectorRole.SCHEDULER:
            # Initialize Rust scheduler
            self._scheduler = ConnectorScheduler(
                engine_id=engine_id,
                block_size=self._block_size,
                side_channel_host=side_channel_host,
                side_channel_port=side_channel_port
            )
            self._worker = None
            logger.info(f"Initialized Rust KV Connector Scheduler (engine={engine_id}, port={side_channel_port})")
        else:
            # Initialize Rust worker
            self._scheduler = None
            self._worker = ConnectorWorker(
                engine_id=engine_id,
                tp_rank=tp_rank,
                block_size=self._block_size,
                side_channel_host=side_channel_host,
                side_channel_port=side_channel_port
            )

            # Initialize NIXL agent for worker
            # TODO: Get NIXL config from vllm_config
            nixl_config = {
                "enable_prog_thread": True,
                "num_workers": 4,
            }
            agent_name = f"{engine_id}_rank_{tp_rank}"
            self._nixl_agent = NixlAgent(agent_name, nixl_config)

            logger.info(f"Initialized Rust KV Connector Worker (engine={engine_id}, tp_rank={tp_rank})")

    # ==============================
    # Worker-side methods
    # ==============================

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Register KV caches with NIXL for GPU-to-GPU transfers"""
        if self._worker is None:
            logger.warning("Worker not initialized, skipping KV cache registration")
            return

        # Call Rust worker to register KV caches
        self._worker.register_kv_caches(kv_caches, self._nixl_agent)
        logger.info(f"Registered {len(kv_caches)} KV cache layers with NIXL")

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """load KV cache via NIXL transfer into vLLM's paged buffer"""
        if self._worker is None:
            return

        # Get connector metadata from forward context
        connector_meta = getattr(forward_context, 'connector_meta', None)
        if connector_meta is None:
            return

        # Convert metadata to dict for Rust
        metadata_dict = {
            "reqs_to_recv": connector_meta.reqs_to_recv if hasattr(connector_meta, 'reqs_to_recv') else {},
            "reqs_to_save": connector_meta.reqs_to_save if hasattr(connector_meta, 'reqs_to_save') else {},
            "reqs_to_send": connector_meta.reqs_to_send if hasattr(connector_meta, 'reqs_to_send') else {},
            "reqs_in_batch": list(connector_meta.reqs_in_batch) if hasattr(connector_meta, 'reqs_in_batch') else [],
            "reqs_not_processed": list(connector_meta.reqs_not_processed) if hasattr(connector_meta, 'reqs_not_processed') else [],
        }

        # Call Rust worker to start loading KV
        self._worker.start_load_kv(metadata_dict)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Wait for a specific layer to finish loading (no-op for NIXL)"""
        # NIXL handles transfers asynchronously, polling happens in get_finished_transfers
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """save KV cache layer (NIXL handles this during registration)"""
        # NOTE: With NIXL, we don't explicitly "save" KV per layer.
        # Instead, we register the entire KV cache with NIXL at initialization,
        # and it handles transfers directly from GPU memory.
        pass

    def wait_for_save(self):
        """no async saving in this simple implementation"""
        pass

    def get_finished_transfers(self) -> tuple[set[str], set[str]]:
        """Poll for finished KV transfers (receiving and sending)"""
        if self._worker is None:
            return set(), set()

        # Call Rust worker to poll for finished transfers
        return self._worker.get_finished()

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Get block IDs that failed to load"""
        if self._worker is None:
            return set()

        # Call Rust worker to get error blocks
        return self._worker.get_block_ids_with_load_errors()

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """check if request has remote KV cache available"""
        if self._scheduler is None:
            return 0, False

        # Check if request has kv_transfer_params indicating remote KV
        if request.kv_transfer_params is None:
            return 0, False

        # Call Rust scheduler implementation
        return self._scheduler.get_num_new_matched_tokens(
            request.request_id,
            num_computed_tokens,
            request.kv_transfer_params
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """mark request for loading if it has external tokens"""
        if self._scheduler is None or num_external_tokens == 0:
            return

        # Extract block IDs from KVCacheBlocks
        block_ids = blocks.block_ids if hasattr(blocks, 'block_ids') else []

        # Call Rust scheduler implementation
        self._scheduler.update_state_after_alloc(
            request.request_id,
            list(block_ids),
            num_external_tokens,
            request.kv_transfer_params or {}
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """build metadata for worker to load/save KV"""
        if self._scheduler is None:
            # Return empty metadata if no scheduler
            return RustConnectorMetadata({})

        # Call Rust scheduler to build metadata
        rust_dict = self._scheduler.build_connector_meta()

        # Wrap in Python metadata class
        return RustConnectorMetadata(rust_dict)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """cleanup when request finishes"""
        if self._scheduler is None:
            return False, None

        # Call Rust scheduler to handle request completion
        return self._scheduler.request_finished(
            request.request_id,
            block_ids,
            request.kv_transfer_params
        )
