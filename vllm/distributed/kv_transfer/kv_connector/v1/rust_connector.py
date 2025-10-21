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
    NIXL_AVAILABLE = True
except ImportError:
    NIXL_AVAILABLE = False
    NixlAgent = None  # type: ignore
    TcpSideChannel = None  # type: ignore

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

        role_str = "scheduler" if role == KVConnectorRole.SCHEDULER else "worker"
        # TODO: Initialize NIXL agent here instead of simple in-memory store
        # For now, we'll keep the basic structure but note that it needs NIXL integration
        self._block_size = vllm_config.cache_config.block_size

        # scheduler-side tracking
        self._requests_need_load: dict[str, "Request"] = {}

        logger.info(f"Initialized Rust KV Connector with NIXL support (role={role_str})")

    # ==============================
    # Worker-side methods
    # ==============================

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """load KV cache via NIXL transfer into vLLM's paged buffer"""
        # TODO: Implement NIXL-based KV loading
        # 1. Get metadata with remote engine info
        # 2. Initiate handshake if needed (via TcpSideChannel)
        # 3. Start NIXL transfer for KV blocks
        # 4. Poll for completion
        # 5. Inject received blocks into paged buffer
        logger.warning("start_load_kv: NIXL integration not yet implemented")
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        """no async loading in this simple implementation"""
        return

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
        return

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """check if request has remote KV cache available"""
        # Check if request has kv_transfer_params indicating remote KV
        if request.kv_transfer_params is None:
            return 0, False

        # For requests with remote KV, calculate how many tokens to load
        num_prompt_tokens = len(request.prompt_token_ids or [])
        num_external_tokens = num_prompt_tokens - num_computed_tokens

        if num_external_tokens > 0:
            logger.info(
                f"Request {request.request_id} can load {num_external_tokens} "
                f"tokens from remote KV cache"
            )
            # Return external tokens and async=False (sync loading for now)
            return num_external_tokens, False

        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """mark request for loading if it has external tokens"""
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """build metadata for worker to load/save KV"""
        # TODO: Create proper metadata class for NIXL connector
        # For now, return a simple placeholder
        class SimpleMetadata(KVConnectorMetadata):
            def __init__(self):
                self.requests_to_load = {}
                self.requests_to_save = {}

        meta = SimpleMetadata()

        # Track requests that need to load from remote KV
        for req_id, request in self._requests_need_load.items():
            if request.kv_transfer_params:
                meta.requests_to_load[req_id] = request.kv_transfer_params

        self._requests_need_load.clear()

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """cleanup when request finishes"""
        # TODO: Implement NIXL cleanup if needed
        # For now, just return False to indicate vLLM should free blocks
        return False, None
