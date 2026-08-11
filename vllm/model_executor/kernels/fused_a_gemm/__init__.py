# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JIT-compiled fused_a_gemm kernel for MLA projections.

Supports generalized shapes via per-(hd_in, hd_out) JIT compilation.
The kernel is optimized for num_tokens in [1, 16] with BF16 I/O and
FP32 accumulation. Falls back to F.linear for num_tokens > 16.

Kill-switch: set VLLM_DISABLE_FUSED_A_GEMM_JIT=1 to disable.
"""

import functools
import logging
import os
import pathlib

import torch
from torch.utils.cpp_extension import load

from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

logger = logging.getLogger(__name__)

_FUSED_A_GEMM_JIT_DISABLED = bool(
    int(os.getenv("VLLM_DISABLE_FUSED_A_GEMM_JIT", "0"))
)

if _FUSED_A_GEMM_JIT_DISABLED:
    logger.info("Fused A GEMM JIT kernel is DISABLED (env override)")
else:
    logger.info("Fused A GEMM JIT kernel is ENABLED")

_CUDA_DIR = pathlib.Path(__file__).parent

# Explicit allowlists — every entry must have a verified CUDA template
# instantiation via pick_tile_m and static_asserts (KB Entry 33).
_SUPPORTED_QKV_A_SHAPES: set[tuple[int, int]] = {
    (2112, 7168),  # DeepSeek-V3 QKV-A (tile_m=16)
    (2624, 6144),  # GLM-5.2 QKV-A at TP=4 (tile_m=32)
}

_SUPPORTED_Q_B_SHAPES: set[tuple[int, int]] = {
    (4096, 2048),  # GLM-5.2 Q-B at TP=4 (tile_m=32)
}

_ALL_SUPPORTED_SHAPES = _SUPPORTED_QKV_A_SHAPES | _SUPPORTED_Q_B_SHAPES


@functools.lru_cache(maxsize=None)
def _load_jit_module(hd_in: int, hd_out: int):
    """JIT compile the fused_a_gemm kernel for specific (hd_in, hd_out).

    Each (hd_in, hd_out) pair is compiled once and cached via lru_cache.
    """
    module = load(
        name=f"fused_a_gemm_{hd_in}_{hd_out}",
        sources=[str(_CUDA_DIR / "dsv3_fused_a_gemm_wrapper.cu")],
        extra_include_paths=[str(_CUDA_DIR)],
        extra_cuda_cflags=[
            f"-DFUSED_A_GEMM_HD_IN={hd_in}",
            f"-DFUSED_A_GEMM_HD_OUT={hd_out}",
            "-gencode", "arch=compute_90a,code=sm_90a",
            "-gencode", "arch=compute_100a,code=sm_100a",
            "--expt-relaxed-constexpr",
            "-O3",
        ],
        verbose=False,
    )
    logger.info(
        "JIT compiled fused_a_gemm kernel for hd_in=%d, hd_out=%d",
        hd_in, hd_out,
    )
    return module


def _fused_a_gemm_jit_impl(
    input_: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    num_tokens = input_.shape[0]
    if (
        not _FUSED_A_GEMM_JIT_DISABLED
        and 0 < num_tokens <= 16
    ):
        hd_in = input_.shape[1]
        hd_out = weight.shape[0]
        module = _load_jit_module(hd_in, hd_out)
        return module.fused_a_gemm_forward(input_, weight.T)
    else:
        return torch.nn.functional.linear(input_, weight)


def _fused_a_gemm_jit_fake(
    input_: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return input_.new_empty(input_.shape[0], weight.shape[0])


direct_register_custom_op(
    op_name="fused_a_gemm_jit",
    op_func=_fused_a_gemm_jit_impl,
    mutates_args=[],
    fake_impl=_fused_a_gemm_jit_fake,
)


def is_fused_a_gemm_eligible(weight: torch.Tensor) -> bool:
    """Check if a weight tensor is eligible for fused_a_gemm JIT.

    Uses explicit shape allowlists per KB Entry 33.
    """
    if _FUSED_A_GEMM_JIT_DISABLED:
        return False
    shape = (weight.shape[0], weight.shape[1])
    return (
        weight.dtype == torch.bfloat16
        and shape in _ALL_SUPPORTED_SHAPES
        and current_platform.is_cuda()
        and (
            current_platform.is_device_capability(90)
            or current_platform.is_device_capability_family(100)
        )
    )
