/*
 * PyTorch extension wrapper for the JIT-compiled fused_a_gemm kernel.
 * Included by torch.utils.cpp_extension.load at runtime.
 *
 * Preprocessor defines (set via -D flags during JIT compilation):
 *   FUSED_A_GEMM_HD_IN  — input dimension (K)
 *   FUSED_A_GEMM_HD_OUT — output dimension (N)
 */

#include "dsv3_fused_a_gemm.cuh"

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

torch::Tensor fused_a_gemm_forward(
    torch::Tensor mat_a,
    torch::Tensor mat_b) {
    TORCH_CHECK(mat_a.dim() == 2 && mat_b.dim() == 2,
                "mat_a and mat_b must be 2D");
    int num_tokens = mat_a.size(0);
    TORCH_CHECK(num_tokens >= 1 && num_tokens <= 16,
                "num_tokens must be in [1, 16], got ", num_tokens);

    constexpr int kHdIn = FUSED_A_GEMM_HD_IN;
    constexpr int kHdOut = FUSED_A_GEMM_HD_OUT;
    TORCH_CHECK(mat_a.size(1) == kHdIn, "mat_a K dim mismatch, expected ", kHdIn,
                " got ", mat_a.size(1));
    TORCH_CHECK(mat_b.size(1) == kHdOut, "mat_b N dim mismatch, expected ", kHdOut,
                " got ", mat_b.size(1));
    TORCH_CHECK(mat_a.stride(1) == 1, "mat_a must be row-major");
    TORCH_CHECK(mat_b.stride(0) == 1, "mat_b must be column-major");
    TORCH_CHECK(mat_a.scalar_type() == torch::kBFloat16,
                "mat_a must be BFloat16");
    TORCH_CHECK(mat_b.scalar_type() == torch::kBFloat16,
                "mat_b must be BFloat16");

    auto output = torch::empty(
        {num_tokens, kHdOut},
        mat_a.options().dtype(torch::kBFloat16));

    constexpr int kTileM = pick_tile_m(kHdIn, kHdOut);
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    auto* out = reinterpret_cast<bf16_t*>(output.mutable_data_ptr());
    auto* a = reinterpret_cast<bf16_t const*>(mat_a.data_ptr());
    auto* b = reinterpret_cast<bf16_t const*>(mat_b.data_ptr());

    if (num_tokens <= 8) {
        invokeFusedAGemm<bf16_t, kHdIn, kHdOut, 8, kTileM>(
            out, a, b, num_tokens, stream);
    } else {
        invokeFusedAGemm<bf16_t, kHdIn, kHdOut, 16, kTileM>(
            out, a, b, num_tokens, stream);
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_a_gemm_forward", &fused_a_gemm_forward,
          "Fused A GEMM forward (JIT compiled)");
}
