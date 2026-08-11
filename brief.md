========================================================================================================================
CROSS-FRAMEWORK IMPROVEMENT PLAN — vLLM (TARGET) vs SGLang (V3)
Model: nvidia/GLM-5.2-NVFP4 | GPU: NVIDIA B200 | TP=4 | BS=1, ISL=4, OSL=1024
========================================================================================================================

TARGET: vLLM v0.25.1 (V1 engine) — commit b07ec92
REFERENCE: SGLang — commit d6ef68

vLLM is 21.2 us/layer (19.7%) slower than SGLang for the dominant SharedIndexer block.
Estimated forward-pass gap: 1653 us (19.5%) across 75 MoE layers.




========================================================================================================================
PROPOSAL #1: GENERALIZE FUSED_A_GEMM FOR QKV-A + Q-B PROJECTIONS
========================================================================================================================

Impact: -7520ns/layer (-7.0% of block time) | Priority: P0 | Difficulty: MEDIUM-HIGH

ARCHITECTURAL PORTABILITY: VALIDATED ✓
  Both frameworks: identical GEMM semantics ([6144]→[2624]), replicated BF16, SM90+.

ROOT CAUSE:
  Two hard-coded gates block GLM-5.2 shapes:
  Gate 1 — Python (deepseek_v2.py:927-932): shape[0]==2112 and shape[1]==7168 — FAILS for [2624,6144]
  Gate 2 — CUDA (dsv3_fused_a_gemm.cu:713-718): constexpr kHdIn=7168, kHdOut=2112 + STD_TORCH_CHECK

  [CRITICAL] Relaxing Gate 1 WITHOUT fixing Gate 2 → RUNTIME CRASH at STD_TORCH_CHECK.

  SGLang uses alignment-based gating: shape[0]%16==0, shape[1]%256==0 (jit_kernel/fused_a_gemm.py:37-44).

STEP-BY-STEP:

  Step 1 (MANDATORY — sub-steps 1a+1b together):

    1a. Replace or extend the CUDA kernel for flexible shapes.

      Option A (recommended): Port SGLang's JIT kernel (jit_kernel/fused_a_gemm.py +
        csrc/gemm/dsv3_fused_a_gemm.cuh). Creates vllm/model_executor/kernels/linear/dsv3_fused_a_gemm_jit.py.
        Register via direct_register_custom_op with fake_impl (torch.compile compat per KB Entry 4).

      Option B (model-specific): Add template instantiations to dsv3_fused_a_gemm.cu:
        template void invokeFusedAGemm<__nv_bfloat16, 6144, 2624, 8>(...);
        template void invokeFusedAGemm<__nv_bfloat16, 6144, 2624, 16>(...);
        template void invokeFusedAGemm<__nv_bfloat16, 2048, 4096, 8>(...);
        template void invokeFusedAGemm<__nv_bfloat16, 2048, 4096, 16>(...);
        Update dispatch function dsv3_fused_a_gemm() to route by (hd_in, hd_out).

    1b. Relax the Python shape gate.

      File: vllm/model_executor/models/deepseek_v2.py (lines 927-932)

      BEFORE:
        self._use_min_latency_gemm = (
            hasattr(self, "weight")
            and self.weight.dtype == torch.bfloat16
            and self.weight.shape[0] == 2112
            and self.weight.shape[1] == 7168
            and current_platform.is_cuda()
            and (current_platform.is_device_capability(90)
                 or current_platform.is_device_capability_family(100))
        )

      AFTER:
        self._use_min_latency_gemm = (
            hasattr(self, "weight")
            and self.weight.dtype == torch.bfloat16
            and self.weight.shape[0] % 16 == 0
            and self.weight.shape[1] % 256 == 0
            and current_platform.is_cuda()
            and (current_platform.is_device_capability(90)
                 or current_platform.is_device_capability_family(100))
        )

  Step 2: Enable fused_a_gemm for Q-B projection.

    vLLM only uses min-latency GEMM for QKV-A. Port SGLang's q_b_proj_forward pattern.

    File: vllm/model_executor/layers/mla/mla.py (around line 169)

      # In __init__:
      self._use_min_latency_q_b = (
          hasattr(self.q_b_proj, "weight")
          and self.q_b_proj.weight.dtype == torch.bfloat16
          and self.q_b_proj.weight.shape[0] % 16 == 0
          and self.q_b_proj.weight.shape[1] % 256 == 0
          and current_platform.is_cuda()
          and (current_platform.is_device_capability(90)
               or current_platform.is_device_capability_family(100))
      )

      # In forward (replace q = self.q_proj_layer(q_proj_input)):
      if self._use_min_latency_q_b and q_proj_input.shape[0] <= 16:
          q = torch.ops.vllm.dsv3_fused_a_gemm_jit(q_proj_input, self.q_b_proj.weight)
      else:
          q = self.q_proj_layer(q_proj_input)


