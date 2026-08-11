# OPERATOR NOTE — proven-port era; anchor to beat is 8.571 ms

The shape-general fused A-GEMM port has now passed the full ladder twice:
GSM8K 0.94-0.965, TPOT 8.613 then 8.571 ms (117 tok/s). The recipe: generalized kernel
body, weight.T column-major views, kill-switch env var; builds must be full when csrc
changes (the gate handles this — diff vs pinned baseline).

This run has PREFLIGHT: env problems refuse the run before your first turn, and the
stock baseline TPOT is measured for you — decide is anchored now. Declare your
kill-switch env name via VLLM_GLM_TOGGLE so the ab-toggle rung stops skipping.

Fetch full logs via broker fetch_log before diagnosing. Delete this file after reading.
