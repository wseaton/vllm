---
name: profile
description: Profile the SERVED vLLM candidate so you optimize the MEASURED hot path instead of guessing. The throughput gate is end-to-end; a GPU trace names the exact CUDA kernels and correlates each one back to its transformer-block source line. Tools: vllm-profile (capture), vllm-analyze (kernel->source correlation). Inverts neuralmagic/ai_auto_perf_analysis into this loop.
---

# profile — see where the GPU time actually goes

The gate measures total token throughput over the whole served run. A single kernel change is a sliver
of that number, so a raw before/after often comes back "within noise" and tells you nothing about
*where* the time went. Stop guessing. A GPU trace attributes time to specific **CUDA kernels**, and the
analysis correlates each kernel to its **transformer-block operation and source line** — exactly the
lines to change.

**Profile to FIND what to change; the gate to SCORE the candidate.** One hypothesis per turn.

These are PATH tools in your sandbox (the vLLM domain runs the broker off — capture + analysis happen
right here, on the same GPU the gate uses, over the gate's frozen workload).

## The two tools

`vllm-profile` — capture a PyTorch GPU trace. It serves your CURRENT edits, drives the frozen workload,
brackets steady state with `/start_profile`+`/stop_profile`, and prints a JSON handle:

```json
{"status":"captured","trace":"/tmp/vllm-prof/.../trace.json.gz","log":"/tmp/vllm-profile-run.log","model":"...","concurrency":16,"input_len":512,"output_len":128}
```

It uses the SAME frozen workload as the gate (so the trace explains the gated number), just fewer
prompts (`VLLM_PROFILE_PROMPTS`) — a trace needs steady state, not the full run. `status:"error"` means
the editable install or the server boot failed; read `error` and fix before profiling.

`vllm-analyze [TRACE]` — correlate the trace to source. Reads your workspace source to list each
transformer block's high-level ops, extracts the GPU kernels from the trace, maps every kernel to its
op + source line, and proposes bottleneck fixes. Defaults to the newest capture. Prints:

```json
{"status":"analyzed","output_dir":"/tmp/vllm-analyze","files":["...median_block.txt","...gpu_ops_to_blocks.txt",...],"median_block_head":"..."}
```

Then `Read` the files in `output_dir` for the full picture:

| file | what you learn |
| --- | --- |
| `median_block.txt` | the representative transformer block the analysis focused on |
| `gpu_ops_to_blocks.txt` | every GPU kernel mapped to its high-level op + source reference |
| `transformer_block_high_level_ops.txt` | the op sequence read from your source |
| (perf analysis) | bottlenecks + ranked improvement proposals with code refs |

## The loop

1. `vllm-profile` — capture a trace of your current candidate over the frozen workload.
2. `vllm-analyze` — get the kernel→source correlation; `Read` `gpu_ops_to_blocks.txt` for the hot path.
3. Form ONE hypothesis against the hottest correlated source lines. Edit them.
4. Re-gate (the loop measures) to SCORE it. Profiling finds the target; the gate decides keep/discard.

If `vllm-analyze` returns `analysis engine not in this image`, this sandbox was built without the
analysis layer — fall back to reading the raw trace yourself (`gpu_ops.txt` after a capture) or
`escalate` if you can't make progress.
