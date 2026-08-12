#!/usr/bin/env python3
"""Emit one code-gen phase's steps as JSON so the crucible agent can run them in ITS OWN session.

The delegated pipeline normally drives these phases by spawning a fresh Claude session per step
(`common.claude_utils.claude_run`). Those sessions inherit none of the harness's environment, which
is where every hard failure came from: no Vertex credentials (`num_turns = 0`, no error text), no
reachable broker MCP tools, and a long-lived background process writing into the very tree
`codegen_build` is trying to snapshot (`sandbox tree changed during sync`).

This keeps the part that carries the domain expertise — the prompts, built by
`use_cases/llm_framework.py` — and drops only the executor. Nothing is forked: the prompts come
from the installed pipeline, so they track upstream. The agent reads a step's `prompt`, does the
work with its own tools, and writes the step's `output_files`.

Phase sequencing, iteration counts and keep/discard are NOT reimplemented here. Crucible owns those.

Usage:
  render-phases.py --config <code_gen_config.json> --phase code_trace
  render-phases.py --config <cfg> --phase code_port_plan --iteration 1
  render-phases.py --config <cfg> --phase code_gen --iteration 1 --code-trace-files a.txt b.txt
"""

import argparse
import inspect
import json
import sys

PHASES = {
    "code_trace": "gen_code_trace_steps",
    "code_port_plan": "gen_code_port_plan_iter_steps",
    "test_plan": "gen_test_plan_iter_steps",
    "code_gen": "gen_code_gen_iter_steps",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--phase", required=True, choices=sorted(PHASES))
    ap.add_argument("--iteration", type=int, default=1)
    ap.add_argument("--code-trace-files", nargs="*", default=[])
    ap.add_argument("--code-port-plan-file", default=None)
    ap.add_argument("--test-plan-file", default=None)
    ap.add_argument("--prev-output-file", default=None)
    ap.add_argument("--prev-output-summary-file", default=None)
    args = ap.parse_args()

    from auto_code_gen.run_code_gen import load_config_and_use_case

    config, use_case = load_config_and_use_case(args.config)
    claude_config = config.make_claude_config()
    context = use_case.create_context_str(claude_config, config)

    gen = getattr(use_case, PHASES[args.phase], None)
    if gen is None:
        print(f"use case has no {PHASES[args.phase]}", file=sys.stderr)
        return 2

    # Bind by NAME against the generator's real signature rather than a fixed positional call: these
    # take different parameter sets per phase and upstream is free to add more. An unknown required
    # parameter fails loudly below instead of silently rendering a half-built prompt.
    available = {
        "context": context,
        "config": config,
        "iteration": args.iteration,
        "code_trace_files": args.code_trace_files,
        "code_port_plan_file": args.code_port_plan_file,
        "test_plan_file": args.test_plan_file,
        "prev_output_file": args.prev_output_file,
        "prev_output_summary_file": args.prev_output_summary_file,
        "disallowed_modules": getattr(config, "disallowed_modules", []),
    }
    sig = inspect.signature(gen)
    kwargs = {n: available[n] for n in sig.parameters if n in available}
    missing = [
        n
        for n, p in sig.parameters.items()
        if n not in kwargs and p.default is inspect.Parameter.empty and n != "self"
    ]
    if missing:
        print(f"{PHASES[args.phase]} needs parameters this renderer does not supply: {missing}",
              file=sys.stderr)
        return 3

    result = gen(**kwargs)
    steps = result[0] if isinstance(result, tuple) else result

    # Some phases return a structured prompt (a dict, occasionally holding a callable the
    # executor would invoke) instead of a string; json.dumps chokes on the callable and every
    # run's agent kept rediscovering and repatching this in-sandbox. Callables render as their
    # name — the agent reads the prompt, it never calls into it.
    def serialize_prompt(p):
        if isinstance(p, str):
            return p
        if isinstance(p, dict):
            return {k: (v.__name__ if callable(v) else v) for k, v in p.items()}
        return str(p)

    print(json.dumps(
        [{"name": s.name, "prompt": serialize_prompt(s.prompt),
          "output_files": list(s.output_files or [])}
         for s in steps],
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
