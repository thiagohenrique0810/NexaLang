#!/usr/bin/env python3
"""TurboLang CLI — compile .tl files, manage runtime, run benchmarks."""

import argparse
import json
import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


def cmd_compile(args):
    """Compile a .tl file and display the execution plan."""
    from turbolang.parser.lexer import Lexer
    from turbolang.parser.parser import Parser
    from turbolang.validator.validator import Validator
    from turboir.ir.nodes import IRGenerator
    from turboir.planner.planner import ExecutionPlanner

    with open(args.file) as f:
        source = f.read()

    print(f"Compiling {args.file}...")
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()

    validator = Validator()
    warnings = validator.validate(ast)
    for w in warnings:
        print(f"  WARNING: {w}")

    ir_program = IRGenerator().generate(ast)
    plans = ExecutionPlanner().plan(ir_program)

    if args.ir:
        print("\n── TurboIR ──")
        for m in ir_program.models:
            print(json.dumps(m.to_dict(), indent=2))

    print("\n── Execution Plan ──")
    for plan in plans:
        print(plan)

    if args.json:
        output = {
            "ir": [m.to_dict() for m in ir_program.models],
            "plans": [{
                "model": p.model_name,
                "device": p.device,
                "steps": [{"name": s.name, "action": s.action, "params": s.params} for s in p.steps],
            } for p in plans],
        }
        with open(args.json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nJSON output: {args.json}")

    return plans


def cmd_serve(args):
    """Start the TurboServe HTTP API."""
    import uvicorn
    os.environ["TURBO_HOST"] = args.host
    os.environ["TURBO_PORT"] = str(args.port)
    if args.model:
        os.environ["TURBO_DEFAULT_MODEL"] = args.model

    print(f"Starting TurboServe on {args.host}:{args.port}")
    uvicorn.run("turboserve.api.server:app", host=args.host, port=args.port, reload=args.reload)


def cmd_generate(args):
    """Generate text directly from CLI."""
    from turboruntime.core.engine import InferenceEngine, EngineConfig

    device = args.device
    if device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        except ImportError:
            device = "cpu"

    config = EngineConfig(
        model_name=args.model,
        device=device,
        quant_type=args.quant,
        cache_strategy=args.cache,
        cache_bits=args.cache_bits,
    )

    engine = InferenceEngine()
    engine.init(config)

    result = engine.generate(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print(f"\n{result.text}")
    print(f"\n── Stats ──")
    print(f"Tokens: {result.tokens_generated}")
    print(f"TTFT: {result.ttft_ms:.1f}ms")
    print(f"Total: {result.total_ms:.1f}ms")
    print(f"Speed: {result.tokens_per_sec:.1f} tok/s")

    engine.shutdown()


def cmd_bench(args):
    """Run benchmark suite."""
    from turbobench.scripts.benchmark import run_benchmark
    run_benchmark(
        model_name=args.model,
        device=args.device,
        num_runs=args.runs,
        max_tokens=args.max_tokens,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="turbolang",
        description="TurboLang LLM Runtime — CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # compile
    p_compile = sub.add_parser("compile", help="Compile a .tl file")
    p_compile.add_argument("file", help="Path to .tl file")
    p_compile.add_argument("--ir", action="store_true", help="Show TurboIR")
    p_compile.add_argument("--json", metavar="FILE", help="Save output as JSON")

    # serve
    p_serve = sub.add_parser("serve", help="Start HTTP API server")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--model", help="Auto-load model")
    p_serve.add_argument("--reload", action="store_true")

    # generate
    p_gen = sub.add_parser("generate", help="Generate text")
    p_gen.add_argument("prompt", help="Input prompt")
    p_gen.add_argument("--model", default="gpt2")
    p_gen.add_argument("--device", default="auto")
    p_gen.add_argument("--max-tokens", type=int, default=100)
    p_gen.add_argument("--temperature", type=float, default=0.7)
    p_gen.add_argument("--quant", default="none", choices=["none", "int4", "int8"])
    p_gen.add_argument("--cache", default="standard", choices=["standard", "turboquant", "none"])
    p_gen.add_argument("--cache-bits", type=int, default=16)

    # bench
    p_bench = sub.add_parser("bench", help="Run benchmarks")
    p_bench.add_argument("--model", default="gpt2")
    p_bench.add_argument("--device", default="auto")
    p_bench.add_argument("--runs", type=int, default=3)
    p_bench.add_argument("--max-tokens", type=int, default=50)

    args = parser.parse_args()

    if args.command == "compile":
        cmd_compile(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "bench":
        cmd_bench(args)


if __name__ == "__main__":
    main()
