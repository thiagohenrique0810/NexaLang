"""TurboBench — Benchmark suite for TurboLang LLM Runtime.

Measures:
  - TTFT (Time to First Token)
  - Tokens per second
  - Latency p50/p95
  - Total generation time
  - Memory usage
  - Impact of quantization and KV cache compression
"""

import json
import csv
import os
import sys
import time
import logging
import statistics
from dataclasses import dataclass, field

logger = logging.getLogger("turbobench")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@dataclass
class BenchmarkResult:
    name: str = ""
    model: str = ""
    config: str = ""
    prompt: str = ""
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    tokens_generated: int = 0
    tokens_per_sec: float = 0.0
    prompt_tokens: int = 0
    memory_used_mb: float = 0.0


@dataclass
class BenchmarkSuite:
    results: list[BenchmarkResult] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


PROMPTS = [
    "Once upon a time in a land far away,",
    "The quick brown fox jumped over the lazy dog.",
    "Explain quantum computing in simple terms:",
    "Write a Python function that sorts a list:",
    "The meaning of life is",
]


def get_memory_mb() -> float:
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def run_benchmark(model_name: str = "gpt2", device: str = "auto",
                  configs: list[dict] = None, num_runs: int = 3,
                  max_tokens: int = 50) -> BenchmarkSuite:
    from turboruntime.core.engine import InferenceEngine, EngineConfig

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

    if configs is None:
        configs = [
            {"name": "baseline", "quant_type": "none", "cache_strategy": "standard", "cache_bits": 16},
            {"name": "int8", "quant_type": "int8", "cache_strategy": "standard", "cache_bits": 16},
            {"name": "int4", "quant_type": "int4", "cache_strategy": "standard", "cache_bits": 16},
            {"name": "turboquant_kv", "quant_type": "none", "cache_strategy": "turboquant", "cache_bits": 4},
            {"name": "int8+turboquant", "quant_type": "int8", "cache_strategy": "turboquant", "cache_bits": 4},
        ]

    suite = BenchmarkSuite()

    for cfg in configs:
        config_name = cfg.pop("name", "unnamed")
        logger.info(f"\n{'='*60}")
        logger.info(f"Config: {config_name}")
        logger.info(f"{'='*60}")

        engine = InferenceEngine()
        mem_before = get_memory_mb()

        try:
            engine_cfg = EngineConfig(
                model_name=model_name,
                device=device,
                **cfg,
            )
            engine.init(engine_cfg)
        except Exception as e:
            logger.error(f"Failed to initialize with config {config_name}: {e}")
            cfg["name"] = config_name  # restore
            continue

        mem_after = get_memory_mb()
        cfg["name"] = config_name  # restore

        for prompt in PROMPTS:
            ttfts = []
            totals = []
            tok_counts = []
            tok_rates = []

            for run in range(num_runs):
                try:
                    result = engine.generate(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=0.7,
                    )
                    ttfts.append(result.ttft_ms)
                    totals.append(result.total_ms)
                    tok_counts.append(result.tokens_generated)
                    tok_rates.append(result.tokens_per_sec)
                except Exception as e:
                    logger.error(f"Run {run+1} failed: {e}")

            if not ttfts:
                continue

            suite.results.append(BenchmarkResult(
                name=config_name,
                model=model_name,
                config=json.dumps(cfg),
                prompt=prompt[:50],
                ttft_ms=statistics.median(ttfts),
                total_ms=statistics.median(totals),
                tokens_generated=int(statistics.median(tok_counts)),
                tokens_per_sec=statistics.median(tok_rates),
                prompt_tokens=len(prompt.split()),
                memory_used_mb=mem_after - mem_before,
            ))

        engine.shutdown()

    # Generate summary
    if suite.results:
        suite.summary = _build_summary(suite.results)
        _print_results(suite)
        _save_results(suite)

    return suite


def _build_summary(results: list[BenchmarkResult]) -> dict:
    by_config = {}
    for r in results:
        if r.name not in by_config:
            by_config[r.name] = []
        by_config[r.name].append(r)

    summary = {}
    for name, res_list in by_config.items():
        ttfts = [r.ttft_ms for r in res_list]
        totals = [r.total_ms for r in res_list]
        rates = [r.tokens_per_sec for r in res_list]

        summary[name] = {
            'avg_ttft_ms': statistics.mean(ttfts),
            'p50_ttft_ms': statistics.median(ttfts),
            'p95_ttft_ms': sorted(ttfts)[int(len(ttfts) * 0.95)] if len(ttfts) > 1 else ttfts[0],
            'avg_total_ms': statistics.mean(totals),
            'avg_tokens_per_sec': statistics.mean(rates),
            'memory_mb': res_list[0].memory_used_mb,
            'num_prompts': len(res_list),
        }

    return summary


def _print_results(suite: BenchmarkSuite):
    print(f"\n{'='*80}")
    print("TurboBench Results")
    print(f"{'='*80}")
    print(f"{'Config':<20} {'TTFT(ms)':<10} {'Total(ms)':<12} {'Tok/s':<10} {'Tokens':<8} {'Mem(MB)':<10}")
    print(f"{'-'*80}")

    for name, s in suite.summary.items():
        print(f"{name:<20} {s['avg_ttft_ms']:<10.1f} {s['avg_total_ms']:<12.1f} "
              f"{s['avg_tokens_per_sec']:<10.1f} {'-':<8} {s['memory_mb']:<10.1f}")

    print(f"{'='*80}")

    # Comparison
    configs = list(suite.summary.keys())
    if len(configs) >= 2:
        baseline = suite.summary[configs[0]]
        print(f"\nComparison vs {configs[0]}:")
        for name in configs[1:]:
            s = suite.summary[name]
            ttft_change = (s['avg_ttft_ms'] / baseline['avg_ttft_ms'] - 1) * 100
            speed_change = (s['avg_tokens_per_sec'] / max(baseline['avg_tokens_per_sec'], 0.01) - 1) * 100
            mem_change = s['memory_mb'] - baseline['memory_mb']
            print(f"  {name}: TTFT {ttft_change:+.1f}%, Speed {speed_change:+.1f}%, "
                  f"Mem {mem_change:+.1f}MB")


def _save_results(suite: BenchmarkSuite):
    report_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "reports")
    os.makedirs(report_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(report_dir, "benchmark_results.json")
    data = {
        'results': [
            {
                'name': r.name, 'model': r.model, 'prompt': r.prompt,
                'ttft_ms': r.ttft_ms, 'total_ms': r.total_ms,
                'tokens_generated': r.tokens_generated, 'tokens_per_sec': r.tokens_per_sec,
                'memory_mb': r.memory_used_mb,
            }
            for r in suite.results
        ],
        'summary': suite.summary,
    }
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Results saved to {json_path}")

    # CSV
    csv_path = os.path.join(report_dir, "benchmark_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['config', 'model', 'prompt', 'ttft_ms', 'total_ms',
                         'tokens_generated', 'tokens_per_sec', 'memory_mb'])
        for r in suite.results:
            writer.writerow([r.name, r.model, r.prompt, f"{r.ttft_ms:.1f}",
                             f"{r.total_ms:.1f}", r.tokens_generated,
                             f"{r.tokens_per_sec:.1f}", f"{r.memory_used_mb:.1f}"])
    logger.info(f"Results saved to {csv_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TurboBench — LLM Inference Benchmark")
    parser.add_argument("--model", default="gpt2", help="Model name")
    parser.add_argument("--device", default="auto", help="Device (auto/cpu/cuda/mps)")
    parser.add_argument("--runs", type=int, default=3, help="Runs per prompt")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    args = parser.parse_args()

    run_benchmark(
        model_name=args.model,
        device=args.device,
        num_runs=args.runs,
        max_tokens=args.max_tokens,
    )
