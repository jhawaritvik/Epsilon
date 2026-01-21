# Epsilon Benchmark Evaluation Suite

This directory contains benchmark evaluation scripts to validate Epsilon as a research-grade AI Scientist architecture.

## Quick Start

```bash
# Activate your environment
source .venv/bin/activate

# Run internal validation (no GPU required)
python benchmarks/run_eval.py --suite internal

# Run MLAgentBench evaluation (GPU recommended for some tasks)
python benchmarks/run_eval.py --suite mlagentbench --tasks 5

# Run full scientific discovery benchmark
python benchmarks/run_eval.py --suite science --tasks 10
```

## Benchmark Suites

| Suite | Description | GPU Required | Approx. Cost |
|-------|-------------|--------------|--------------|
| `internal` | Validates Epsilon's core functionality | ❌ | ~$1-2 |
| `mlagentbench` | MLAgentBench CPU-friendly subset | ⚠️ Optional | ~$5-10 |
| `science` | ScienceAgentBench-inspired tasks | ❌ | ~$3-5 |

## Files

- `run_eval.py` - Main benchmark runner
- `tasks/` - Benchmark task definitions
- `results/` - Evaluation results (auto-generated)
