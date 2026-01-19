# Testing Framework - Quick Reference

## What Was Built

A comprehensive, compiler-style testing framework for Epsilon with **120 tests** across **11 categories**.

## Directory Structure

```
tests/
├── conftest.py                          # Pytest fixtures
├── invariants.py                        # Core invariant validators
├── README.md                            # Full documentation
│
├── test_invariants.py                   # 23 tests - Invariant validation
├── test_golden.py                       # 4 tests - Golden cases
├── test_determinism.py                  # 7 tests - Reproducibility
├── test_memory_behavior.py              # 15 tests - Memory system
├── test_evaluation_sanity.py            # 6 tests - Evaluation agent
├── test_execution_stress.py             # 14 tests - Execution agent
├── test_report_validation.py            # 10 tests - Report quality
├── test_metrics.py                      # 11 tests - Metrics collection
├── test_litmus.py                       # 3 tests - Trust validation
│
├── golden/                              # Golden test specs
│   ├── test_l2_vs_pruning.yaml
│   ├── test_depth_vs_width.yaml
│   └── test_label_noise.yaml
│
├── adversarial/                         # Adversarial tests
│   └── test_constraint_violations.py    # 20 tests
│
└── metrics/                             # Metrics system
    └── pipeline_metrics.py
```

## Quick Start

### Install Dependencies
```bash
pip install pytest pytest-cov pyyaml
```

### Run All Tests
```bash
pytest tests/ -v
```

### Run By Category
```bash
# Critical tests (must pass before deploy)
pytest tests/ -m "invariant or integration" -v

# Fast tests only
pytest tests/ -m "not slow" -v

# Specific category
pytest tests/ -m adversarial -v
```

## Test Categories

| Category | Tests | Purpose |
|----------|-------|---------|
| Invariant | 23 | Validate system rules |
| Golden | 4 | Known research goals |
| Adversarial | 20 | Try to break system |
| Determinism | 7 | Reproducibility |
| Memory | 15 | Memory correctness |
| Evaluation | 6 | Statistical sanity |
| Execution | 14 | Stress testing |
| Report | 10 | Report quality |
| Metrics | 11 | Metrics accuracy |
| Integration | 3 | End-to-end trust |

**Total: 120 tests**

## Key Invariants

1. **Design Authority** - Design Agent never writes code
2. **Execution Authority** - Execution Agent never changes hypotheses
3. **Evaluation Authority** - Evaluation Agent never runs training
4. **Data Modality** - Tabular→tabular, vision→vision (no silent switching)
5. **Artifact Completeness** - All required files present
6. **Memory Correctness** - No duplicates, robust-only crystallization

## Metrics Targets

- Invariant Violations: **0%** (CRITICAL)
- False Discovery Rate: **<5%** (CRITICAL)
- Success Rate: **>70%**
- Reproducibility: **>80%**
- Memory Hit Rate: **>20%**

## Verification Results

```
Collected: 120 tests in 0.66s
Sample Run: 22 passed, 1 failed (95.7% pass rate)
```

## Trust Levels

- ✅ **FULL TRUST** - All checks passed
- ⚠️ **TRUST WITH CAUTION** - Minor issues
- ❌ **LIMITED TRUST** - Significant issues  
- 🚫 **DO NOT TRUST** - Critical failures

## The Litmus Test

**Question:** "Would you trust this system's conclusions without re-running everything?"

**Answer (if tests pass):** "Yes, with logs."

## Links

- [Full Documentation](file:///c:/Python/Epsilon/tests/README.md)
- [Implementation Plan](file:///C:/Users/LENOVO/.gemini/antigravity/brain/bae90225-1796-457d-8588-79d19da4a950/implementation_plan.md)
- [Walkthrough](file:///C:/Users/LENOVO/.gemini/antigravity/brain/bae90225-1796-457d-8588-79d19da4a950/walkthrough.md)
