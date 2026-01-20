# Epsilon Testing Framework

## Overview

This testing framework implements a **compiler-style** testing methodology for the Epsilon research pipeline. Unlike traditional unit tests that focus on individual functions, these tests verify **system-level invariants** and **research validity**.

## Philosophy

> "If you hand this system to a PhD student, would you trust the conclusions without re-running everything manually?"

The testing framework is designed to answer this question with "YES, WITH LOGS".

## Test Organization

### Phase 1: Foundation

#### `conftest.py`
Pytest fixtures and test utilities:
- Test isolation (temporary directories)
- Mock Supabase connections
- Sample data generators
- Helper assertion functions

#### `invariants.py`
Core invariant validators:
- **Design Authority**: Ensures Design Agent never writes executable code
- **Execution Authority**: Ensures Execution Agent never modifies hypotheses
- **Evaluation Authority**: Ensures Evaluation Agent never executes training
- **Data Modality**: Enforces tabular→tabular, vision→vision consistency
- **Artifact Completeness**: Verifies all required outputs exist
- **Memory Correctness**: Validates memory insertion rules

#### `test_invariants.py`
Unit tests for all invariant validators.

---

### Phase 2: Core Tests

#### `golden/`
Directory containing golden test specifications (YAML files):
- `test_l2_vs_pruning.yaml` - Regularization comparison
- `test_depth_vs_width.yaml` -Architecture comparison (vision)
- `test_label_noise.yaml` - Calibration study

#### `test_golden.py`
Golden test runner that verifies the system produces correct, deterministic results for known research goals.

#### `adversarial/`
Adversarial test cases that deliberately try to break the system.

#### `test_constraint_violations.py`
Tests for:
- Modality traps (e.g., "Use CIFAR-10 for house price prediction")
- Ambiguous specifications (system should halt, not guess)
- Forbidden behaviors (authority violations)
- Silent fallbacks (should fail explicitly, not silently switch datasets)
- Edge cases (empty goals, special characters, etc.)

#### `test_determinism.py`
Verifies reproducible behavior:
- Same dataset selection
- Same model family
- Same statistical plan
- Same artifact structure
- Structural hashing utilities

#### `test_memory_behavior.py`
Tests for memory system:
- Duplicate suppression
- Failure learning (past failures inform future designs)
- Knowledge crystallization (only from robust verdicts)
- Query functionality

---

### Phase 3: Agent-Specific Tests

#### `test_evaluation_sanity.py`
Evaluation Agent sanity checks:
- Identical distributions → H0 not rejected
- Different distributions → H0 rejected
- Zero-variance handling (graceful)
- Known effect sizes classified correctly
- No false "robust" classifications
- Assumption violations trigger fallback tests

#### `test_execution_stress.py`
Execution Agent stress tests:
- Missing dependencies (auto-install)
- Long runtimes (timeout with partial artifacts)
- Malformed code (retry with corrections)
- Resource exhaustion (clean failure)
- Error recovery (install → retry → success)

#### `test_report_validation.py`
Report generator validation:
- No broken links
- Embedded plots render
- JSON blocks are valid
- Narrative doesn't contradict raw_results.json
- All required sections present

---

### Phase 4: Metrics & Integration

#### `metrics/pipeline_metrics.py`
Metrics collection system:

**Pipeline Metrics:**
- % runs that halt correctly on ambiguity
- % runs that violate invariants (target: 0%)
- Mean retries per execution
- Mean tokens per run
- Memory hit rate

**Research Metrics:**
- Reproducibility rate
- False discovery rate (target: <5%)
- Failure recovery success rate

#### `test_metrics.py`
Tests for metrics calculation and collection.

#### `test_litmus.py`
**The ultimate test**: End-to-end integration test that answers:
> "Would you trust this system's conclusions?"

Generates a trust report with:
- Invariant compliance
- Artifact completeness
- Statistical validity
- Reproducibility
- Audit trail clarity
- Internal consistency

---

### Phase 5: Petri Fuzz Testing (Optional)

#### `petri/`
Behavioral fuzz testing for **instruction brittleness** using Anthropic's Petri methodology.

**Purpose**: Tests how well agents resist adversarial prompts that attempt to:
- **Instruction boundary erosion** - Blur agent authority
- **Prompt-induced authority creep** - Exceed contractual limits
- **Tool misuse** - Call forbidden tools

**Components**:
- `special_instructions.py` - 30+ adversarial probes per agent type
- `violation_scorer.py` - Pattern-based violation detection
- `conftest.py` - Fixtures, transcript recording, agent wrappers
- `test_instruction_fuzz.py` - Offline scorer tests + live brittleness tests

**Running Petri Tests**:
```bash
# Offline tests (no API calls, pattern matching only)
pytest tests/petri/test_instruction_fuzz.py::TestViolationScorer -v

# Live fuzz tests (requires OpenAI API key, incurs costs)
pytest tests/petri/ -m petri --petri-run -v
```

**Transcripts**: Saved to `tests/petri/transcripts/` for review.

---

## Running Tests


### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Categories
```bash
# Invariant tests
pytest tests/ -m invariant -v

# Golden tests
pytest tests/ -m golden -v

# Adversarial tests
pytest tests/ -m adversarial -v

# Determinism tests
pytest tests/ -m determinism -v

# Memory tests
pytest tests/ -m memory -v

# Agent-specific tests
pytest tests/ -m agent -v

# Metrics tests
pytest tests/ -m metrics -v

# Integration/Litmus tests
pytest tests/ -m integration -v
```

### Run Fast Tests Only (exclude slow)
```bash
pytest tests/ -v -m "not slow"
```

### Run with Coverage
```bash
pytest tests/ -v --cov=. --cov-report=html
```

## Test Markers

Available markers (defined in `pytest.ini`):
- `invariant` - Tests that validate core system invariants
- `golden` - Golden tests with known expected behavior
- `adversarial` - Tests that deliberately try to break the system
- `determinism` - Tests that verify reproducible behavior
- `memory` - Tests for memory system behavior
- `agent` - Agent-specific validation tests
- `stress` - Stress and edge case tests
- `metrics` - Metrics collection validation
- `integration` - End-to-end integration tests
- `slow` - Tests that take significant time to run
- `petri` - Petri fuzz tests for instruction brittleness (requires `--petri-run`)

## Interpreting Results

### Critical Failures

If any of these fail, **DO NOT DEPLOY**:
- `test_invariants.py` - Invariant validators broken
- `test_litmus.py` - System not trustworthy
- Any test showing `invariant_violation_rate > 0%`

### Important Failures

These indicate problems that should be fixed soon:
- `test_golden.py` - System producing incorrect results
- `test_determinism.py` - Non-reproducible behavior
- `test_memory_behavior.py` - Memory system not working properly

### Informational Failures

These indicate areas for improvement:
- `test_adversarial.py` - System vulnerable to edge cases
- `test_metrics.py` - Metrics collection issues
- `test_report_validation.py` - Report quality issues

## Metrics Targets

The testing framework enforces these targets:

| Metric | Target | Severity |
|--------|--------|----------|
| Invariant Violation Rate | 0% | Critical |
| False Discovery Rate | <5% | Critical |
| Success Rate | >70% | Important |
| Reproducibility Rate | >80% | Important |
| Memory Hit Rate | >20% | Informational |

## Adding New Tests

### Adding a Golden Test

1. Create a YAML file in `tests/golden/`:
```yaml
goal: "Your research question"
expected:
  data_modality: tabular
  dataset_family: sklearn
  execution_mode: validation
  min_artifacts: 6
  allowed_failure_types: []
```

2. The test runner will automatically pick it up.

### Adding an Invariant

1. Create a new validator class in `tests/invariants.py` inheriting from `InvariantValidator`
2. Implement the `validate()` method
3. Add it to `InvariantChecker`
4. Add tests in `test_invariants.py`

### Adding a Metric

1. Add the metric to `PipelineMetrics` or `ResearchMetrics` in `tests/metrics/pipeline_metrics.py`
2. Add a property for computed values
3. Add tests in `test_metrics.py`

## Continuous Integration

To set up CI (GitHub Actions, etc.):

```yaml
- name: Run Tests
  run: |
    pip install pytest pytest-cov
    pytest tests/ -v --cov=. --cov-report=xml

- name: Check Invariant Violations
  run: |
    # Parse test output for invariant violations
    # Fail build if any are found
```

## Debugging Test Failures

### Enable Verbose Logging
```bash
pytest tests/ -v -s --log-cli-level=DEBUG
```

### Run Single Test
```bash
pytest tests/test_invariants.py::TestDesignAuthorityValidator::test_valid_design_output -v
```

### Inspect Fixtures
```bash
pytest tests/ --fixtures
```

## Philosophy: Why Test Like This?

### Traditional Testing
- Tests individual functions
- Mocks everything
- Fast but shallow
- Misses integration issues

### Compiler-Style Testing
- Tests system invariants
- Uses real components where possible
- Slower but comprehensive
- Catches architectural issues

**Epsilon is more like a compiler than an app**. We test it accordingly.

### The Litmus Test

The litmus test (`test_litmus.py`) is the culmination of this philosophy. It asks:

> "If I handed this system to a PhD student, would I trust the conclusions without re-running everything manually?"

If the answer is:
- **"No"** - The system needs fundamental fixes
- **"Almost"** - You're very close
- **"Yes, with logs"** - You've built something rare

The logs (audit trails) allow verification without re-execution, which is the mark of a trustworthy research system.

## Contributing

When adding new features to Epsilon:

1. **First**: Define the invariants (what must NEVER happen)
2. **Second**: Write tests that enforce those invariants
3. **Third**: Implement the feature
4. **Fourth**: Verify with litmus test

This order ensures correctness by construction.

## Questions?

See the main Epsilon README for architecture details and the testing_eval.txt for the original testing requirements that inspired this framework.
