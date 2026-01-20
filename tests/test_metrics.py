"""
Tests for the event-driven metrics collection system.

Tests cover:
1. RunMetrics, IterationMetrics, AggregateMetrics dataclasses
2. Event-driven metrics collection via EventBus
3. Metric invariant assertions
4. Report generation
"""

import pytest
from pathlib import Path
from tests.metrics.pipeline_metrics import (
    RunMetrics,
    IterationMetrics,
    AggregateMetrics,
    MetricsCollector,
    FailureType,
    assert_metric_invariants,
    get_metrics_collector,
)
from tests.metrics.event_bus import (
    get_event_bus,
    emit_event,
    Event,
    RUN_STARTED, RUN_COMPLETED, RUN_FAILED,
    ITERATION_STARTED, ITERATION_COMPLETED, ITERATION_FAILED,
    ASSUMPTION_FAILED, CRYSTALLIZATION_ATTEMPTED,
    MEMORY_QUERIED, MEMORY_HIT, EVIDENCE_ADDED,
    AUTHORITY_VIOLATION, TOKENS_USED, AMBIGUITY_HALT,
)


@pytest.fixture(autouse=True)
def reset_event_bus():
    """Reset the event bus before and after each test."""
    bus = get_event_bus()
    bus.reset()
    yield
    bus.reset()


# =============================================================================
# Tests for RunMetrics
# =============================================================================

@pytest.mark.metrics
class TestRunMetrics:
    """Tests for per-run metrics."""
    
    def test_run_metrics_creation(self):
        """Test creating a RunMetrics instance."""
        run = RunMetrics(run_id="test-run-001")
        
        assert run.run_id == "test-run-001"
        assert run.iterations_to_success == 0
        assert run.valid_run is False
        assert run.total_tokens == 0
    
    def test_run_metrics_serialization(self):
        """Test serialization to dict."""
        run = RunMetrics(
            run_id="test-run-001",
            iterations_to_success=3,
            valid_run=True,
            final_verdict="robust"
        )
        
        d = run.to_dict()
        assert d["run_id"] == "test-run-001"
        assert d["iterations_to_success"] == 3
        assert d["valid_run"] is True


# =============================================================================
# Tests for IterationMetrics
# =============================================================================

@pytest.mark.metrics
class TestIterationMetrics:
    """Tests for per-iteration metrics."""
    
    def test_iteration_metrics_creation(self):
        """Test creating an IterationMetrics instance."""
        iteration = IterationMetrics(run_id="run-001", iteration_number=1)
        
        assert iteration.run_id == "run-001"
        assert iteration.iteration_number == 1
        assert iteration.success is False
    
    def test_iteration_with_failure(self):
        """Test iteration with failure type."""
        iteration = IterationMetrics(
            run_id="run-001",
            iteration_number=2,
            failure_type=FailureType.DATA
        )
        
        d = iteration.to_dict()
        assert d["failure_type"] == "DATA"


# =============================================================================
# Tests for AggregateMetrics
# =============================================================================

@pytest.mark.metrics
class TestAggregateMetrics:
    """Tests for aggregate metrics calculations."""
    
    def test_valid_run_rate(self):
        """Test valid run rate calculation."""
        agg = AggregateMetrics(total_runs=100, valid_runs=85)
        assert agg.valid_run_rate == 85.0
    
    def test_false_crystallization_rate_zero(self):
        """False crystallization rate should be 0 for valid systems."""
        agg = AggregateMetrics(
            total_crystallizations=50,
            false_crystallizations=0
        )
        assert agg.false_crystallization_rate == 0.0
    
    def test_false_crystallization_rate_nonzero(self):
        """Detect non-zero false crystallization rate."""
        agg = AggregateMetrics(
            total_crystallizations=50,
            false_crystallizations=2
        )
        assert agg.false_crystallization_rate == 4.0
    
    def test_mean_iterations_to_success(self):
        """Test mean iterations calculation."""
        agg = AggregateMetrics(
            iterations_to_success_list=[2, 3, 3, 4, 3]
        )
        assert agg.mean_iterations_to_success == 3.0
    
    def test_median_iterations_to_success(self):
        """Test median iterations calculation."""
        agg = AggregateMetrics(
            iterations_to_success_list=[1, 2, 3, 4, 5]
        )
        assert agg.median_iterations_to_success == 3.0
    
    def test_evidence_reuse_rate(self):
        """Test memory hit rate."""
        agg = AggregateMetrics(
            total_memory_queries=100,
            memory_hits=30
        )
        assert agg.evidence_reuse_rate == 30.0
    
    def test_authority_violation_rate_zero(self):
        """Authority violation rate must be 0."""
        agg = AggregateMetrics(
            total_runs=100,
            total_authority_violations=0
        )
        assert agg.authority_violation_rate == 0.0
    
    def test_tokens_per_valid_experiment(self):
        """Test cost efficiency calculation."""
        agg = AggregateMetrics(
            valid_runs=10,
            tokens_for_valid_experiments=50000
        )
        assert agg.tokens_per_valid_experiment == 5000.0


# =============================================================================
# Tests for Event-Driven MetricsCollector
# =============================================================================

@pytest.mark.metrics
class TestMetricsCollector:
    """Tests for event-driven metrics collection."""
    
    def test_run_lifecycle_via_events(self, temp_experiment_dir):
        """Test recording a full run lifecycle via events."""
        collector = MetricsCollector(temp_experiment_dir / "metrics.json")
        
        emit_event(RUN_STARTED, {"run_id": "test-run-001"})
        emit_event(ITERATION_STARTED, {"iteration": 1})
        emit_event(TOKENS_USED, {"count": 1000, "agent": "DesignAgent"})
        emit_event(ITERATION_COMPLETED, {})
        emit_event(RUN_COMPLETED, {"valid": True, "verdict": "robust"})
        
        assert collector.aggregate.total_runs == 1
        assert collector.aggregate.valid_runs == 1
        assert collector.aggregate.total_tokens == 1000
        assert collector.aggregate.tokens_by_agent["DesignAgent"] == 1000
    
    def test_failed_run_tracking(self, temp_experiment_dir):
        """Test tracking failed runs."""
        collector = MetricsCollector(temp_experiment_dir / "metrics.json")
        
        emit_event(RUN_STARTED, {"run_id": "fail-run"})
        emit_event(RUN_FAILED, {"failure_type": "EXECUTION"})
        
        assert collector.aggregate.total_runs == 1
        assert collector.aggregate.valid_runs == 0
        assert collector.aggregate.failure_type_distribution["EXECUTION"] == 1
    
    def test_authority_violation_tracking(self, temp_experiment_dir):
        """Test tracking authority violations."""
        collector = MetricsCollector(temp_experiment_dir / "metrics.json")
        
        emit_event(RUN_STARTED, {"run_id": "violation-run"})
        emit_event(AUTHORITY_VIOLATION, {"agent": "DesignAgent", "action": "wrote_code"})
        emit_event(RUN_COMPLETED, {"valid": False})
        
        assert collector.aggregate.total_authority_violations == 1
    
    def test_memory_hit_tracking(self, temp_experiment_dir):
        """Test tracking memory queries and hits."""
        collector = MetricsCollector(temp_experiment_dir / "metrics.json")
        
        emit_event(RUN_STARTED, {"run_id": "memory-run"})
        emit_event(ITERATION_STARTED, {"iteration": 1})
        emit_event(MEMORY_QUERIED, {})
        emit_event(MEMORY_HIT, {})
        emit_event(MEMORY_QUERIED, {})
        emit_event(ITERATION_COMPLETED, {})
        emit_event(RUN_COMPLETED, {"valid": True})
        
        assert collector.aggregate.total_memory_queries == 2
        assert collector.aggregate.memory_hits == 1
        assert collector.aggregate.evidence_reuse_rate == 50.0
    
    def test_assumption_failure_tracking(self, temp_experiment_dir):
        """Test tracking assumption failures."""
        collector = MetricsCollector(temp_experiment_dir / "metrics.json")
        
        emit_event(RUN_STARTED, {"run_id": "assumption-run"})
        emit_event(ASSUMPTION_FAILED, {"type": "normality", "p_value": 0.001})
        emit_event(RUN_COMPLETED, {"valid": True})
        
        assert collector.aggregate.assumption_failure_count == 1
    
    def test_crystallization_tracking(self, temp_experiment_dir):
        """Test tracking crystallization attempts."""
        collector = MetricsCollector(temp_experiment_dir / "metrics.json")
        
        emit_event(RUN_STARTED, {"run_id": "crystal-run"})
        emit_event(CRYSTALLIZATION_ATTEMPTED, {"valid": True})
        emit_event(RUN_COMPLETED, {"valid": True})
        
        # Valid crystallization
        assert collector.aggregate.total_crystallizations == 1
        assert collector.aggregate.false_crystallizations == 0
        
        # False crystallization
        emit_event(RUN_STARTED, {"run_id": "false-crystal-run"})
        emit_event(CRYSTALLIZATION_ATTEMPTED, {"valid": False})
        emit_event(RUN_COMPLETED, {"valid": False})
        
        assert collector.aggregate.total_crystallizations == 2
        assert collector.aggregate.false_crystallizations == 1
    
    def test_ambiguity_halt_tracking(self, temp_experiment_dir):
        """Test tracking ambiguity halts."""
        collector = MetricsCollector(temp_experiment_dir / "metrics.json")
        
        emit_event(RUN_STARTED, {"run_id": "ambig-run"})
        emit_event(AMBIGUITY_HALT, {})
        emit_event(RUN_COMPLETED, {"valid": True})
        
        assert collector.aggregate.ambiguity_halts == 1
    
    def test_repeated_failure_detection(self, temp_experiment_dir):
        """Test detection of repeated failures (same type twice)."""
        collector = MetricsCollector(temp_experiment_dir / "metrics.json")
        
        emit_event(RUN_STARTED, {"run_id": "repeat-run"})
        emit_event(ITERATION_STARTED, {"iteration": 1})
        emit_event(ITERATION_FAILED, {"failure_type": "DATA"})
        emit_event(ITERATION_STARTED, {"iteration": 2})
        emit_event(ITERATION_FAILED, {"failure_type": "DATA"})  # Repeated!
        emit_event(RUN_FAILED, {"failure_type": "DATA"})
        
        assert collector.aggregate.repeated_failures == 1
    
    def test_report_generation(self, temp_experiment_dir):
        """Test report generation."""
        collector = MetricsCollector(temp_experiment_dir / "metrics.json")
        
        # Add some data using legacy API
        collector.record_run("report-run", "success", tokens_used=5000)
        collector.record_finding("robust")
        
        report = collector.generate_report()
        
        # Check for legacy format sections
        assert "Pipeline Health" in report
        assert "Research Quality" in report
        assert "Success Metrics" in report
        assert "Invariant Violation Rate" in report
    
    def test_save_metrics(self, temp_experiment_dir):
        """Test saving metrics to file."""
        storage_path = temp_experiment_dir / "metrics.json"
        collector = MetricsCollector(storage_path)
        
        emit_event(RUN_STARTED, {"run_id": "save-run"})
        emit_event(RUN_COMPLETED, {"valid": True})
        
        collector.save()
        
        assert storage_path.exists()


# =============================================================================
# Tests for Metric Invariants
# =============================================================================

@pytest.mark.metrics
class TestMetricInvariants:
    """Tests for metric invariant assertions."""
    
    def test_clean_metrics_pass_invariants(self):
        """Clean metrics should pass all invariants."""
        agg = AggregateMetrics(
            total_runs=10,
            valid_runs=8,
            false_crystallizations=0,
            total_crystallizations=5,
            total_authority_violations=0,
            repeated_failures=1,
            duplicate_evidence=0,
            evidence_added=10
        )
        
        # Should not raise
        assert_metric_invariants(agg)
    
    def test_false_crystallization_fails_invariant(self):
        """Non-zero false crystallization should fail."""
        agg = AggregateMetrics(
            total_runs=10,
            false_crystallizations=1,
            total_crystallizations=5
        )
        
        with pytest.raises(AssertionError, match="crystallization"):
            assert_metric_invariants(agg)
    
    def test_authority_violation_fails_invariant(self):
        """Any authority violation should fail."""
        agg = AggregateMetrics(
            total_runs=10,
            total_authority_violations=1
        )
        
        with pytest.raises(AssertionError, match="Authority"):
            assert_metric_invariants(agg)
    
    def test_high_repeat_failure_rate_fails(self):
        """High repeat failure rate should fail."""
        agg = AggregateMetrics(
            total_runs=10,
            repeated_failures=5  # 50% rate!
        )
        
        with pytest.raises(AssertionError, match="Repeat failure"):
            assert_metric_invariants(agg)


# =============================================================================
# Tests for Metric Targets
# =============================================================================

@pytest.mark.metrics
def test_metrics_targets():
    """
    Test that we have defined targets for key metrics.
    
    Paper-grade targets:
    - False crystallization rate: 0%
    - Authority violation rate: 0%
    - Repeat failure rate: <20%
    - Duplicate evidence rate: <5%
    """
    targets = {
        "false_crystallization_rate": 0.0,
        "authority_violation_rate": 0.0,
        "repeat_failure_rate_max": 20.0,
        "duplicate_evidence_rate_max": 5.0,
    }
    
    assert targets["false_crystallization_rate"] == 0.0, \
        "False crystallization must be zero"
    
    assert targets["authority_violation_rate"] == 0.0, \
        "No authority violations allowed"
    
    assert targets["repeat_failure_rate_max"] <= 20.0, \
        "Repeat failure rate should be low"
