"""
Tests for metrics collection system.
"""

import pytest
from pathlib import Path
from tests.metrics.pipeline_metrics import (
    PipelineMetrics,
    ResearchMetrics,
    MetricsCollector,
)


@pytest.mark.metrics
class TestPipelineMetrics:
    """Tests for pipeline metrics calculations."""
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = PipelineMetrics(
            total_runs=100,
            successful_runs=85,
            failed_runs=15
        )
        
        assert metrics.success_rate == 85.0
    
    def test_invariant_violation_rate(self):
        """Test invariant violation rate (should be 0%)."""
        metrics = PipelineMetrics(
            total_runs=50,
            invariant_violations=0
        )
        
        assert metrics.invariant_violation_rate == 0.0
    
    def test_invariant_violation_detected(self):
        """Even one violation should show up."""
        metrics = PipelineMetrics(
            total_runs=100,
            invariant_violations=2
        )
        
        assert metrics.invariant_violation_rate == 2.0
    
    def test_memory_hit_rate(self):
        """Test memory hit rate calculation."""
        metrics = PipelineMetrics(
            memory_queries=50,
            memory_hits=40
        )
        
        assert metrics.memory_hit_rate == 80.0
    
    def test_mean_tokens_per_run(self):
        """Test average tokens calculation."""
        metrics = PipelineMetrics(
            total_runs=10,
            total_tokens_used=50000
        )
        
        assert metrics.mean_tokens_per_run == 5000.0


@pytest.mark.metrics
class TestResearchMetrics:
    """Tests for research quality metrics."""
    
    def test_reproducibility_rate(self):
        """Test reproducibility rate calculation."""
        metrics = ResearchMetrics(
            reproduction_attempts=20,
            reproduced_findings=18
        )
        
        assert metrics.reproducibility_rate == 90.0
    
    def test_false_discovery_rate(self):
        """Test FDR calculation (should be low)."""
        metrics = ResearchMetrics(
            robust_findings=100,
            contradicted_findings=3
        )
        
        assert metrics.false_discovery_rate == 3.0
        assert metrics.false_discovery_rate < 5.0  # Target
    
    def test_high_fdr_is_problematic(self):
        """High FDR should be flagged."""
        metrics = ResearchMetrics(
            robust_findings=50,
            contradicted_findings=10  # 20% FDR!
        )
        
        assert metrics.false_discovery_rate == 20.0
        assert metrics.false_discovery_rate > 5.0  # Above target
    
    def test_failure_recovery_rate(self):
        """Test failure recovery rate."""
        metrics = ResearchMetrics(
            failures=25,
            recovered_failures=20
        )
        
        assert metrics.failure_recovery_rate == 80.0
    
    def test_robust_finding_rate(self):
        """Test percentage of findings that are robust."""
        metrics = ResearchMetrics(
            total_findings=100,
            robust_findings=70,
            spurious_findings=20,
            promising_findings=10
        )
        
        assert metrics.robust_finding_rate == 70.0


@pytest.mark.metrics
class TestMetricsCollector:
    """Tests for metrics collector."""
    
    def test_record_successful_run(self, temp_experiment_dir):
        """Test recording a successful run."""
        collector = MetricsCollector(temp_experiment_dir / "metrics.json")
        
        collector.record_run(
            run_id="run_001",
            status="success",
            retries=1,
            tokens_used=5000,
            invariant_violations=0
        )
        
        assert collector.pipeline_metrics.total_runs == 1
        assert collector.pipeline_metrics.successful_runs == 1
        assert collector.pipeline_metrics.mean_tokens_per_run == 5000.0
    
    def test_record_failed_run(self, temp_experiment_dir):
        """Test recording a failed run."""
        collector = MetricsCollector(temp_experiment_dir / "metrics.json")
        
        collector.record_run(
            run_id="run_002",
            status="failed",
            retries=3,
            tokens_used=8000
        )
        
        assert collector.pipeline_metrics.failed_runs == 1
        assert collector.pipeline_metrics.mean_retries_per_run == 3.0
    
    def test_record_ambiguity_halt(self, temp_experiment_dir):
        """Test recording ambiguity halt."""
        collector = MetricsCollector(temp_experiment_dir / "metrics.json")
        
        collector.record_run(
            run_id="run_003",
            status="ambiguous",
            halted_on_ambiguity=True
        )
        
        assert collector.pipeline_metrics.ambiguity_halts == 1
    
    def test_record_memory_queries(self, temp_experiment_dir):
        """Test recording memory queries."""
        collector = MetricsCollector(temp_experiment_dir / "metrics.json")
        
        collector.record_memory_query(found_useful_info=True)
        collector.record_memory_query(found_useful_info=True)
        collector.record_memory_query(found_useful_info=False)
        
        assert collector.pipeline_metrics.memory_queries == 3
        assert collector.pipeline_metrics.memory_hits == 2
        assert collector.pipeline_metrics.memory_hit_rate == pytest.approx(66.67, rel=0.1)
    
    def test_record_findings(self, temp_experiment_dir):
        """Test recording research findings."""
        collector = MetricsCollector(temp_experiment_dir / "metrics.json")
        
        collector.record_finding("robust")
        collector.record_finding("robust")
        collector.record_finding("spurious")
        collector.record_finding("promising")
        
        assert collector.research_metrics.total_findings == 4
        assert collector.research_metrics.robust_findings == 2
        assert collector.research_metrics.spurious_findings == 1
    
    def test_save_and_load(self, temp_experiment_dir):
        """Test saving and loading metrics."""
        storage_path = temp_experiment_dir / "metrics.json"
        
        # Create and populate collector
        collector1 = MetricsCollector(storage_path)
        collector1.record_run("run_001", "success", tokens_used=1000)
        collector1.record_finding("robust")
        collector1.save()
        
        # Load into new collector
        collector2 = MetricsCollector(storage_path)
        collector2.load()
        
        # Should have same data
        assert collector2.pipeline_metrics.total_runs == 1
        assert collector2.research_metrics.robust_findings == 1
    
    def test_generate_report(self, temp_experiment_dir):
        """Test report generation."""
        collector = MetricsCollector(temp_experiment_dir / "metrics.json")
        
        # Add some data
        for i in range(10):
            collector.record_run(f"run_{i}", "success", tokens_used=5000)
        
        collector.record_finding("robust")
        collector.record_finding("robust")
        collector.record_finding("spurious")
        
        report = collector.generate_report()
        
        # Report should contain key sections
        assert "Pipeline Health" in report
        assert "Research Quality" in report
        assert "Success Rate" in report
        assert "Invariant Violation Rate" in report


@pytest.mark.metrics
def test_metrics_targets():
    """
    Test that we have defined targets for key metrics.
    
    Targets:
    - Invariant violation rate: 0%
    - False discovery rate: <5%
    - Reproducibility rate: >80%
    - Failure recovery rate: >60%
    """
    targets = {
        "invariant_violation_rate": 0.0,
        "false_discovery_rate_max": 5.0,
        "reproducibility_rate_min": 80.0,
        "failure_recovery_rate_min": 60.0,
    }
    
    # Verify targets are reasonable
    assert targets["invariant_violation_rate"] == 0.0, \
        "Invariant violations should never be acceptable"
    
    assert targets["false_discovery_rate_max"] < 10.0, \
        "FDR target should be strict"
    
    assert targets["reproducibility_rate_min"] > 50.0, \
        "Reproducibility should be majority"
