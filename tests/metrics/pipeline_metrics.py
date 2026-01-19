"""
Pipeline metrics collection and analysis.

Metrics that actually matter for a research pipeline:

Pipeline Metrics:
- % runs that halt correctly on ambiguity
- % runs that violate invariants (target: 0%)
- Mean retries per execution
- Mean tokens per run (budget control)
- Memory hit rate (reuse vs recompute)

Research Metrics:
- Reproducibility rate
- False discovery rate (robust verdicts later contradicted)
- Failure recovery success rate
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class PipelineMetrics:
    """Metrics for pipeline health and performance."""
    
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    ambiguity_halts: int = 0
    invariant_violations: int = 0
    
    total_retries: int = 0
    total_tokens_used: int = 0
    
    memory_queries: int = 0
    memory_hits: int = 0
    
    @property
    def success_rate(self) -> float:
        """Percentage of runs that completed successfully."""
        if self.total_runs == 0:
            return 0.0
        return (self.successful_runs / self.total_runs) * 100
    
    @property
    def ambiguity_halt_rate(self) -> float:
        """Percentage of runs that correctly halted on ambiguity."""
        if self.total_runs == 0:
            return 0.0
        return (self.ambiguity_halts / self.total_runs) * 100
    
    @property
    def invariant_violation_rate(self) -> float:
        """Percentage of runs with invariant violations (should be 0%)."""
        if self.total_runs == 0:
            return 0.0
        return (self.invariant_violations / self.total_runs) * 100
    
    @property
    def mean_retries_per_run(self) -> float:
        """Average number of retries per run."""
        if self.total_runs == 0:
            return 0.0
        return self.total_retries / self.total_runs
    
    @property
    def mean_tokens_per_run(self) -> float:
        """Average tokens used per run."""
        if self.total_runs == 0:
            return 0.0
        return self.total_tokens_used / self.total_runs
    
    @property
    def memory_hit_rate(self) -> float:
        """Percentage of memory queries that found useful information."""
        if self.memory_queries == 0:
            return 0.0
        return (self.memory_hits / self.memory_queries) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with computed properties."""
        d = asdict(self)
        d.update({
            "success_rate": self.success_rate,
            "ambiguity_halt_rate": self.ambiguity_halt_rate,
            "invariant_violation_rate": self.invariant_violation_rate,
            "mean_retries_per_run": self.mean_retries_per_run,
            "mean_tokens_per_run": self.mean_tokens_per_run,
            "memory_hit_rate": self.memory_hit_rate,
        })
        return d


@dataclass
class ResearchMetrics:
    """Metrics for research quality and validity."""
    
    total_findings: int = 0
    robust_findings: int = 0
    spurious_findings: int = 0
    promising_findings: int = 0
    
    reproduced_findings: int = 0
    reproduction_attempts: int = 0
    
    contradicted_findings: int = 0
    
    failures: int = 0
    recovered_failures: int = 0
    
    @property
    def reproducibility_rate(self) -> float:
        """Percentage of findings that were successfully reproduced."""
        if self.reproduction_attempts == 0:
            return 0.0
        return (self.reproduced_findings / self.reproduction_attempts) * 100
    
    @property
    def false_discovery_rate(self) -> float:
        """Percentage of robust verdicts later contradicted (should be low)."""
        if self.robust_findings == 0:
            return 0.0
        return (self.contradicted_findings / self.robust_findings) * 100
    
    @property
    def failure_recovery_rate(self) -> float:
        """Percentage of failures that were recovered."""
        if self.failures == 0:
            return 0.0
        return (self.recovered_failures / self.failures) * 100
    
    @property
    def robust_finding_rate(self) -> float:
        """Percentage of findings that are robust."""
        if self.total_findings == 0:
            return 0.0
        return (self.robust_findings / self.total_findings) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with computed properties."""
        d = asdict(self)
        d.update({
            "reproducibility_rate": self.reproducibility_rate,
            "false_discovery_rate": self.false_discovery_rate,
            "failure_recovery_rate": self.failure_recovery_rate,
            "robust_finding_rate": self.robust_finding_rate,
        })
        return d


class MetricsCollector:
    """Collects and aggregates metrics across runs."""
    
    def __init__(self, storage_path: Path = None):
        """
        Args:
            storage_path: Path to store metrics JSON
        """
        self.pipeline_metrics = PipelineMetrics()
        self.research_metrics = ResearchMetrics()
        self.storage_path = storage_path or Path("metrics.json")
        self.run_history: List[Dict[str, Any]] = []
    
    def record_run(self, 
                   run_id: str,
                   status: str,
                   retries: int = 0,
                   tokens_used: int = 0,
                   invariant_violations: int = 0,
                   halted_on_ambiguity: bool = False):
        """
        Records a single run.
        
        Args:
            run_id: Unique run identifier
            status: "success", "failed", or "ambiguous"
            retries: Number of retry attempts
            tokens_used: Total tokens consumed
            invariant_violations: Number of invariant violations
            halted_on_ambiguity: Whether run halted due to ambiguity
        """
        self.pipeline_metrics.total_runs += 1
        
        if status == "success":
            self.pipeline_metrics.successful_runs += 1
        elif status == "failed":
            self.pipeline_metrics.failed_runs += 1
        
        if halted_on_ambiguity:
            self.pipeline_metrics.ambiguity_halts += 1
        
        self.pipeline_metrics.invariant_violations += invariant_violations
        self.pipeline_metrics.total_retries += retries
        self.pipeline_metrics.total_tokens_used += tokens_used
        
        # Record in history
        self.run_history.append({
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "retries": retries,
            "tokens_used": tokens_used,
            "invariant_violations": invariant_violations,
            "halted_on_ambiguity": halted_on_ambiguity,
        })
    
    def record_memory_query(self, found_useful_info: bool):
        """Records a memory query."""
        self.pipeline_metrics.memory_queries += 1
        if found_useful_info:
            self.pipeline_metrics.memory_hits += 1
    
    def record_finding(self, verdict: str):
        """
        Records a research finding.
        
        Args:
            verdict: "robust", "spurious", "promising", etc.
        """
        self.research_metrics.total_findings += 1
        
        if verdict == "robust" or verdict == "very_robust":
            self.research_metrics.robust_findings += 1
        elif verdict == "spurious":
            self.research_metrics.spurious_findings += 1
        elif verdict == "promising":
            self.research_metrics.promising_findings += 1
    
    def record_reproduction_attempt(self, successful: bool):
        """Records a reproduction attempt."""
        self.research_metrics.reproduction_attempts += 1
        if successful:
            self.research_metrics.reproduced_findings += 1
    
    def record_contradiction(self):
        """Records a finding that was later contradicted."""
        self.research_metrics.contradicted_findings += 1
    
    def record_failure_recovery(self, recovered: bool):
        """Records a failure and whether it was recovered."""
        self.research_metrics.failures += 1
        if recovered:
            self.research_metrics.recovered_failures += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Returns a summary of all metrics."""
        return {
            "pipeline_metrics": self.pipeline_metrics.to_dict(),
            "research_metrics": self.research_metrics.to_dict(),
            "collected_at": datetime.now().isoformat(),
        }
    
    def save(self):
        """Saves metrics to JSON file."""
        with open(self.storage_path, 'w') as f:
            json.dump({
                "summary": self.get_summary(),
                "run_history": self.run_history,
            }, f, indent=2)
    
    def load(self):
        """Loads metrics from JSON file."""
        if not self.storage_path.exists():
            return
        
        with open(self.storage_path, 'r') as f:
            data = json.load(f)
            self.run_history = data.get("run_history", [])
            
            # Reconstruct metrics from history
            for run in self.run_history:
                self.record_run(
                    run_id=run["run_id"],
                    status=run["status"],
                    retries=run.get("retries", 0),
                    tokens_used=run.get("tokens_used", 0),
                    invariant_violations=run.get("invariant_violations", 0),
                    halted_on_ambiguity=run.get("halted_on_ambiguity", False)
                )
    
    def generate_report(self) -> str:
        """Generates a human-readable metrics report."""
        summary = self.get_summary()
        pm = summary["pipeline_metrics"]
        rm = summary["research_metrics"]
        
        report = f"""
# Epsilon Pipeline Metrics Report
Generated: {summary["collected_at"]}

## Pipeline Health

### Success Metrics
- Total Runs: {pm["total_runs"]}
- Success Rate: {pm["success_rate"]:.1f}%
- Failed Runs: {pm["failed_runs"]}

### Quality Metrics
- Invariant Violation Rate: {pm["invariant_violation_rate"]:.2f}% (Target: 0%)
- Ambiguity Halt Rate: {pm["ambiguity_halt_rate"]:.1f}%

### Efficiency Metrics
- Mean Retries per Run: {pm["mean_retries_per_run"]:.2f}
- Mean Tokens per Run: {pm["mean_tokens_per_run"]:.0f}
- Memory Hit Rate: {pm["memory_hit_rate"]:.1f}%

## Research Quality

### Finding Distribution
- Total Findings: {rm["total_findings"]}
- Robust Findings: {rm["robust_findings"]} ({rm["robust_finding_rate"]:.1f}%)
- Spurious Findings: {rm["spurious_findings"]}
- Promising Findings: {rm["promising_findings"]}

### Validity Metrics
- Reproducibility Rate: {rm["reproducibility_rate"]:.1f}%
- False Discovery Rate: {rm["false_discovery_rate"]:.2f}% (Target: <5%)

### Resilience Metrics
- Failure Recovery Rate: {rm["failure_recovery_rate"]:.1f}%
- Total Failures: {rm["failures"]}
- Recovered: {rm["recovered_failures"]}

## Assessment

"""
        
        # Add automated assessment
        issues = []
        
        if pm["invariant_violation_rate"] > 0:
            issues.append(f"⚠️ CRITICAL: Invariant violations detected ({pm['invariant_violation_rate']:.2f}%)")
        
        if rm["false_discovery_rate"] > 5.0:
            issues.append(f"⚠️ WARNING: High false discovery rate ({rm['false_discovery_rate']:.1f}%)")
        
        if pm["memory_hit_rate"] < 20.0:
            issues.append(f"ℹ️ INFO: Low memory hit rate ({pm['memory_hit_rate']:.1f}%). Consider improving queries.")
        
        if issues:
            report += "### Issues Detected\n"
            for issue in issues:
                report += f"- {issue}\n"
        else:
            report += "✅ All metrics within acceptable ranges.\n"
        
        return report


# Global metrics collector instance
_global_collector = None


def get_metrics_collector() -> MetricsCollector:
    """Returns the global metrics collector instance."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector
