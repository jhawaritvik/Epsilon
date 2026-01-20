"""
Pipeline metrics collection and analysis (Refactored).

Uses a three-layer approach for paper-grade metrics:

1. RunMetrics (Per Run): iterations_to_success, valid_run, final_verdict, etc.
2. IterationMetrics (Per Iteration): failure_type, assumption_failures, etc.
3. AggregateMetrics (Across Runs): valid_run_rate, false_crystallization_rate, etc.

Uses EventBus for decoupled collection. Agents emit events; MetricsCollector subscribes.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime
from collections import defaultdict
from enum import Enum

# Import the event bus
try:
    from tests.metrics.event_bus import (
        get_event_bus, Event,
        ITERATION_STARTED, ITERATION_COMPLETED, ITERATION_FAILED,
        ASSUMPTION_FAILED, ASSUMPTION_PASSED, CRYSTALLIZATION_ATTEMPTED,
        MEMORY_QUERIED, MEMORY_HIT, EVIDENCE_ADDED, KNOWLEDGE_REUSED, FAILURE_RECALLED,
        AUTHORITY_VIOLATION, MODALITY_VIOLATION, TOOL_MISUSE, AMBIGUITY_HALT,
        TOKENS_USED, TIME_RECORDED,
        RUN_STARTED, RUN_COMPLETED, RUN_FAILED
    )
except ImportError:
    # Fallback for standalone testing - should not happen in production
    from event_bus import (
        get_event_bus, Event,
        ITERATION_STARTED, ITERATION_COMPLETED, ITERATION_FAILED,
        ASSUMPTION_FAILED, ASSUMPTION_PASSED, CRYSTALLIZATION_ATTEMPTED,
        MEMORY_QUERIED, MEMORY_HIT, EVIDENCE_ADDED, KNOWLEDGE_REUSED, FAILURE_RECALLED,
        AUTHORITY_VIOLATION, MODALITY_VIOLATION, TOOL_MISUSE, AMBIGUITY_HALT,
        TOKENS_USED, TIME_RECORDED,
        RUN_STARTED, RUN_COMPLETED, RUN_FAILED
    )


class FailureType(Enum):
    """Types of failures in the pipeline."""
    DESIGN = "DESIGN"
    DATA = "DATA"
    EXECUTION = "EXECUTION"
    STATISTICAL = "STATISTICAL"
    CONTRACT = "CONTRACT"


# =============================================================================
# LAYER 1: RunMetrics (Per Run)
# =============================================================================

@dataclass
class RunMetrics:
    """Metrics for a single pipeline run."""
    
    run_id: str
    iterations_to_success: int = 0
    valid_run: bool = False
    final_verdict: str = ""
    total_tokens: int = 0
    total_time_seconds: float = 0.0
    
    # Breakdown by agent
    tokens_by_agent: Dict[str, int] = field(default_factory=dict)
    time_by_agent: Dict[str, float] = field(default_factory=dict)
    
    # Contract integrity
    authority_violations: int = 0
    modality_violations: int = 0
    tool_misuse_attempts: int = 0
    
    # Scientific validity
    assumption_failures: int = 0
    crystallized: bool = False
    
    halted_on_ambiguity: bool = False
    failure_type: Optional[FailureType] = None
    
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        d = asdict(self)
        d["started_at"] = self.started_at.isoformat()
        d["completed_at"] = self.completed_at.isoformat() if self.completed_at else None
        d["failure_type"] = self.failure_type.value if self.failure_type else None
        return d


# =============================================================================
# LAYER 2: IterationMetrics (Per Iteration)
# =============================================================================

@dataclass
class IterationMetrics:
    """Metrics for a single iteration within a run."""
    
    run_id: str
    iteration_number: int
    
    failure_type: Optional[FailureType] = None
    assumption_failures: int = 0
    
    tokens_used: int = 0
    time_seconds: float = 0.0
    
    # Agent activity
    agent_tokens: Dict[str, int] = field(default_factory=dict)
    
    # Contract integrity
    authority_violations: int = 0
    
    # Memory interactions
    memory_queries: int = 0
    memory_hits: int = 0
    
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    success: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        d = asdict(self)
        d["started_at"] = self.started_at.isoformat()
        d["completed_at"] = self.completed_at.isoformat() if self.completed_at else None
        d["failure_type"] = self.failure_type.value if self.failure_type else None
        return d


# =============================================================================
# LAYER 3: AggregateMetrics (Across Runs)
# =============================================================================

@dataclass
class AggregateMetrics:
    """Aggregated metrics across all runs. Maps directly to paper figures."""
    
    # Scientific Validity
    total_runs: int = 0
    valid_runs: int = 0
    false_crystallizations: int = 0
    total_crystallizations: int = 0
    assumption_failure_count: int = 0
    
    # Convergence & Efficiency
    total_iterations: int = 0
    successful_iterations: int = 0
    iterations_to_success_list: List[int] = field(default_factory=list)
    failure_type_distribution: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    repeated_failures: int = 0  # Same failure type twice without spec change
    
    # Memory Utilization
    total_memory_queries: int = 0
    memory_hits: int = 0
    evidence_added: int = 0
    duplicate_evidence: int = 0
    knowledge_reuse_count: int = 0
    failure_recall_count: int = 0
    
    # Contract Integrity
    total_authority_violations: int = 0
    total_modality_violations: int = 0
    total_tool_misuse: int = 0
    ambiguity_halts: int = 0
    
    # Cost-Efficiency
    total_tokens: int = 0
    tokens_for_valid_experiments: int = 0
    tokens_for_crystallized_results: int = 0
    tokens_for_discarded_failures: int = 0
    total_time_seconds: float = 0.0
    
    # Breakdown by agent
    tokens_by_agent: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # =========================================================================
    # Computed Properties (for paper tables)
    # =========================================================================
    
    @property
    def valid_run_rate(self) -> float:
        """% of runs passing all checks."""
        return (self.valid_runs / self.total_runs * 100) if self.total_runs > 0 else 0.0
    
    @property
    def false_crystallization_rate(self) -> float:
        """% of crystallizations that were invalid (MUST be 0)."""
        return (self.false_crystallizations / self.total_crystallizations * 100) if self.total_crystallizations > 0 else 0.0
    
    @property
    def assumption_failure_rate(self) -> float:
        """% of runs with assumption failures."""
        return (self.assumption_failure_count / self.total_runs * 100) if self.total_runs > 0 else 0.0
    
    @property
    def mean_iterations_to_success(self) -> float:
        """Mean iterations needed for success."""
        return sum(self.iterations_to_success_list) / len(self.iterations_to_success_list) if self.iterations_to_success_list else 0.0
    
    @property
    def median_iterations_to_success(self) -> float:
        """Median iterations needed for success."""
        if not self.iterations_to_success_list:
            return 0.0
        sorted_list = sorted(self.iterations_to_success_list)
        mid = len(sorted_list) // 2
        return sorted_list[mid] if len(sorted_list) % 2 else (sorted_list[mid - 1] + sorted_list[mid]) / 2
    
    @property
    def repeat_failure_rate(self) -> float:
        """% of runs with repeated failures (bad)."""
        return (self.repeated_failures / self.total_runs * 100) if self.total_runs > 0 else 0.0
    
    @property
    def evidence_reuse_rate(self) -> float:
        """% of memory queries that found useful info."""
        return (self.memory_hits / self.total_memory_queries * 100) if self.total_memory_queries > 0 else 0.0
    
    @property
    def duplicate_evidence_rate(self) -> float:
        """% of evidence inserts that were duplicates (bad)."""
        return (self.duplicate_evidence / self.evidence_added * 100) if self.evidence_added > 0 else 0.0
    
    @property
    def knowledge_reuse_rate(self) -> float:
        """How often crystallized knowledge is consulted."""
        return (self.knowledge_reuse_count / self.total_runs * 100) if self.total_runs > 0 else 0.0
    
    @property
    def failure_recall_rate(self) -> float:
        """Did Design Agent query past failures?"""
        return (self.failure_recall_count / self.total_runs * 100) if self.total_runs > 0 else 0.0
    
    @property
    def authority_violation_rate(self) -> float:
        """Authority violation rate (MUST be 0)."""
        return (self.total_authority_violations / self.total_runs * 100) if self.total_runs > 0 else 0.0
    
    @property
    def ambiguity_halt_rate(self) -> float:
        """% of runs that correctly halted on ambiguity."""
        return (self.ambiguity_halts / self.total_runs * 100) if self.total_runs > 0 else 0.0
    
    @property
    def tokens_per_valid_experiment(self) -> float:
        """Tokens per valid experiment."""
        return self.tokens_for_valid_experiments / self.valid_runs if self.valid_runs > 0 else 0.0
    
    @property
    def tokens_per_crystallized_result(self) -> float:
        """Tokens per crystallized result."""
        return self.tokens_for_crystallized_results / self.total_crystallizations if self.total_crystallizations > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Full serialization with computed properties."""
        d = asdict(self)
        d["failure_type_distribution"] = dict(self.failure_type_distribution)
        d["tokens_by_agent"] = dict(self.tokens_by_agent)
        d.update({
            "valid_run_rate": self.valid_run_rate,
            "false_crystallization_rate": self.false_crystallization_rate,
            "assumption_failure_rate": self.assumption_failure_rate,
            "mean_iterations_to_success": self.mean_iterations_to_success,
            "median_iterations_to_success": self.median_iterations_to_success,
            "repeat_failure_rate": self.repeat_failure_rate,
            "evidence_reuse_rate": self.evidence_reuse_rate,
            "duplicate_evidence_rate": self.duplicate_evidence_rate,
            "knowledge_reuse_rate": self.knowledge_reuse_rate,
            "failure_recall_rate": self.failure_recall_rate,
            "authority_violation_rate": self.authority_violation_rate,
            "ambiguity_halt_rate": self.ambiguity_halt_rate,
            "tokens_per_valid_experiment": self.tokens_per_valid_experiment,
            "tokens_per_crystallized_result": self.tokens_per_crystallized_result,
        })
        return d


# =============================================================================
# MetricsCollector (Event-Driven)
# =============================================================================

class MetricsCollector:
    """
    Collects metrics by subscribing to events emitted by agents.
    
    Agents remain pure - they just emit events. This collector subscribes
    and aggregates data.
    """
    
    def __init__(self, storage_path: Path = None):
        self.storage_path = storage_path or Path("metrics.json")
        self.aggregate = AggregateMetrics()
        self.run_history: List[RunMetrics] = []
        self.iteration_history: List[IterationMetrics] = []
        
        # Current run/iteration tracking
        self._current_run: Optional[RunMetrics] = None
        self._current_iteration: Optional[IterationMetrics] = None
        self._last_failure_type: Optional[FailureType] = None
        
        # Subscribe to events
        self._subscribe_to_events()
    
    def _subscribe_to_events(self):
        """Subscribe to all relevant events."""
        bus = get_event_bus()
        
        bus.subscribe(RUN_STARTED, self._on_run_started)
        bus.subscribe(RUN_COMPLETED, self._on_run_completed)
        bus.subscribe(RUN_FAILED, self._on_run_failed)
        
        bus.subscribe(ITERATION_STARTED, self._on_iteration_started)
        bus.subscribe(ITERATION_COMPLETED, self._on_iteration_completed)
        bus.subscribe(ITERATION_FAILED, self._on_iteration_failed)
        
        bus.subscribe(ASSUMPTION_FAILED, self._on_assumption_failed)
        bus.subscribe(CRYSTALLIZATION_ATTEMPTED, self._on_crystallization)
        
        bus.subscribe(MEMORY_QUERIED, self._on_memory_queried)
        bus.subscribe(MEMORY_HIT, self._on_memory_hit)
        bus.subscribe(EVIDENCE_ADDED, self._on_evidence_added)
        bus.subscribe(KNOWLEDGE_REUSED, self._on_knowledge_reused)
        bus.subscribe(FAILURE_RECALLED, self._on_failure_recalled)
        
        bus.subscribe(AUTHORITY_VIOLATION, self._on_authority_violation)
        bus.subscribe(MODALITY_VIOLATION, self._on_modality_violation)
        bus.subscribe(TOOL_MISUSE, self._on_tool_misuse)
        bus.subscribe(AMBIGUITY_HALT, self._on_ambiguity_halt)
        
        bus.subscribe(TOKENS_USED, self._on_tokens_used)
        bus.subscribe(TIME_RECORDED, self._on_time_recorded)
    
    # =========================================================================
    # Event Handlers
    # =========================================================================
    
    def _on_run_started(self, event: Event):
        run_id = event.data.get("run_id", f"run-{datetime.now().isoformat()}")
        self._current_run = RunMetrics(run_id=run_id)
        self.aggregate.total_runs += 1
    
    def _on_run_completed(self, event: Event):
        if self._current_run:
            self._current_run.completed_at = datetime.now()
            self._current_run.valid_run = event.data.get("valid", True)
            self._current_run.final_verdict = event.data.get("verdict", "")
            
            if self._current_run.valid_run:
                self.aggregate.valid_runs += 1
                self.aggregate.tokens_for_valid_experiments += self._current_run.total_tokens
                self.aggregate.iterations_to_success_list.append(self._current_run.iterations_to_success)
            else:
                self.aggregate.tokens_for_discarded_failures += self._current_run.total_tokens
            
            self.run_history.append(self._current_run)
            self._last_failure_type = None
            self._current_run = None
    
    def _on_run_failed(self, event: Event):
        if self._current_run:
            self._current_run.completed_at = datetime.now()
            self._current_run.valid_run = False
            failure_type_str = event.data.get("failure_type", "EXECUTION")
            self._current_run.failure_type = FailureType[failure_type_str]
            
            self.aggregate.failure_type_distribution[failure_type_str] += 1
            self.aggregate.tokens_for_discarded_failures += self._current_run.total_tokens
            
            self.run_history.append(self._current_run)
            self._current_run = None
    
    def _on_iteration_started(self, event: Event):
        if self._current_run:
            iteration_num = event.data.get("iteration", self._current_run.iterations_to_success + 1)
            self._current_iteration = IterationMetrics(
                run_id=self._current_run.run_id,
                iteration_number=iteration_num
            )
            self._current_run.iterations_to_success = iteration_num
            self.aggregate.total_iterations += 1
    
    def _on_iteration_completed(self, event: Event):
        if self._current_iteration:
            self._current_iteration.completed_at = datetime.now()
            self._current_iteration.success = True
            self.aggregate.successful_iterations += 1
            self.iteration_history.append(self._current_iteration)
            self._last_failure_type = None
            self._current_iteration = None
    
    def _on_iteration_failed(self, event: Event):
        if self._current_iteration:
            self._current_iteration.completed_at = datetime.now()
            self._current_iteration.success = False
            failure_type_str = event.data.get("failure_type", "EXECUTION")
            self._current_iteration.failure_type = FailureType[failure_type_str]
            
            # Check for repeated failure
            if self._last_failure_type == FailureType[failure_type_str]:
                self.aggregate.repeated_failures += 1
            self._last_failure_type = FailureType[failure_type_str]
            
            self.iteration_history.append(self._current_iteration)
            self._current_iteration = None
    
    def _on_assumption_failed(self, event: Event):
        self.aggregate.assumption_failure_count += 1
        if self._current_run:
            self._current_run.assumption_failures += 1
        if self._current_iteration:
            self._current_iteration.assumption_failures += 1
    
    def _on_crystallization(self, event: Event):
        self.aggregate.total_crystallizations += 1
        valid = event.data.get("valid", True)
        if not valid:
            self.aggregate.false_crystallizations += 1
        if self._current_run:
            self._current_run.crystallized = True
            if self._current_run.valid_run:
                self.aggregate.tokens_for_crystallized_results += self._current_run.total_tokens
    
    def _on_memory_queried(self, event: Event):
        self.aggregate.total_memory_queries += 1
        if self._current_iteration:
            self._current_iteration.memory_queries += 1
    
    def _on_memory_hit(self, event: Event):
        self.aggregate.memory_hits += 1
        if self._current_iteration:
            self._current_iteration.memory_hits += 1
    
    def _on_evidence_added(self, event: Event):
        self.aggregate.evidence_added += 1
        if event.data.get("duplicate", False):
            self.aggregate.duplicate_evidence += 1
    
    def _on_knowledge_reused(self, event: Event):
        self.aggregate.knowledge_reuse_count += 1
    
    def _on_failure_recalled(self, event: Event):
        self.aggregate.failure_recall_count += 1
    
    def _on_authority_violation(self, event: Event):
        self.aggregate.total_authority_violations += 1
        if self._current_run:
            self._current_run.authority_violations += 1
        if self._current_iteration:
            self._current_iteration.authority_violations += 1
    
    def _on_modality_violation(self, event: Event):
        self.aggregate.total_modality_violations += 1
        if self._current_run:
            self._current_run.modality_violations += 1
    
    def _on_tool_misuse(self, event: Event):
        self.aggregate.total_tool_misuse += 1
        if self._current_run:
            self._current_run.tool_misuse_attempts += 1
    
    def _on_ambiguity_halt(self, event: Event):
        self.aggregate.ambiguity_halts += 1
        if self._current_run:
            self._current_run.halted_on_ambiguity = True
    
    def _on_tokens_used(self, event: Event):
        tokens = event.data.get("count", 0)
        agent = event.data.get("agent", "unknown")
        
        self.aggregate.total_tokens += tokens
        self.aggregate.tokens_by_agent[agent] += tokens
        
        if self._current_run:
            self._current_run.total_tokens += tokens
            self._current_run.tokens_by_agent[agent] = self._current_run.tokens_by_agent.get(agent, 0) + tokens
        
        if self._current_iteration:
            self._current_iteration.tokens_used += tokens
            self._current_iteration.agent_tokens[agent] = self._current_iteration.agent_tokens.get(agent, 0) + tokens
    
    def _on_time_recorded(self, event: Event):
        seconds = event.data.get("seconds", 0.0)
        agent = event.data.get("agent", "unknown")
        
        self.aggregate.total_time_seconds += seconds
        
        if self._current_run:
            self._current_run.total_time_seconds += seconds
            self._current_run.time_by_agent[agent] = self._current_run.time_by_agent.get(agent, 0.0) + seconds
        
        if self._current_iteration:
            self._current_iteration.time_seconds += seconds
    
    # =========================================================================
    # Persistence
    # =========================================================================
    
    def save(self):
        """Save metrics to JSON file."""
        data = {
            "aggregate": self.aggregate.to_dict(),
            "run_history": [r.to_dict() for r in self.run_history],
            "iteration_history": [i.to_dict() for i in self.iteration_history],
            "collected_at": datetime.now().isoformat()
        }
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def reset(self):
        """Reset all metrics."""
        self.aggregate = AggregateMetrics()
        self.run_history.clear()
        self.iteration_history.clear()
        self._current_run = None
        self._current_iteration = None
        self._last_failure_type = None
    
    def generate_report(self) -> str:
        """Generate a human-readable metrics report."""
        agg = self.aggregate
        
        return f"""
# Epsilon Pipeline Metrics Report
Generated: {datetime.now().isoformat()}

## Scientific Validity

| Metric | Value | Target |
|--------|-------|--------|
| Valid Run Rate | {agg.valid_run_rate:.1f}% | >70% |
| False Crystallization Rate | {agg.false_crystallization_rate:.2f}% | **0%** |
| Assumption Failure Rate | {agg.assumption_failure_rate:.1f}% | <10% |

## Convergence & Efficiency

| Metric | Value |
|--------|-------|
| Mean Iterations to Success | {agg.mean_iterations_to_success:.2f} |
| Median Iterations to Success | {agg.median_iterations_to_success:.1f} |
| Repeat Failure Rate | {agg.repeat_failure_rate:.1f}% |

### Failure Type Distribution
{json.dumps(dict(agg.failure_type_distribution), indent=2)}

## Memory Utilization

| Metric | Value | Target |
|--------|-------|--------|
| Evidence Reuse Rate | {agg.evidence_reuse_rate:.1f}% | >20% |
| Duplicate Evidence Rate | {agg.duplicate_evidence_rate:.1f}% | 0% |
| Knowledge Reuse Rate | {agg.knowledge_reuse_rate:.1f}% | - |
| Failure Recall Rate | {agg.failure_recall_rate:.1f}% | >50% |

## Contract Integrity

| Metric | Value | Target |
|--------|-------|--------|
| Authority Violation Rate | {agg.authority_violation_rate:.2f}% | **0%** |
| Ambiguity Halt Rate | {agg.ambiguity_halt_rate:.1f}% | >90% when triggered |

## Cost-Efficiency

| Metric | Value |
|--------|-------|
| Total Tokens | {agg.total_tokens:,} |
| Tokens per Valid Experiment | {agg.tokens_per_valid_experiment:,.0f} |
| Tokens per Crystallized Result | {agg.tokens_per_crystallized_result:,.0f} |

### Tokens by Agent
{json.dumps(dict(agg.tokens_by_agent), indent=2)}

## Assessment

{"✅ All critical invariants (false_crystallization_rate=0, authority_violations=0) are satisfied." if agg.false_crystallization_rate == 0 and agg.total_authority_violations == 0 else "⚠️ CRITICAL: Invariant violations detected!"}
"""


# =============================================================================
# Metric Invariant Assertions (for tests)
# =============================================================================

def assert_metric_invariants(metrics: AggregateMetrics):
    """
    Assert that critical metric invariants hold.
    
    Use this in tests to ensure the pipeline is paper-grade.
    """
    # CRITICAL: These must be zero
    assert metrics.false_crystallization_rate == 0, \
        f"False crystallization rate must be 0, got {metrics.false_crystallization_rate:.2f}%"
    
    assert metrics.total_authority_violations == 0, \
        f"Authority violations must be 0, got {metrics.total_authority_violations}"
    
    # WARNING: These should be low
    assert metrics.repeat_failure_rate < 20, \
        f"Repeat failure rate too high: {metrics.repeat_failure_rate:.1f}%"
    
    assert metrics.duplicate_evidence_rate < 5, \
        f"Duplicate evidence rate too high: {metrics.duplicate_evidence_rate:.1f}%"


# =============================================================================
# Global Collector Instance
# =============================================================================

_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Returns the global MetricsCollector singleton."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


# =============================================================================
# BACKWARD COMPATIBILITY LAYER
# These classes maintain API compatibility with legacy tests.
# =============================================================================

@dataclass
class PipelineMetrics:
    """
    Legacy PipelineMetrics class for backward compatibility.
    
    New code should use the event-driven MetricsCollector instead.
    """
    
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
    """
    Legacy ResearchMetrics class for backward compatibility.
    
    New code should use the event-driven MetricsCollector instead.
    """
    
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


class LegacyMetricsCollector(MetricsCollector):
    """
    Legacy MetricsCollector with old API for backward compatibility.
    
    Provides record_run, record_finding, etc. methods that wrap the new
    event-driven system.
    """
    
    def __init__(self, storage_path: Path = None):
        super().__init__(storage_path)
        self.pipeline_metrics = PipelineMetrics()
        self.research_metrics = ResearchMetrics()
        self._run_history: List[Dict[str, Any]] = []
    
    def record_run(self, 
                   run_id: str,
                   status: str,
                   retries: int = 0,
                   tokens_used: int = 0,
                   invariant_violations: int = 0,
                   halted_on_ambiguity: bool = False):
        """
        Records a single run (legacy API).
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
        self._run_history.append({
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "retries": retries,
            "tokens_used": tokens_used,
            "invariant_violations": invariant_violations,
            "halted_on_ambiguity": halted_on_ambiguity,
        })
    
    def record_memory_query(self, found_useful_info: bool):
        """Records a memory query (legacy API)."""
        self.pipeline_metrics.memory_queries += 1
        if found_useful_info:
            self.pipeline_metrics.memory_hits += 1
    
    def record_finding(self, verdict: str):
        """Records a research finding (legacy API)."""
        self.research_metrics.total_findings += 1
        
        if verdict == "robust" or verdict == "very_robust":
            self.research_metrics.robust_findings += 1
        elif verdict == "spurious":
            self.research_metrics.spurious_findings += 1
        elif verdict == "promising":
            self.research_metrics.promising_findings += 1
    
    def record_reproduction_attempt(self, successful: bool):
        """Records a reproduction attempt (legacy API)."""
        self.research_metrics.reproduction_attempts += 1
        if successful:
            self.research_metrics.reproduced_findings += 1
    
    def record_contradiction(self):
        """Records a finding that was later contradicted (legacy API)."""
        self.research_metrics.contradicted_findings += 1
    
    def record_failure_recovery(self, recovered: bool):
        """Records a failure and whether it was recovered (legacy API)."""
        self.research_metrics.failures += 1
        if recovered:
            self.research_metrics.recovered_failures += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Returns a summary of all metrics (legacy API)."""
        return {
            "pipeline_metrics": self.pipeline_metrics.to_dict(),
            "research_metrics": self.research_metrics.to_dict(),
            "collected_at": datetime.now().isoformat(),
        }
    
    def save(self):
        """Saves metrics to JSON file (legacy API)."""
        with open(self.storage_path, 'w') as f:
            json.dump({
                "summary": self.get_summary(),
                "run_history": self._run_history,
            }, f, indent=2)
    
    def load(self):
        """Loads metrics from JSON file (legacy API)."""
        if not self.storage_path.exists():
            return
        
        with open(self.storage_path, 'r') as f:
            data = json.load(f)
            self._run_history = data.get("run_history", [])
            
            # Reconstruct metrics from history
            for run in self._run_history:
                self.record_run(
                    run_id=run["run_id"],
                    status=run["status"],
                    retries=run.get("retries", 0),
                    tokens_used=run.get("tokens_used", 0),
                    invariant_violations=run.get("invariant_violations", 0),
                    halted_on_ambiguity=run.get("halted_on_ambiguity", False)
                )
    
    def generate_report(self) -> str:
        """Generates a human-readable metrics report (legacy API)."""
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


# Alias for backward compatibility - tests using MetricsCollector(path) will get legacy version
# Override: When `MetricsCollector` is instantiated with legacy API usage, use LegacyMetricsCollector
def MetricsCollector(storage_path: Path = None) -> LegacyMetricsCollector:  # type: ignore
    """Factory function that returns LegacyMetricsCollector for backward compatibility."""
    return LegacyMetricsCollector(storage_path)

