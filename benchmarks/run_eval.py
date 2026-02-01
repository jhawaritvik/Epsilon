"""
Epsilon Benchmark Evaluation Suite
===================================
This module runs standardized benchmarks to evaluate Epsilon's performance
as an autonomous research engine.

Benchmark Suites:
- internal: Validates core functionality (epistemic loop, statistical tests)
- mlagentbench: MLAgentBench-inspired tasks (ML improvement tasks)
- science: Scientific discovery tasks (data analysis, visualization)

Usage:
    python benchmarks/run_eval.py --suite internal
    python benchmarks/run_eval.py --suite mlagentbench --tasks 5
    python benchmarks/run_eval.py --suite science --tasks 10
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from controller import ResearchController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BenchmarkRunner")


@dataclass
class BenchmarkResult:
    """Result of a single benchmark task."""
    task_id: str
    task_name: str
    suite: str
    success: bool
    execution_time_seconds: float
    improvement_percentage: Optional[float] = None
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


@dataclass
class SuiteResult:
    """Aggregate result of a benchmark suite."""
    suite_name: str
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    success_rate: float
    avg_execution_time: float
    total_time: float
    task_results: List[BenchmarkResult]


# =============================================================================
# BENCHMARK TASK DEFINITIONS
# =============================================================================

INTERNAL_TASKS = [
    {
        "id": "internal_001",
        "name": "Basic Hypothesis Testing",
        "goal": "Test if there is a statistically significant difference between two normally distributed samples with different means (mu1=10, mu2=12, n=100 each).",
        "expected_outcome": "p-value < 0.05, reject null hypothesis",
        "difficulty": "easy"
    },
    {
        "id": "internal_002", 
        "name": "No Effect Detection",
        "goal": "Test if there is a difference between two samples drawn from the same distribution (mu=50, sigma=10, n=50 each). The test should correctly fail to reject the null.",
        "expected_outcome": "p-value > 0.05, fail to reject null hypothesis",
        "difficulty": "easy"
    },
    {
        "id": "internal_003",
        "name": "Learning Rate Impact Analysis",
        "goal": "Investigate the impact of learning rate (0.001 vs 0.1) on training loss variance in a simple linear regression model over 100 epochs.",
        "expected_outcome": "Successfully identify learning rate effect on variance",
        "difficulty": "medium"
    },
    {
        "id": "internal_004",
        "name": "ANOVA Multi-Group Comparison",
        "goal": "Compare the performance of three different activation functions (ReLU, Tanh, Sigmoid) on a synthetic classification task. Use ANOVA to test for significant differences.",
        "expected_outcome": "Perform ANOVA test with valid F-statistic",
        "difficulty": "medium"
    },
    {
        "id": "internal_005",
        "name": "Correlation Discovery",
        "goal": "Investigate the correlation between input feature dimensionality (10, 50, 100, 500) and model training time for a simple neural network.",
        "expected_outcome": "Identify positive correlation with statistical significance",
        "difficulty": "medium"
    },
]

MLAGENTBENCH_TASKS = [
    {
        "id": "mlab_001",
        "name": "House Prices Prediction",
        "goal": "Improve a baseline linear regression model for predicting house prices. Target: achieve >10% reduction in RMSE on validation set.",
        "baseline_rmse": 45000,
        "target_improvement": 0.10,
        "difficulty": "easy",
        "gpu_required": False,
        # Target validation: check if RMSE reduction percentage meets threshold
        "target": {
            "metric": "mean_relative_improvement",
            "direction": "higher",
            "threshold": 0.10,  # 10% as decimal
            "key_path": "aggregate.mean_relative_improvement"
        }
    },
    {
        "id": "mlab_002",
        "name": "Tabular Classification",
        "goal": "Improve classification accuracy on a tabular dataset (Spaceship Titanic style). Baseline accuracy: 75%. Target: >78%.",
        "baseline_accuracy": 0.75,
        "target_improvement": 0.04,
        "difficulty": "easy",
        "gpu_required": False,
        "target": {
            "metric": "accuracy",
            "direction": "higher",
            "threshold": 0.78,
            "key_path": "final_accuracy"
        }
    },
    {
        "id": "mlab_003",
        "name": "Text Sentiment Analysis",
        "goal": "Improve sentiment classification on IMDB reviews using feature engineering. Baseline accuracy: 85%. Target: >87%.",
        "baseline_accuracy": 0.85,
        "target_improvement": 0.024,
        "difficulty": "medium",
        "gpu_required": False,
        "target": {
            "metric": "accuracy",
            "direction": "higher",
            "threshold": 0.87,
            "key_path": "final_accuracy"
        }
    },
    {
        "id": "mlab_004",
        "name": "Time Series Forecasting",
        "goal": "Improve time series prediction for synthetic stock data. Reduce MAE by at least 10% over naive baseline.",
        "baseline_mae": 5.2,
        "target_improvement": 0.10,
        "difficulty": "medium",
        "gpu_required": False,
        "target": {
            "metric": "percent_mae_reduction",
            "direction": "higher",
            "threshold": 10.0,
            "key_path": "mean_percent_mae_reduction"
        }
    },
    {
        "id": "mlab_005",
        "name": "Hyperparameter Optimization Study",
        "goal": "Systematically study the effect of hyperparameters (learning_rate, batch_size, hidden_units) on model performance. Report optimal configuration with statistical confidence.",
        "expected_outcome": "Identify optimal hyperparameters with confidence intervals",
        "difficulty": "hard",
        "gpu_required": False,
        # For hyperparameter study, success = having valid results with multiple configs
        "target": {
            "metric": "num_configurations_tested",
            "direction": "higher",
            "threshold": 9,  # At least 9 configs (3x3 grid)
            "key_path": None  # Use len(results) if results is a list
        }
    },
]

SCIENCE_TASKS = [
    {
        "id": "sci_001",
        "name": "Statistical Power Analysis",
        "goal": "Determine the minimum sample size required to detect a medium effect size (Cohen's d=0.5) with 80% power at alpha=0.05.",
        "expected_answer_range": (60, 70),
        "difficulty": "easy"
    },
    {
        "id": "sci_002",
        "name": "Distribution Fitting",
        "goal": "Given a synthetic dataset, determine whether it better fits a normal, exponential, or log-normal distribution using appropriate statistical tests.",
        "expected_outcome": "Correctly identify distribution type",
        "difficulty": "medium"
    },
    {
        "id": "sci_003",
        "name": "Regression Diagnostics",
        "goal": "Perform a complete regression analysis including residual diagnostics, multicollinearity check (VIF), and heteroscedasticity testing on a provided dataset.",
        "expected_outcome": "Complete diagnostic report with all tests",
        "difficulty": "medium"
    },
    {
        "id": "sci_004",
        "name": "Bootstrap Confidence Intervals",
        "goal": "Compute bootstrap confidence intervals for the median of a non-normal distribution and compare with parametric intervals.",
        "expected_outcome": "Valid bootstrap CI with comparison",
        "difficulty": "medium"
    },
    {
        "id": "sci_005",
        "name": "Experimental Design Validation",
        "goal": "Design and validate a factorial experiment (2x2) testing the effect of two factors on a response variable. Include power analysis and effect size estimation.",
        "expected_outcome": "Complete experimental design with analysis",
        "difficulty": "hard"
    },
]


def get_tasks_for_suite(suite: str) -> List[Dict[str, Any]]:
    """Get task definitions for a benchmark suite."""
    if suite == "internal":
        return INTERNAL_TASKS
    elif suite == "mlagentbench":
        return MLAGENTBENCH_TASKS
    elif suite == "science":
        return SCIENCE_TASKS
    else:
        raise ValueError(f"Unknown suite: {suite}")


def _extract_metric_value(results: Any, target: Dict[str, Any]) -> Optional[float]:
    """
    Extract the metric value from raw_results.json based on target definition.
    Handles nested paths (e.g., 'aggregate.mean_relative_improvement') and list averaging.
    """
    key_path = target.get("key_path")
    metric_name = target.get("metric")
    
    # If results is a list (like hyperparameter study), return length
    if key_path is None and isinstance(results, list):
        return float(len(results))
    
    # Handle dotted nested paths (e.g., "aggregate.mean_relative_improvement")
    if key_path and isinstance(results, dict) and "." in key_path:
        parts = key_path.split(".")
        current = results
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                current = None
                break
        if current is not None:
            if isinstance(current, (int, float)):
                return float(current)
            elif isinstance(current, list) and current:
                return float(sum(current) / len(current))
    
    # Try direct key access (non-nested)
    if key_path and isinstance(results, dict):
        if key_path in results:
            val = results[key_path]
            if isinstance(val, (int, float)):
                return float(val)
            elif isinstance(val, list) and val:
                # Average of list
                return float(sum(val) / len(val))
    
    # Try metric name as key
    if metric_name and isinstance(results, dict):
        if metric_name in results:
            val = results[metric_name]
            if isinstance(val, (int, float)):
                return float(val)
            elif isinstance(val, list) and val:
                return float(sum(val) / len(val))
    
    # Search recursively in dict for partial metric name match
    if isinstance(results, dict):
        for key, val in results.items():
            if metric_name and metric_name.lower() in key.lower():
                if isinstance(val, (int, float)):
                    return float(val)
                elif isinstance(val, list) and val:
                    return float(sum(val) / len(val))
            # Also search in nested dicts
            if isinstance(val, dict):
                for subkey, subval in val.items():
                    if metric_name and metric_name.lower() in subkey.lower():
                        if isinstance(subval, (int, float)):
                            return float(subval)
                        elif isinstance(subval, list) and subval:
                            return float(sum(subval) / len(subval))
    
    return None


def _check_target_achieved(results: Any, target: Dict[str, Any]) -> tuple:
    """
    Check if the target was achieved based on the results.
    Returns (success: bool, achieved_value: float or None, reason: str)
    """
    if not target:
        return (True, None, "No target specified")
    
    threshold = target.get("threshold")
    direction = target.get("direction", "higher")
    
    achieved_value = _extract_metric_value(results, target)
    
    if achieved_value is None:
        return (False, None, f"Could not extract metric '{target.get('metric')}' from results")
    
    if direction == "higher":
        success = achieved_value >= threshold
        reason = f"Achieved {achieved_value:.4f}, Target >= {threshold}"
    else:  # lower
        success = achieved_value <= threshold
        reason = f"Achieved {achieved_value:.4f}, Target <= {threshold}"
    
    return (success, achieved_value, reason)


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

class BenchmarkRunner:
    """Runs benchmark evaluations on Epsilon."""
    
    def __init__(self, results_dir: str = "benchmarks/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Use a fixed benchmark user ID
        self.user_id = "00000000-0000-0000-0000-000000000099"
    
    def run_single_task(self, task: Dict[str, Any], suite: str) -> BenchmarkResult:
        """Run a single benchmark task and return the result."""
        task_id = task["id"]
        task_name = task["name"]
        goal = task["goal"]
        
        logger.info(f"Running task: {task_name} ({task_id})")
        
        start_time = time.time()
        success = False
        error_message = None
        metrics = {}
        
        try:
            # Create a controller for this task
            controller = ResearchController(
                user_id=self.user_id,
                max_iterations=3  # Limit iterations for benchmarking
            )
            
            # Run the research
            controller.run(goal)
            
            # Check for success indicators
            experiment_dir = controller._get_experiment_dir()
            
            # Look for results file
            results_file = Path(experiment_dir) / "raw_results.json"
            report_file = Path(experiment_dir) / "FINAL_REPORT.md"
            
            if results_file.exists():
                with open(results_file, 'r') as f:
                    try:
                        results = json.load(f)
                        metrics["has_results"] = True
                        metrics["results_preview"] = str(results)[:500]
                        
                        # Target validation - THE CRITICAL FIX
                        target = task.get("target")
                        if target:
                            target_success, achieved_value, reason = _check_target_achieved(results, target)
                            metrics["target_achieved"] = target_success
                            metrics["achieved_value"] = achieved_value
                            metrics["target_reason"] = reason
                            
                            if target_success:
                                success = True
                                logger.info(f"✅ TARGET MET: {reason}")
                            else:
                                success = False
                                error_message = f"Target not met: {reason}"
                                logger.warning(f"❌ TARGET NOT MET: {reason}")
                        else:
                            # No target defined, having results is enough
                            success = True
                            metrics["target_achieved"] = None
                            
                    except json.JSONDecodeError as e:
                        metrics["has_results"] = False
                        metrics["parse_error"] = str(e)
                        success = False
                        error_message = f"Invalid JSON in raw_results.json: {e}"
            else:
                # NO RAW_RESULTS.JSON = AUTOMATIC FAILURE
                metrics["has_results"] = False
                success = False
                error_message = "raw_results.json not found - experiment did not produce results"
                logger.error(f"❌ FAILED: No raw_results.json in {experiment_dir}")
            
            # Report is optional for success, but track it
            if report_file.exists():
                metrics["has_report"] = True
            else:
                metrics["has_report"] = False
            
            # Additional success checks based on task type
            if "expected_outcome" in task:
                metrics["expected_outcome"] = task["expected_outcome"]
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Task {task_id} failed: {error_message}")
        
        execution_time = time.time() - start_time
        
        return BenchmarkResult(
            task_id=task_id,
            task_name=task_name,
            suite=suite,
            success=success,
            execution_time_seconds=execution_time,
            error_message=error_message,
            metrics=metrics
        )
    
    def run_suite(self, suite: str, max_tasks: Optional[int] = None, skip_tasks: int = 0) -> SuiteResult:
        """Run all tasks in a benchmark suite."""
        tasks = get_tasks_for_suite(suite)
        
        # Skip first N tasks
        if skip_tasks > 0:
            tasks = tasks[skip_tasks:]
            logger.info(f"Skipping first {skip_tasks} tasks")
        
        if max_tasks:
            tasks = tasks[:max_tasks]
        
        logger.info(f"Running {len(tasks)} tasks from suite: {suite}")
        
        results = []
        suite_start = time.time()
        
        for i, task in enumerate(tasks):
            logger.info(f"\n{'='*60}")
            logger.info(f"Task {i+1}/{len(tasks)}: {task['name']}")
            logger.info(f"{'='*60}")
            
            result = self.run_single_task(task, suite)
            results.append(result)
            
            status = "✅ PASS" if result.success else "❌ FAIL"
            logger.info(f"{status} - Completed in {result.execution_time_seconds:.1f}s")
        
        total_time = time.time() - suite_start
        successful = sum(1 for r in results if r.success)
        
        suite_result = SuiteResult(
            suite_name=suite,
            total_tasks=len(tasks),
            successful_tasks=successful,
            failed_tasks=len(tasks) - successful,
            success_rate=successful / len(tasks) if tasks else 0,
            avg_execution_time=sum(r.execution_time_seconds for r in results) / len(results) if results else 0,
            total_time=total_time,
            task_results=results
        )
        
        # Save results
        self._save_results(suite_result)
        
        return suite_result
    
    def _save_results(self, suite_result: SuiteResult):
        """Save benchmark results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{suite_result.suite_name}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Convert to dict for JSON serialization
        data = {
            "suite_name": suite_result.suite_name,
            "timestamp": timestamp,
            "total_tasks": suite_result.total_tasks,
            "successful_tasks": suite_result.successful_tasks,
            "failed_tasks": suite_result.failed_tasks,
            "success_rate": suite_result.success_rate,
            "avg_execution_time": suite_result.avg_execution_time,
            "total_time": suite_result.total_time,
            "task_results": [asdict(r) for r in suite_result.task_results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to: {filepath}")
        
        # Also save a summary
        summary_file = self.results_dir / f"{suite_result.suite_name}_latest.txt"
        with open(summary_file, 'w') as f:
            f.write(self._generate_summary(suite_result))
        
        logger.info(f"Summary saved to: {summary_file}")
    
    def _generate_summary(self, suite_result: SuiteResult) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 60,
            f"EPSILON BENCHMARK RESULTS: {suite_result.suite_name.upper()}",
            "=" * 60,
            "",
            f"Total Tasks:     {suite_result.total_tasks}",
            f"Successful:      {suite_result.successful_tasks}",
            f"Failed:          {suite_result.failed_tasks}",
            f"Success Rate:    {suite_result.success_rate * 100:.1f}%",
            f"Avg Time/Task:   {suite_result.avg_execution_time:.1f}s",
            f"Total Time:      {suite_result.total_time:.1f}s",
            "",
            "-" * 60,
            "TASK BREAKDOWN:",
            "-" * 60,
        ]
        
        for r in suite_result.task_results:
            status = "✅" if r.success else "❌"
            lines.append(f"{status} {r.task_name} ({r.execution_time_seconds:.1f}s)")
            if r.error_message:
                lines.append(f"   Error: {r.error_message[:100]}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)


def print_banner():
    """Print the benchmark runner banner."""
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║        EPSILON BENCHMARK EVALUATION SUITE                        ║
║        Autonomous Research Engine Validation                     ║
╚═══════════════════════════════════════════════════════════════════╝
    """)


def main():
    parser = argparse.ArgumentParser(
        description="Run Epsilon benchmark evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmarks/run_eval.py --suite internal
  python benchmarks/run_eval.py --suite mlagentbench --tasks 3
  python benchmarks/run_eval.py --suite science --tasks 5
        """
    )
    
    parser.add_argument(
        "--suite",
        type=str,
        choices=["internal", "mlagentbench", "science"],
        required=True,
        help="Benchmark suite to run"
    )
    
    parser.add_argument(
        "--tasks",
        type=int,
        default=None,
        help="Maximum number of tasks to run (default: all)"
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default="benchmarks/results",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Number of tasks to skip from the beginning (e.g., --skip 1 to skip task 1)"
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # Validate environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY not set. Please configure .env file.")
        sys.exit(1)
    
    # Run benchmarks
    runner = BenchmarkRunner(results_dir=args.results_dir)
    result = runner.run_suite(args.suite, max_tasks=args.tasks, skip_tasks=args.skip)
    
    # Print final summary
    print("\n" + runner._generate_summary(result))
    
    # Exit with appropriate code
    if result.success_rate >= 0.7:
        print("\n✅ BENCHMARK PASSED (>70% success rate)")
        sys.exit(0)
    else:
        print("\n⚠️ BENCHMARK NEEDS IMPROVEMENT (<70% success rate)")
        sys.exit(1)


if __name__ == "__main__":
    main()
