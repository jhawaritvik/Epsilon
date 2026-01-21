"""
Epsilon Metrics Analyzer
========================
Analyzes benchmark results and generates reports comparing Epsilon's
performance against established benchmarks.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


# Industry benchmark baselines for comparison
BENCHMARK_BASELINES = {
    "mlagentbench": {
        "sota_success_rate": 0.30,  # ~30% success rate with >10% improvement
        "description": "MLAgentBench (Stanford) - AI research agents on ML tasks",
        "reference": "https://arxiv.org/abs/2310.03302"
    },
    "scienceagentbench": {
        "sota_success_rate": 0.324,  # 32.4% best agent
        "description": "ScienceAgentBench - Scientific code generation tasks",
        "reference": "https://github.com/OSU-NLP-Group/ScienceAgentBench"
    },
    "mle_bench": {
        "sota_success_rate": 0.169,  # 16.9% bronze+ medals
        "description": "MLE-bench (OpenAI) - Kaggle competition performance",
        "reference": "https://openai.com/research/mle-bench"
    }
}


def load_results(results_dir: str) -> List[Dict[str, Any]]:
    """Load all benchmark results from directory."""
    results_path = Path(results_dir)
    results = []
    
    for file in results_path.glob("*.json"):
        with open(file, 'r') as f:
            data = json.load(f)
            data["filename"] = file.name
            results.append(data)
    
    return sorted(results, key=lambda x: x.get("timestamp", ""), reverse=True)


def generate_comparison_report(results: List[Dict[str, Any]]) -> str:
    """Generate a comparison report against industry benchmarks."""
    lines = [
        "=" * 70,
        "EPSILON BENCHMARK COMPARISON REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
    ]
    
    # Get latest result for each suite
    latest_by_suite = {}
    for r in results:
        suite = r.get("suite_name", "unknown")
        if suite not in latest_by_suite:
            latest_by_suite[suite] = r
    
    # Epsilon results summary
    lines.append("EPSILON RESULTS")
    lines.append("-" * 70)
    
    for suite, data in latest_by_suite.items():
        success_rate = data.get("success_rate", 0)
        total = data.get("total_tasks", 0)
        successful = data.get("successful_tasks", 0)
        
        lines.append(f"\n{suite.upper()}:")
        lines.append(f"  Success Rate: {success_rate * 100:.1f}%")
        lines.append(f"  Tasks: {successful}/{total}")
        lines.append(f"  Avg Time: {data.get('avg_execution_time', 0):.1f}s")
    
    # Comparison with industry benchmarks
    lines.append("\n")
    lines.append("=" * 70)
    lines.append("COMPARISON WITH INDUSTRY BENCHMARKS")
    lines.append("=" * 70)
    
    for bench_name, baseline in BENCHMARK_BASELINES.items():
        lines.append(f"\n{baseline['description']}")
        lines.append(f"  Industry SOTA: {baseline['sota_success_rate'] * 100:.1f}%")
        
        # Find comparable Epsilon result
        epsilon_suite = None
        if "mlagentbench" in bench_name:
            epsilon_suite = latest_by_suite.get("mlagentbench")
        elif "science" in bench_name:
            epsilon_suite = latest_by_suite.get("science")
        
        if epsilon_suite:
            epsilon_rate = epsilon_suite.get("success_rate", 0)
            diff = epsilon_rate - baseline["sota_success_rate"]
            
            if diff > 0:
                status = f"✅ ABOVE SOTA (+{diff * 100:.1f}%)"
            elif diff > -0.1:
                status = f"⚠️ NEAR SOTA ({diff * 100:.1f}%)"
            else:
                status = f"❌ BELOW SOTA ({diff * 100:.1f}%)"
            
            lines.append(f"  Epsilon: {epsilon_rate * 100:.1f}% {status}")
        else:
            lines.append("  Epsilon: No comparable results")
        
        lines.append(f"  Reference: {baseline['reference']}")
    
    # Recommendations
    lines.append("\n")
    lines.append("=" * 70)
    lines.append("RECOMMENDATIONS FOR PUBLICATION")
    lines.append("=" * 70)
    
    total_epsilon_rate = sum(
        r.get("success_rate", 0) for r in latest_by_suite.values()
    ) / len(latest_by_suite) if latest_by_suite else 0
    
    if total_epsilon_rate >= 0.35:
        lines.append("\n✅ READY FOR PUBLICATION")
        lines.append("   Epsilon demonstrates competitive performance with SOTA agents.")
    elif total_epsilon_rate >= 0.25:
        lines.append("\n⚠️ CLOSE TO PUBLICATION READY")
        lines.append("   Consider improving on specific task categories before submission.")
    else:
        lines.append("\n❌ NEEDS IMPROVEMENT")
        lines.append("   Focus on increasing success rate through:")
        lines.append("   - Better prompt engineering")
        lines.append("   - Enhanced error recovery")
        lines.append("   - More robust code generation")
    
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze Epsilon benchmark results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="benchmarks/results",
        help="Directory containing benchmark results"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for report (default: print to stdout)"
    )
    
    args = parser.parse_args()
    
    results = load_results(args.results_dir)
    
    if not results:
        print("No benchmark results found. Run benchmarks first:")
        print("  python benchmarks/run_eval.py --suite internal")
        return
    
    report = generate_comparison_report(results)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report saved to: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
