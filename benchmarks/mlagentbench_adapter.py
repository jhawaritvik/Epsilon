"""
Adapter to run Epsilon on MLAgentBench tasks.
Compares Epsilon vs MLAgentBench's GPT-4o agent.
"""
import sys
import json
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from controller import ResearchController
# MLAgentBench task mappings
TASKS = {
    "cifar10": {
        "goal": "Given a CNN training script for CIFAR-10, improve the model performance (accuracy) while keeping epochs <= 10. The baseline achieves ~54% test accuracy. Target: achieve >10% improvement.",
        "success_metric": "accuracy >= 59.4"  # 54 * 1.10
    },
    "vectorization": {
        "goal": "Given a Python script with nested loops for convolution, vectorize the forward function (lines 105-123) using numpy to improve execution speed. Baseline: ~2.2 seconds.",
        "success_metric": "speedup > 10x"
    }
}
def run_epsilon_on_task(task_name: str):
    task = TASKS.get(task_name)
    if not task:
        print(f"Unknown task: {task_name}")
        return
    
    print(f"\n{'='*60}")
    print(f"Running Epsilon on: {task_name}")
    print(f"Goal: {task['goal']}")
    print(f"{'='*60}\n")
    
    start = time.time()
    controller = ResearchController(
        user_id="00000000-0000-0000-0000-000000000099",
        max_iterations=3
    )
    controller.run(task["goal"])
    elapsed = time.time() - start
    
    print(f"\n{'='*60}")
    print(f"Epsilon completed in {elapsed:.1f}s")
    print(f"Check experiments/ for results")
    print(f"{'='*60}")
if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "cifar10"
    run_epsilon_on_task(task)
