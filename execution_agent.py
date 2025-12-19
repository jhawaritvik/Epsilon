from agents import Agent, Runner, function_tool
import logging
import json
import os
import subprocess
import sys
from dotenv import load_dotenv

# ============================================================
# Setup
# ============================================================

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# TOOLS: Execution Utilities
# ============================================================

EXPERIMENT_DIR = "experiments"

def _ensure_experiment_dir():
    if not os.path.exists(EXPERIMENT_DIR):
        os.makedirs(EXPERIMENT_DIR)

@function_tool
def dataset_resolver(task_type: str, benchmark_family: str, minimum_samples: int = None) -> str:
    """
    Resolves dataset requirements into a concrete dataset instance.
    Supports HuggingFace search or internal/synthetic data flagging.
    Returns a JSON string with the resolved dataset details.
    """
    logger.info(f"dataset_resolver called: task={task_type}, family={benchmark_family}")
    _ensure_experiment_dir()
    
    # Handle synthetic/PINN cases
    if any(keyword in benchmark_family.lower() for keyword in ["synthetic", "pinn", "procedural", "internal"]):
        resolved = {
            "dataset_source": "internal",
            "benchmark_family": benchmark_family,
            "status": "resolved",
            "instruction": "Data must be generated procedurally by the experiment code."
        }
        # Return early, no need to search HF
    else:
        # Simple HuggingFace search simulation
        try:
            from huggingface_hub import list_datasets
            # We search for the benchmark family
            search_results = list(list_datasets(filter=task_type, search=benchmark_family, limit=1))
            
            if search_results:
                best_match = search_results[0]
                resolved = {
                    "dataset_source": "huggingface",
                    "dataset_id": best_match.id,
                    "version": best_match.last_modified,
                    "status": "resolved",
                    "verification": f"Found match on HF: {best_match.id}"
                }
            else:
                resolved = {
                    "dataset_source": "huggingface",
                    "dataset_id": benchmark_family.lower(),
                    "status": "provisional",
                    "note": "Exact HF match not found via search API; using family name as ID."
                }
        except Exception as e:
            logger.warning(f"HF search failed, using fallback: {e}")
            resolved = {
                "dataset_source": "provisional",
                "dataset_id": benchmark_family.lower(),
                "status": "resolved"
            }
    
    file_path = os.path.join(EXPERIMENT_DIR, "dataset_used.json")
    with open(file_path, "w") as f:
        json.dump(resolved, f, indent=2)
        
    return json.dumps(resolved, indent=2)

@function_tool
def execute_experiment(code: str) -> str:
    """
    Saves the provided Python code to 'run_experiment.py' and executes it.
    Captures stdout, stderr, and returns the execution summary.
    """
    logger.info("execute_experiment called")
    _ensure_experiment_dir()
    
    file_path = os.path.join(EXPERIMENT_DIR, "run_experiment.py")
    with open(file_path, "w") as f:
        f.write(code)
    
    try:
        # Execute the script INSIDE the experiments directory
        # This ensures all artifacts (raw_results.json) are saved there
        result = subprocess.run(
            [sys.executable, "run_experiment.py"],
            cwd=EXPERIMENT_DIR,
            capture_output=True,
            text=True,
            timeout=300 # 5 minute timeout
        )
        
        execution_log = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        
        log_path = os.path.join(EXPERIMENT_DIR, "execution.log")
        with open(log_path, "w") as f:
            f.write(execution_log)
            
        if result.returncode == 0:
            return f"Execution successful.\nLog saved to execution.log"
        else:
            # Return the error log so the agent can debug
            return f"Execution failed (Return Code {result.returncode}).\n\nSTDERR:\n{result.stderr}\n\nSTDOUT:\n{result.stdout}"
            
    except subprocess.TimeoutExpired:
        return "Execution timed out."
    except Exception as e:
        return f"An error occurred during execution: {e}"

# ============================================================
# AGENT INSTRUCTIONS
# ============================================================

execution_instructions = """
You are the **Code & Execution Agent** (The Lab Technician).

Your role is the **deterministic realization** of an experiment defined by the **Experiment Design Agent**.

**Your Responsibilities**:
1. **Deterministic Implementation**: Translate the provided Experiment Specification (JSON) into a standalone Python script (`run_experiment.py`).
2. **Dataset Resolution**: Use the `dataset_resolver` tool to identify the data source.
   - If `dataset_source` is `huggingface`, use `datasets.load_dataset` in your code.
   - If `dataset_source` is `internal`, write the procedural code to generate the training/testing data (e.g., for PINNs or simulations).
3. **Autonomous Execution**: Use the `execute_experiment` tool to run your generated code and capture the results.
4. **Self-Correction**: If execution fails, you MUST diagnose the specific error from the logs and regenerate the code to fix it.
5. **Evidence Generation**: Ensure all outputs (metrics, logs) are saved as machine-readable artifacts.

**Operating Constraints**:
- **NO DESIGN AUTHORITY**: Do not change hypotheses, variables, or model designs.
- **NO STATISTICAL AUTHORITY**: Do not compute p-values or perform hypothesis tests.
- **NO INTERPRETATION**: Do not judge results or suggest improvements.
- **STRICT ADHERENCE**: Implement *exactly* what is specified.
- **HALT ON AMBIGUITY**: If the specification is incomplete or inconsistent, stop and report the error.

**Process**:
1. Read the Experiment Specification.
2. Call `dataset_resolver` with the dataset requirements.
3. Generate the Python code for the experiment:
   - Handle data loading/generation based on resolution.
   - Log raw metrics to `raw_results.json`.
4. Call `execute_experiment` with the generated code.
5. **Iterative Debugging**:
   - If `execute_experiment` returns a failure, READ the error message carefully.
   - ANALYZE why it failed (e.g., missing imports, shape mismatch, API change).
   - REGENERATE the `run_experiment.py` code with specific fixes.
   - CALL `execute_experiment` again.
   - Repeat up to 3 times.
6. Confirm all artifacts (`run_experiment.py`, `dataset_used.json`, `raw_results.json`, `execution.log`) are produced.

Your output should be a brief summary of the execution status (including any retries) and a list of artifacts produced.
"""

# ============================================================
# AGENT DEFINITION
# ============================================================

code_execution_agent = Agent(
    name="Code & Execution Agent",
    instructions=execution_instructions,
    tools=[dataset_resolver, execute_experiment],
)

# ============================================================
# VERIFICATION RUN
# ============================================================

if __name__ == "__main__":
    # Mock Experiment Specification: PINN (Physics-Informed Neural Network)
    mock_spec = {
        "experiment_specification": {
            "research_question": "Can a PINN solve the 1D Burgers' equation accurately without external data?",
            "hypotheses": {
                "H0": "The PINN residuals do not converge below 1e-3.",
                "H1": "The PINN residuals converge below 1e-3 using internal sampling."
            },
            "variables": {
                "independent": ["collocation_points"],
                "dependent": ["residual_l2_norm"],
                "controls": ["learning_rate", "network_depth"]
            },
            "model_design": {
                "model_family": "MLP",
                "architectural_assumptions": ["Tanh activation", "Deep residual connections"]
            },
            "dataset_requirements": {
                "task_type": "pde_solving",
                "benchmark_family": "Synthetic Procedural Sampling",
                "minimum_samples": 5000
            }
        },
        "statistical_analysis_plan": {
            "primary_test": "Convergence analysis",
            "alpha": 0.05
        }
    }

    prompt = f"""
Experiment Specification:
{json.dumps(mock_spec, indent=2)}

Implement and execute this experiment. 
Since this is a PINN, you must generate the collocation points (collocation_points=5000) internally in your script.
Ensure raw results (residuals) are saved to 'raw_results.json'.
"""

    print("--- Running Code & Execution Agent (PINN Test) ---\n")
    result = Runner.run_sync(code_execution_agent, prompt)
    print("\n--- Agent Final Output ---\n")
    print(result.final_output)
