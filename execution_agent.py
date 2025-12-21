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
def dataset_resolver(
    data_modality_json: str,
    dataset_requirements_json: str = None
) -> str:
    """
    Resolves data acquisition STRICTLY from explicit design signals.
    Inputs are JSON strings to comply with strict schema requirements.
    """
    logger.info("dataset_resolver called")
    _ensure_experiment_dir()

    data_modality = json.loads(data_modality_json)
    dataset_requirements = (
        json.loads(dataset_requirements_json)
        if dataset_requirements_json else None
    )

    modality_type = data_modality.get("type")

    if modality_type == "external":
        source = data_modality.get("source_family")
        if not source:
            raise RuntimeError("Missing source_family for external data modality")

        try:
            from huggingface_hub import list_datasets
            results = list(list_datasets(search=source, limit=1))

            if not results:
                raise RuntimeError(
                    "No dataset satisfies dataset_requirements. "
                    "Escalate to Evaluation → Design."
                )

            ds = results[0]
            resolved = {
                "dataset_source": "huggingface",
                "dataset_id": ds.id,
                "version": ds.last_modified,
                "status": "resolved"
            }

        except Exception as e:
            raise RuntimeError(f"Dataset resolution failed: {e}")

    elif modality_type == "procedural":
        resolved = {
            "dataset_source": "procedural",
            "generation_method": data_modality.get("generation_method"),
            "parameters": data_modality,
            "status": "resolved",
            "instruction": "Generate data procedurally inside experiment code."
        }

    elif modality_type == "simulation":
        resolved = {
            "dataset_source": "simulation",
            "generator": data_modality.get("generator"),
            "parameters": data_modality,
            "status": "resolved"
        }

    else:
        raise RuntimeError(
            f"Unsupported data_modality.type: {modality_type}"
        )

    if dataset_requirements:
        resolved["requirements"] = dataset_requirements

    path = os.path.join(EXPERIMENT_DIR, "dataset_used.json")
    with open(path, "w") as f:
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

**The Correct Compromise (This Is the Lock)**:
The Code Agent does not decide how data should be obtained; it infers the data acquisition mechanism exclusively from explicit design signals and executes it deterministically.

**Your Responsibilities**:
1. **Deterministic Implementation**: Translate the provided Experiment Specification (JSON) into a standalone Python script (`run_experiment.py`).
   - **Revision Directives**: If `revision_directives` or `execution_hints` are present in the input, you MUST prioritize them. They are corrective feedback from previous runs.
2. **Dataset Resolution**: Use the `dataset_resolver` tool to identify the data source.
   - You MUST pass the `data_modality` and `dataset_requirements` as **JSON strings** (use `json.dumps`).
   - **Allowed Actions based on `data_modality.type`**:
     - `external` → load dataset (e.g. huggingface)
     - `procedural` → generate samples (implement generation logic)
     - `simulation` → run solver
   - **Not Allowed**:
     - Decide whether data should be procedural
     - Override modality
     - "Fallback" to synthetic data
     - Mix modes without instruction
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
2. Call `dataset_resolver` with `json.dumps(data_modality)` and `json.dumps(dataset_requirements)`.
3. Generate the Python code for the experiment:
   - Handle data loading/generation based ONLY on resolution result.
   - Log raw metrics to `raw_results.json`.
4. Call `execute_experiment` with the generated code.
5. **Iterative Debugging**:
   - If `execute_experiment` returns a failure, READ the error message carefully.
   - ANALYZE why it failed.
   - REGENERATE the `run_experiment.py` code with specific fixes.
   - CALL `execute_experiment` again.
   - Repeat up to 5 times.
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
                "minimum_samples": 5000
            },
            "data_modality": {
                "type": "procedural",
                "generation_method": "uniform_collocation_sampling",
                "domain": "x ∈ [0,1], t ∈ [0,1]"
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
