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

from pathlib import Path

# Dynamic Experiment Directory
def get_experiment_dir():
    run_id = os.environ.get("CURRENT_RUN_ID")
    if run_id:
        return str(Path(__file__).parent / "experiments" / run_id)
    return str(Path(__file__).parent / "experiments")

def _ensure_experiment_dir():
    Path(get_experiment_dir()).mkdir(parents=True, exist_ok=True)

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
    EXPERIMENT_DIR = get_experiment_dir()
    _ensure_experiment_dir()

    data_modality = json.loads(data_modality_json)
    dataset_requirements = (
        json.loads(dataset_requirements_json)
        if dataset_requirements_json else None
    )

    modality_type = data_modality.get("type")

    if modality_type == "external":
        # PATH A: Explicit ID provided (User Requirement)
        dataset_id = data_modality.get("dataset_id")
        if dataset_id:
            # Check if this is already a canonical name (contains '/')
            # If not, search HuggingFace to find the canonical name
            if "/" not in dataset_id:
                try:
                    from huggingface_hub import list_datasets
                    logger.info(f"Searching HuggingFace for canonical name of '{dataset_id}'")
                    results = list(list_datasets(search=dataset_id, limit=5))
                    
                    # Find exact match or best match
                    canonical_id = None
                    for ds in results:
                        # Exact match on the dataset name part
                        if ds.id.split("/")[-1].lower() == dataset_id.lower():
                            canonical_id = ds.id
                            break
                    
                    if not canonical_id and results:
                        # Use first result as fallback
                        canonical_id = results[0].id
                    
                    if canonical_id:
                        logger.info(f"Resolved '{dataset_id}' -> '{canonical_id}'")
                        dataset_id = canonical_id
                    else:
                        logger.warning(f"Could not find canonical name for '{dataset_id}', using as-is")
                except Exception as e:
                    logger.warning(f"HuggingFace search failed: {e}. Using '{dataset_id}' as-is")
            
            resolved = {
                "dataset_source": "huggingface",
                "dataset_id": dataset_id,
                "version": "latest",
                "status": "resolved",
                "load_instruction": f"Use: load_dataset('{dataset_id}')"
            }
        
        # PATH B: Family search (Fallback)
        else:
            source = data_modality.get("source_family")
            if not source:
                raise RuntimeError("Missing source_family or dataset_id for external data modality")

            try:
                from huggingface_hub import list_datasets
                results = list(list_datasets(search=source, limit=1))

                if not results:
                    # Return structured error instead of raising exception
                    error_result = {
                        "dataset_source": "huggingface",
                        "status": "resolution_failed",
                        "error": f"No dataset found matching source_family: '{source}'",
                        "action": "Report this to the user. Consider specifying an explicit dataset_id."
                    }
                    path = os.path.join(EXPERIMENT_DIR, "dataset_used.json")
                    with open(path, "w") as f:
                        json.dump(error_result, f, indent=2)
                    return json.dumps(error_result, indent=2)

                ds = results[0]
                resolved = {
                    "dataset_source": "huggingface",
                    "dataset_id": ds.id,
                    "version": ds.last_modified,
                    "status": "resolved"
                }

            except Exception as e:
                # Return structured error for any resolution failure
                error_result = {
                    "dataset_source": "huggingface",
                    "status": "resolution_failed",
                    "error": str(e),
                    "action": "Check network connectivity or specify explicit dataset_id."
                }
                return json.dumps(error_result, indent=2)

        # PATH C: Descriptive Search (User Requested Optimization)
        # If no dataset_id is provided, use the 'description' or 'source_family' to find a match.

    # PATH C: Descriptive Search (User Requested Optimization) - Explicit Check
    if "resolved" not in locals() and modality_type == "external":
        # Check if we have a description/family but failed ID match
        description = data_modality.get("description") or data_modality.get("source_family")
        if description and not data_modality.get("dataset_id"):
             logger.info(f"Resolving via Descriptive Search: '{description}'")
             try:
                from huggingface_hub import list_datasets
                results = list(list_datasets(search=description, limit=3))
                if results:
                    best_match = results[0]
                    logger.info(f"Resolved description '{description}' -> '{best_match.id}'")
                    resolved = {
                        "dataset_source": "huggingface",
                        "dataset_id": best_match.id,
                        "version": best_match.last_modified,
                        "status": "resolved",
                        "resolution_method": "descriptive_search",
                        "load_instruction": f"Use: load_dataset('{best_match.id}')"
                    }
                else:
                    return json.dumps({
                        "dataset_source": "huggingface",
                        "status": "resolution_failed",
                        "error": f"No datasets found matching description: '{description}'",
                        "action": "Refine description or provide explicit dataset_id."
                    }, indent=2)
             except Exception as e:
                 logger.warning(f"Descriptive search failed: {e}")

    # PATH D: Procedural (Standard)
    if "resolved" not in locals() and modality_type == "procedural":
        resolved = {
            "dataset_source": "procedural",
            "generation_method": data_modality.get("generation_method"),
            "parameters": data_modality,
            "status": "resolved",
            "instruction": "Generate data procedurally inside experiment code."
        }
        
    # PATH E: Simulation
    elif "resolved" not in locals() and modality_type == "simulation":
        resolved = {
            "dataset_source": "simulation",
            "generator": data_modality.get("generator"),
            "parameters": data_modality,
            "status": "resolved"
        }

    # Final Check
    if "resolved" not in locals():
        raise RuntimeError(f"Unsupported data_modality.type: {modality_type} or failed resolution.")



    if dataset_requirements:
        resolved["requirements"] = dataset_requirements

    path = os.path.join(EXPERIMENT_DIR, "dataset_used.json")
    with open(path, "w") as f:
        json.dump(resolved, f, indent=2)

    return json.dumps(resolved, indent=2)

@function_tool
def execute_experiment(code: str, execution_mode: str = "validation") -> str:
    """
    Saves the provided Python code to 'run_experiment.py' and executes it.
    Captures stdout, stderr, and returns the execution summary.
    """
    # Use explicit argument
    logger.info(f"execute_experiment called (mode={execution_mode})")
    EXPERIMENT_DIR = get_experiment_dir()
    _ensure_experiment_dir()
    
    # ----------------------------------------------------
    # CONTRACT ENFORCEMENT LAYER (Behavioral Guard)
    # ----------------------------------------------------
    dataset_info_path = os.path.join(EXPERIMENT_DIR, "dataset_used.json")
    if os.path.exists(dataset_info_path):
        try:
            with open(dataset_info_path, "r") as f:
                dataset_meta = json.load(f)
                
            source = dataset_meta.get("dataset_source", "unknown")
            
            # 1. Procedural Invariant: No external data loaders
            # 1. Procedural Invariant: No external data loaders
            if source == "procedural" or source == "simulation":
                # Forbidden: Downloading external data
                forbidden_loaders = ["torchvision.datasets", "keras.datasets", "tensorflow_datasets"]
                
                # Check for sklearn loaders (load_*) but ALLOW generators (make_*)
                if "sklearn.datasets" in code:
                    if "load_" in code or "fetch_" in code:
                         return f"CONTRACT VIOLATION: Dataset source is '{source}'. You cannot use 'load_' or 'fetch_' from sklearn.datasets. You MAY use 'make_' functions (e.g. make_classification)."

                for bad in forbidden_loaders:
                    if bad in code:
                        return f"CONTRACT VIOLATION: Dataset source is '{source}', but code uses '{bad}'. This is strictly forbidden. You must generate data internally (e.g. numpy, make_classification)."
                        
            # 2. External Invariant: Must use authorised loaders if specified
            # (Future: Check strictly for the dataset_id)
            
        except Exception as e:
            logger.warning(f"Could not verify contract: {e}")
    else:
        # If no resolution happened, we might warn, but let's allow for now (backward compat)
        pass

    file_path = os.path.join(EXPERIMENT_DIR, "run_experiment.py")
    with open(file_path, "w") as f:
        f.write(code)
        
    # Set timeout based on mode
    timeout_seconds = 300 # Default / Validation
    if execution_mode == "scientific":
        timeout_seconds = 1800 # 30 mins
    
    try:
        # Execute the script INSIDE the experiments directory
        # This ensures all artifacts (raw_results.json) are saved there
        result = subprocess.run(
            [sys.executable, "run_experiment.py"],
            cwd=EXPERIMENT_DIR,
            capture_output=True,
            text=True,
            timeout=timeout_seconds
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
3. **Autonomous Execution**: Use the `execute_experiment` tool to run your generated code.
   - You MUST pass the `execution_mode` provided in the input (default to "validation").
   - `execute_experiment(code=..., execution_mode=...)`
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
   - **BINDING CLAUSE**: You MUST use the STRICT output of `dataset_resolver`.
     - If `external`: Load ONLY the `dataset_id` returned. Do NOT substitute (e.g., do not load 'mnist' if resolver returned 'fashion_mnist').
     - If `procedural`: Implement the generation logic exactly as instructed.
     - **FORBIDDEN**: Do not use `sklearn.datasets.load_*` or `torchvision.datasets.*` unless explicitly returned by the resolver.
   - Log raw metrics to `raw_results.json`.
     - **CRITICAL**: Save strictly to `.`, i.e., `raw_results.json`. **DO NOT create an 'artifacts' subdirectory**.
     - **CRITICAL**: Convert ALL numpy types to native Python.
4. Call `execute_experiment` with the generated code and the `execution_mode` from the input.
5. **MANDATORY Iterative Debugging**:
   - If `execute_experiment` returns a failure (e.g., ImportError, SyntaxError, runtime error):
     - **DO NOT** return a "failed" status immediately.
     - **DO NOT** give up.
     - **DIAGNOSE** the error from the logs.
     - **REGENERATE** the full `run_experiment.py` code with the fix.
     - **RE-EXECUTE** by calling `execute_experiment` again.
   - You MUST attempt at least 3 repair cycles before declaring failure.
6. Confirm all artifacts (`run_experiment.py`, `dataset_used.json`, `raw_results.json`, `execution.log`) are produced.

Your output MUST be a valid JSON object with the following schema:
{
  "execution_status": "success | failed | partial",
  "artifacts": ["list", "of", "files", "created"],
  "summary": "Brief summary of what happened"
}

DO NOT output conversational text. ONLY output the JSON.
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
