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
# Docker Configuration
# ============================================================

# Set USE_DOCKER=true to execute experiments in Docker containers
# Falls back to local execution if Docker is unavailable
USE_DOCKER = os.environ.get("USE_DOCKER", "false").lower() == "true"

# Lazy import for Docker executor (only when needed)
_docker_executor = None
_docker_available = None

def _get_docker_executor():
    """Lazy initialization of Docker executor."""
    global _docker_executor, _docker_available
    
    if _docker_available is None:
        try:
            from docker_executor import DockerExecutor, is_docker_available
            _docker_available = is_docker_available()
            if _docker_available:
                _docker_executor = DockerExecutor()
                logger.info("Docker executor initialized successfully")
            else:
                logger.info("Docker not available, will use local execution")
        except ImportError as e:
            logger.warning(f"Docker executor module not found: {e}")
            _docker_available = False
        except Exception as e:
            logger.warning(f"Failed to initialize Docker executor: {e}")
            _docker_available = False
    
    return _docker_executor if _docker_available else None

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
                    "version": ds.last_modified.isoformat() if hasattr(ds.last_modified, 'isoformat') else str(ds.last_modified),
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
                        "version": best_match.last_modified.isoformat() if hasattr(best_match.last_modified, 'isoformat') else str(best_match.last_modified),
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
    
    Execution can occur in Docker (if USE_DOCKER=true and Docker is available)
    or locally via subprocess (fallback).
    """
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
            
            # Procedural Invariant: No external data loaders
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
                        
        except Exception as e:
            logger.warning(f"Could not verify contract: {e}")
        
    # Set timeout based on mode
    timeout_seconds = 300  # Default / Validation
    if execution_mode == "scientific":
        timeout_seconds = 1800  # 30 mins
    
    # ----------------------------------------------------
    # EXECUTION: Docker or Local
    # ----------------------------------------------------
    
    # Try Docker execution if enabled
    if USE_DOCKER:
        docker_executor = _get_docker_executor()
        if docker_executor:
            logger.info("Executing experiment in Docker container")
            return _execute_in_docker(docker_executor, code, EXPERIMENT_DIR, timeout_seconds)
        else:
            logger.info("Docker unavailable, falling back to local execution")
    
    # Local execution (default or fallback)
    return _execute_locally(code, EXPERIMENT_DIR, timeout_seconds)


def _execute_in_docker(executor, code: str, experiment_dir: str, timeout: int) -> str:
    """
    Execute code inside a Docker container.
    
    Args:
        executor: DockerExecutor instance.
        code: Python code to execute.
        experiment_dir: Directory for artifacts.
        timeout: Execution timeout in seconds.
        
    Returns:
        Execution result message.
    """
    try:
        result = executor.execute_code(
            code=code,
            experiment_dir=experiment_dir,
            timeout=timeout
        )
        
        if result.success:
            return f"Execution successful (Docker).\nLog saved to execution.log"
        elif result.timed_out:
            return "Execution timed out (Docker)."
        else:
            return (
                f"Execution failed (Docker, Return Code {result.return_code}).\n\n"
                f"STDERR:\n{result.stderr}\n\n"
                f"STDOUT:\n{result.stdout}"
            )
            
    except Exception as e:
        logger.error(f"Docker execution failed: {e}")
        # Fallback to local execution on Docker failure
        logger.info("Falling back to local execution due to Docker error")
        return _execute_locally(code, experiment_dir, timeout)


def _execute_locally(code: str, experiment_dir: str, timeout: int) -> str:
    """
    Execute code locally via subprocess.
    
    Args:
        code: Python code to execute.
        experiment_dir: Directory for artifacts.
        timeout: Execution timeout in seconds.
        
    Returns:
        Execution result message.
    """
    file_path = os.path.join(experiment_dir, "run_experiment.py")
    with open(file_path, "w") as f:
        f.write(code)
    
    try:
        # Execute the script INSIDE the experiments directory
        result = subprocess.run(
            [sys.executable, "run_experiment.py"],
            cwd=experiment_dir,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        execution_log = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        
        log_path = os.path.join(experiment_dir, "execution.log")
        with open(log_path, "w") as f:
            f.write(execution_log)
            
        if result.returncode == 0:
            return f"Execution successful.\nLog saved to execution.log"
        else:
            return (
                f"Execution failed (Return Code {result.returncode}).\n\n"
                f"STDERR:\n{result.stderr}\n\n"
                f"STDOUT:\n{result.stdout}"
            )
            
    except subprocess.TimeoutExpired:
        return "Execution timed out."
    except Exception as e:
        return f"An error occurred during execution: {e}"

@function_tool
def install_package(package_name: str) -> str:
    """
    Installs a Python package into the current environment using pip.
    Use this ONLY when you encounter a `ModuleNotFoundError` or need a specific library not present.
    
    Note: When using Docker execution, packages should be pre-installed in the Docker image.
    This function installs locally; for Docker, rebuild the image with the new package.
    """
    logger.info(f"install_package called for: {package_name}")
    
    # Check if we're in Docker mode - warn user about package management
    if USE_DOCKER and _get_docker_executor():
        logger.warning(
            f"Docker mode active. Package '{package_name}' will be installed locally, "
            "but Docker container uses pre-built image. Consider adding to requirements.txt "
            "and rebuilding the Docker image."
        )
    
    try:
        # Use module pip to ensure it installs in the current python environment
        cmd = [sys.executable, "-m", "pip", "install", package_name]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            return f"Successfully installed '{package_name}'. Output:\n{result.stdout}"
        else:
            return f"Failed to install '{package_name}'. Error:\n{result.stderr}"
    except Exception as e:
        return f"Error executing pip install: {e}"

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
4. **Self-Correction & Dependency Management**:
   - If execution fails with `ModuleNotFoundError`, use the `install_package` tool to install the missing library.
   - Then, RE-EXECUTE the experiment.
   - Do not ask for permission; just fix the environment.
   - General errors: Diagnose from logs, regenerate code, and retry (at least 3 attempts).
5. **Evidence Generation**: Ensure all outputs (metrics, logs) are saved as machine-readable artifacts.
6. **MANDATORY VISUALIZATION**: You MUST generate at least one visualization using `matplotlib` to illustrate your findings (e.g., loss curve, comparison bar chart).
   - Save it as a PNG file, e.g., `comparison_plot.png`.
   - Ensure the plot has a title, labels, and legend.
   - Do NOT use `plt.show()`; use `plt.savefig()`.

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
   - If `execute_experiment` returns a failure:
     - Check for `ModuleNotFoundError`. If found -> `install_package("missing_lib")`.
     - Other errors -> regenerate code.
     - **RE-EXECUTE** the tool.
   - You MUST attempt at least 3 repair cycles before declaring failure.
6. Confirm all artifacts (`run_experiment.py`, `dataset_used.json`, `raw_results.json`, `execution.log`, `*.png`) are produced.

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
    model=os.getenv("MODEL_NAME", "gpt-5.2"),
    instructions=execution_instructions,
    tools=[dataset_resolver, execute_experiment, install_package],
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
