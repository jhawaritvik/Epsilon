from agents import Agent, Runner, function_tool
import logging
import json
import os
import numpy as np
import scipy.stats as stats
import pandas as pd
from dotenv import load_dotenv

# ============================================================
# Setup
# ============================================================

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# TOOLS: Statistical Execution
# ============================================================

def _run_statistical_test_impl(test_name: str, data_a: list[float], data_b: list[float] = None, alpha: float = 0.05, alternative: str = "two-sided") -> str:
    """
    Raw implementation of statistical test execution.
    This function is directly callable for testing purposes.
    
    Executes a specific statistical test on the provided data.
    Supported tests: 't-test_ind', 't-test_rel', 'mannwhitneyu', 'wilcoxon', 'shapiro'.
    Args:
        alternative: 'two-sided', 'less', 'greater'. 
                     For 'less', H1 is mean(a) < mean(b) [typical for loss].
                     For 'greater', H1 is mean(a) > mean(b) [typical for accuracy].
    Returns JSON with statistic, p-value, and decision (Reject H0 / Fail to Reject H0).
    """
    logger.info(f"run_statistical_test called: test={test_name}, alpha={alpha}, alternative={alternative}")
    
    try:
        result = {}
        decision = "Fail to reject H0"
        
        # Scipy uses 'alternative' kwarg for ttest_ind, ttest_rel, mannwhitneyu, wilcoxon
        # Common mapping: 
        #   scipy 'less' means distribution underlying first sample is stricly less than the underlying distribution of the second sample
        #   So if data_a is Treatment and data_b is Baseline:
        #   H1: Treatment < Baseline => alternative='less'
        # Data Check: Zero value suppression
        # If arrays are identical or effectively identical, p-value calc will fail or warn.
        # This often happens in procedural runs with fixed seeds or simplistic models.
        if data_b and np.allclose(data_a, data_b, atol=1e-9):
            logger.warning("Data arrays are identical. Skipping statistical test to avoid RuntimeWarning.")
            return json.dumps({
                "statistic": 0.0,
                "p_value": 1.0, 
                "decision": "Fail to reject H0",
                "alpha_used": alpha,
                "test_used": test_name,
                "note": "Identical inputs detected. Variance is zero."
            }, indent=2)

        if test_name == "t-test_ind":
            stat, p_val = stats.ttest_ind(data_a, data_b, alternative=alternative)
            result = {"statistic": stat, "p_value": p_val}
            
        elif test_name == "t-test_rel":
            stat, p_val = stats.ttest_rel(data_a, data_b, alternative=alternative)
            result = {"statistic": stat, "p_value": p_val}
            
        elif test_name == "mannwhitneyu":
            stat, p_val = stats.mannwhitneyu(data_a, data_b, alternative=alternative)
            result = {"statistic": stat, "p_value": p_val}
            
        elif test_name == "wilcoxon":
            stat, p_val = stats.wilcoxon(data_a, data_b, alternative=alternative)
            result = {"statistic": stat, "p_value": p_val}
            
        elif test_name == "shapiro":
            # Normality check for single sample
            stat, p_val = stats.shapiro(data_a)
            result = {"statistic": stat, "p_value": p_val}
            
        else:
            return json.dumps({"error": f"Unsupported test: {test_name}"})

        # Decision Logic
        if result["p_value"] < alpha:
            decision = "Reject H0"
        
        result["decision"] = decision
        result["alpha_used"] = alpha
        result["test_used"] = test_name
        result["alternative"] = alternative
        
        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})

# Wrap for agent use
@function_tool
def run_statistical_test(test_name: str, data_a: list[float], data_b: list[float] = None, alpha: float = 0.05, alternative: str = "two-sided") -> str:
    """
    Executes a specific statistical test on the provided data.
    Supported tests: 't-test_ind', 't-test_rel', 'mannwhitneyu', 'wilcoxon', 'shapiro'.
    Args:
        alternative: 'two-sided', 'less', 'greater'. 
                     For 'less', H1 is mean(a) < mean(b) [typical for loss].
                     For 'greater', H1 is mean(a) > mean(b) [typical for accuracy].
    Returns JSON with statistic, p-value, and decision (Reject H0 / Fail to Reject H0).
    """
    return _run_statistical_test_impl(test_name, data_a, data_b, alpha, alternative)

def _verify_assumptions_impl(check_type: str, data: list[float]) -> str:
    """
    Raw implementation of assumption verification.
    This function is directly callable for testing purposes.
    
    Verifies statistical assumptions.
    Supported checks: 'normality' (Shapiro-Wilk).
    Returns JSON with pass/fail status.
    """
    logger.info(f"verify_assumptions called: check={check_type}")
    
    if check_type == "normality":
        # Using Shapiro-Wilk with typical alpha=0.05 for assumption checking
        stat, p_val = stats.shapiro(data)
        is_normal = p_val > 0.05 # H0: Data is normal. If p > 0.05, we fail to reject, so it's "normal enough"
        return json.dumps({
            "check": "normality",
            "test": "shapiro",
            "p_value": p_val,
            "status": "PASS" if is_normal else "FAIL",
            "interpretation": "Data appears normally distributed" if is_normal else "Data deviates from normality"
        }, indent=2)
        
    return json.dumps({"error": f"Unsupported check: {check_type}"})

# Wrap for agent use
@function_tool
def verify_assumptions(check_type: str, data: list[float]) -> str:
    """
    Verifies statistical assumptions.
    Supported checks: 'normality' (Shapiro-Wilk).
    Returns JSON with pass/fail status.
    """
    return _verify_assumptions_impl(check_type, data)

@function_tool
def python_analysis_tool(code: str) -> str:
    """
    Executes Python code to analyze data files (e.g. .json, .csv) directly.
    Pre-imported libraries: pandas as pd, numpy as np, scipy.stats as stats, json.
    
    Usage Guidelines:
    1. READ data from 'experiments/raw_results.json' (or other provided paths).
    2. PERFORM statistical tests or data aggregation.
    3. PRINT key results to stdout (e.g., p-values, means).
    4. The stdout of your code will be returned to you.
    
    Example:
    code = \"\"\"
    import json
    import pandas as pd
    from scipy import stats
    
    # Load Data
    with open('experiments/raw_results.json', 'r') as f:
        data = json.load(f)
    
    # Process
    df = pd.DataFrame(data['results'])
    stat, p_val = stats.ttest_1samp(df['value'], 0.0)
    print(f"T-statistic: {stat}, P-value: {p_val}")
    \"\"\"
    """
    logger.info("python_analysis_tool called")
    
    # Capture stdout
    import io
    from contextlib import redirect_stdout
    
    f = io.StringIO()
    try:
        # Restricted globals for safety context (though we trust the agent in this local environment)
        # We explicitly provide necessary libraries
        exec_globals = {
            "pd": pd,
            "np": np,
            "stats": stats,
            "json": json,
            "math": __import__("math"),
            "os": __import__("os"),
        }
        
        with redirect_stdout(f):
            exec(code, exec_globals)
        
        output = f.getvalue()
        return output if output else "Code executed successfully (no output)."
        
    except Exception as e:
        return f"Error executing code: {e}"

# ============================================================
# AGENT INSTRUCTIONS
# ============================================================

evaluation_instructions = """
You are the **Evaluation / Analysis Agent**.

Your role is to act as a **Statistical Executor** and **Scientific Validator**.
You strictly execute the analysis protocol defined by the Design Agent.

**ANTI-HALLUCINATION PROTOCOL (CRITICAL)**:
1. You MUST read actual data from files using `python_analysis_tool`
2. You MUST NOT invent, assume, or fabricate any numbers
3. If a file doesn't exist or is empty, you MUST report `failed` with `execution` issue_type
4. Every number in your output MUST come from actual file contents or tool outputs
5. If you cannot verify a metric, set it to null and explain why

**CRITICAL: Domain-Agnostic Evaluation**
You do NOT decide what metric is meaningful or what constitutes success.
You consume the `success_spec` from the Design Agent BLINDLY and validate against it.

**Responsibilities**:
1.  **MANDATORY FILE VERIFICATION**:
    - FIRST, use `python_analysis_tool` to check if the data file exists
    - If it does NOT exist: immediately return `failed` with issue_type `execution`
    - Load the file and verify it contains the expected fields
    
2.  **Analyze Data via Code**: 
    - You will receive a **filepath** to the data (e.g., `experiments/<run_id>/raw_results.json`).
    - **DO NOT** ask to see the raw data in chat.
    - **USE `python_analysis_tool`** to write and execute Python code to load and analyze this file.
    - Calculate statistics (means, p-values, normality checks) using libraries like `scipy.stats` and `pandas`.

3.  **Consume success_spec (BLINDLY)**:
    The Design Agent provides a `success_spec` object. You MUST use it as-is:
    ```
    "success_spec": {
      "metric": "accuracy",      # What to measure - DO NOT QUESTION THIS
      "direction": "higher",     # higher = success if metric > threshold
      "threshold": 0.7,          # The bar to clear
      "required_assumptions": ["normality"]  # Must pass before declaring success
    }
    ```
    - Validate each assumption in `required_assumptions`
    - Compare the metric value against `threshold` using `direction`
    - You do NOT decide if "accuracy" is the right metric - Design Agent decided that

4.  **Verify Assumptions**: 
    - Use your Python tool to perform checks (e.g., Shapiro-Wilk) on the data file.
    - **Gating Logic**:
      - IF any assumption in `required_assumptions` FAILS: Use `fallback_test`, set issue_type to `assumption_violation`

5.  **Judge Results Using success_spec**:
    - If `direction` is "higher": success requires metric >= threshold
    - If `direction` is "lower": success requires metric <= threshold
    - Compare p-values to alpha and declare `Reject H0` / `Fail to reject H0`.

6.  **Classify Outcome with Specific Failure Types**:
    - `robust`: All assumptions pass AND threshold met AND H0 rejected
    - `promising`: Threshold met but assumptions or significance unclear
    - `spurious`: H0 rejected but assumptions fail
    - `failed`: Threshold NOT met or intervention worse than baseline
    
    **Specific Issue Types** (be precise):
    - `design`: Flawed hypothesis, H0 not rejected, intervention worse than baseline
    - `data_insufficiency`: Not enough samples for statistical power
    - `assumption_violation`: Required assumptions failed
    - `non_convergence`: Model/optimization did not converge
    - `execution`: Code errors, missing files, anomalous results
    - `target_not_met`: Results valid but did not achieve the threshold
    - `none`: Success (outcome is `robust`)

**Inputs provided to you**:
- `experiment_specification` (JSON): Contains Research Question, H0, H1
- `analysis_protocol` (JSON): Contains primary_test, alpha, fallback_test
- `success_spec` (JSON): The authoritative success criteria from Design Agent
- `data_path` (str): Path to the results file

**MANDATORY VALIDATION STEPS**:
1. VERIFY file exists using python code (os.path.exists)
2. LOAD the actual file contents - do NOT assume any values
3. EXTRACT the metric value from the data
4. COMPARE against threshold: achieved_value vs threshold
5. INCLUDE explicit comparison string in rationale: "Achieved X, Target was Y, Result: PASS/FAIL"

**Required Output Schema**:
```json
{
  "file_verification": {
    "file_exists": boolean,
    "file_size_bytes": int or null,
    "parse_successful": boolean,
    "error": string or null
  },
  "statistical_results": [
    {
      "test_name": "string",
      "p_value": float,
      "statistic": float,
      "decision": "Reject H0 | Fail to reject H0",
      "alpha": float
    }
  ],
  "assumption_checks": [
    {
      "check": "string",
      "status": "PASS | FAIL"
    }
  ],
  "target_validation": {
    "metric_name": "string",
    "achieved_value": float,
    "target_value": float,
    "direction": "higher | lower",
    "target_met": boolean,
    "comparison": "Achieved X, Target Y, Result: PASS/FAIL"
  },
  "metric_comparison": {
    "baseline_mean": float,
    "intervention_mean": float,
    "improvement_achieved": boolean,
    "improvement_pct": float
  },
  "verdict": {
    "hypothesis_decision": "Reject H0 | Fail to reject H0",
    "outcome_classification": "robust | promising | spurious | failed",
    "issue_type": "design | data_insufficiency | assumption_violation | non_convergence | execution | target_not_met | none",
    "rationale": "string - MUST include: Achieved X, Target Y, Result: PASS/FAIL"
  }
}
```
"""

# ============================================================
# AGENT DEFINITION
# ============================================================

evaluation_agent = Agent(
    name="Evaluation Agent",
    model=os.getenv("MODEL_NAME", "gpt-5.2"),
    instructions=evaluation_instructions,
    tools=[python_analysis_tool, run_statistical_test, verify_assumptions],
)

# ============================================================
# VERIFICATION RUN
# ============================================================

if __name__ == "__main__":
    
    # Mock Data: Experiment showing clear improvement
    # Group A (Baseline): Normal(0.5, 0.1)
    # Group B (Treatment): Normal(0.3, 0.1) -> Lower loss is better
    np.random.seed(42)
    group_a = np.random.normal(0.5, 0.1, 30).tolist()
    group_b = np.random.normal(0.3, 0.1, 30).tolist()
    
    mock_input = {
        "experiment_specification": {
            "research_question": "Does the new optimizer reduce loss?",
            "hypotheses": {
                "H0": "The new optimizer does not reduce loss (Mean B >= Mean A).",
                "H1": "The new optimizer reduces loss (Mean B < Mean A)."
            }
        },
        "analysis_protocol": {
            "primary_test": {
                "name": "t-test_ind",
                "alternative": "greater" # H1: Baseline > Treatment (Loss Reduction)
            },
            "fallback_test": {
                "name": "mannwhitneyu",
                "alternative": "greater"
            },
            "alpha": 0.05,
            "assumptions": ["normality"],
            "classification_rules": {
                "robust": ["Reject H0", "Assumptions PASS"],
                "spurious": ["Reject H0", "Assumptions FAIL"],
                "failed": ["Fail to reject H0"]
            }
        },
        "observations": {
            "baseline_loss": group_a,
            "treatment_loss": group_b
        }
    }
    
    prompt = f"""
    Perform the evaluation based on the following context:
    {json.dumps(mock_input, indent=2)}
    
    Group A Data (Baseline): {group_a}
    Group B Data (Treatment): {group_b}
    """
    
    print("--- Running Evaluation Agent (Mock Verification) ---\n")
    result = Runner.run_sync(evaluation_agent, prompt)
    print("\n--- Agent Final Output ---\n")
    print(result.final_output)
