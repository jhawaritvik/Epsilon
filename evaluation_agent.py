from agents import Agent, Runner, function_tool
import logging
import json
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
    logger.info(f"run_statistical_test called: test={test_name}, alpha={alpha}, alternative={alternative}")
    
    try:
        result = {}
        decision = "Fail to reject H0"
        
        # Scipy uses 'alternative' kwarg for ttest_ind, ttest_rel, mannwhitneyu, wilcoxon
        # Common mapping: 
        #   scipy 'less' means distribution underlying first sample is stricly less than the underlying distribution of the second sample
        #   So if data_a is Treatment and data_b is Baseline:
        #   H1: Treatment < Baseline => alternative='less'
        
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

@function_tool
def verify_assumptions(check_type: str, data: list[float]) -> str:
    """
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

# ============================================================
# AGENT INSTRUCTIONS
# ============================================================

evaluation_instructions = """
You are the **Evaluation / Analysis Agent**.

Your role is to act as a **Statistical Executor** and **Scientific Validator**.
You strictly execute the analysis protocol defined by the Design Agent.

**Responsibilities**:
1.  **Verify Assumptions FIRST**: 
    - Execute `verify_assumptions` for checks listed in `analysis_protocol` (e.g., normality).
    - **Gating Logic**:
      - IF any assumption FAILS: You MUST NOT run the `primary_test`. Instead, run the `fallback_test` defined in the protocol.
      - IF all assumptions PASS: Run the `primary_test`.
2.  **Execute Statistical Test**: 
    - Run the selected test (primary or fallback) using `run_statistical_test`.
    - **Directionality**: You MUST pass the `alternative` argument ("less", "greater", "two-sided") as specified in the protocol.
3.  **Judge Results**: Compare p-values to $\alpha$ and declare `Reject H0` / `Fail to reject H0`.
4.  **Classify Outcome**: 
    - Use the `classification_rules` from the protocol to label results (e.g., `robust`, `spurious`, `promising`).
    - strictly matching the rule conditions to your findings (Verdict + Assumption Status).

**Inputs provided to you**:
- `experiment_specification` (JSON): Contains Research Question, $H_0$, $H_1$.
- `analysis_protocol` (JSON): The authoritative guide.
  - `primary_test`: { "name": "...", "alternative": "..." }
  - `fallback_test`: { "name": "...", "alternative": "..." }
  - `assumptions`: ["normality", ...]
  - `classification_rules`: { "robust": [...], "promising": [...] }
- `observations` (JSON): The raw data/metrics.

**Constraints**:
- **Strict Gating**: Never run a parametric test (e.g. t-test) if normality assumption fails.
- **Strict Directionality**: Always respect the H1 direction (e.g. loss B < loss A => 'less').
- **Strict Output**: Your final answer must be a valid JSON object.

**Required Output Schema**:
```json
{
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
  "verdict": {
    "hypothesis_decision": "Reject H0 | Fail to reject H0",
    "outcome_classification": "robust | promising | spurious | failed",
    "rationale": "string"
  }
}
```
"""

# ============================================================
# AGENT DEFINITION
# ============================================================

evaluation_agent = Agent(
    name="Evaluation Agent",
    instructions=evaluation_instructions,
    tools=[run_statistical_test, verify_assumptions],
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
