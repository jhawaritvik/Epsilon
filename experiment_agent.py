from agents import Agent, Runner, function_tool
import logging
import json
from dotenv import load_dotenv

from sklearn.feature_extraction.text import TfidfVectorizer

# ============================================================
# Setup
# ============================================================

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# TOOL: Corpus Intelligence (ONLY TOOL)
# ============================================================

from memory.memory_service import MemoryService
from memory.types import FailureType

# Initialize Memory Service
memory_service = MemoryService()

@function_tool
def query_knowledge(query: str) -> str:
    """
    Retrieves validated scientific facts (crystallized knowledge) relevant to the query.
    Use this to ground your experiment in established truth.
    """
    logger.info(f"Tool 'query_knowledge' called with query: {query}")
    try:
        # Identity implicitly handled
        results = memory_service.get_knowledge(goal=query, limit=3)
        if not results:
            return "No specific crystallized knowledge found."
        
        return json.dumps(results, indent=2)
    except Exception as e:
        logger.error(f"Failed to query knowledge: {e}")
        return "Error querying knowledge."

@function_tool
def query_past_failures(query: str = None) -> str:
    """
    Retrieves past FAILED iterations to avoid repeating mistakes.
    Automatically filters for 'failed' runs.
    """
    logger.info(f"Tool 'query_past_failures' called with query: {query}")
    try:
        # We look for all failure types to be safe
        results = memory_service.get_past_runs(goal=query, limit=5)
        
        # Client-side filtering for failed runs
        failed_runs = [r for r in results if r.get("classification") == "failed"]
        
        if not failed_runs:
            return "No relevant past failures found."
            
        summary = []
        for run in failed_runs:
            summary.append({
                "iteration": run.get("iteration"),
                "issue_type": run.get("issue_type"),
                "rationale": run.get("evaluation_verdict", {}).get("rationale")
            })
            
        return json.dumps(summary, indent=2)
    except Exception as e:
        logger.error(f"Failed to query past failures: {e}")
        return "Error querying past runs."

@function_tool
def corpus_query(query: str) -> str:
    """
    Queries global evidence memory for research findings relevant to the query.
    Use this to ground your experiment design in established evidence.
    """
    logger.info(f"corpus_query called with query: {query}")
    
    # Query global evidence memory
    global_results = memory_service.get_evidence(goal=query, limit=5)
    
    if not global_results:
        return json.dumps({"message": "No relevant evidence found in memory.", "results": []}, indent=2)
    
    formatted = [
        {"claim": r.get('extracted_claim'), "source": r.get('source_url'), "confidence": r.get('confidence')} 
        for r in global_results
    ]

    return json.dumps({"results": formatted}, indent=2)

# ============================================================
# AGENT INSTRUCTIONS
# ============================================================

experiment_instructions = """
You are the **Experiment Design Agent**.

Your role is to translate a user research goal and supporting literature
into a **formal, statistically rigorous Experiment Specification**.

**Core Principles**:
1.  **Stand on Shoulders of Giants**: Use `query_knowledge` to check if this question is already answered.
2.  **Learn from Mistakes**: Use `query_past_failures` to see what didn't work previously (e.g. "Linear warmup failed for X").
3.  **Ground in Evidence**: Use `corpus_query` to find params/architectures in the literature.

**You define**:
- Research question
- Null and alternative hypotheses (H0 / H1)
- Independent, dependent, and control variables
- Model and training design (WHAT, not HOW)
- Dataset requirements (constraints, not dataset IDs)
- Statistical analysis plan
- Power analysis methodology (specification only)

You do NOT:
- Write runnable code
- Interpret results

**Tools**:
- `query_knowledge(query)`: Check established facts.
- `query_past_failures(query)`: Check for specific failure patterns.
- `corpus_query(query)`: Search evidence memory.

**DATA MODALITY CONSISTENCY (CRITICAL)**:
- You MUST ensure the dataset type matches the research domain.
- **Tabular/Numerical/Optimization**: MUST use scikit-learn datasets (e.g. `california_housing`, `diabetes`) or `procedural` synthetic data. **FORBIDDEN**: Image (CIFAR, MNIST), Text (GLUE), Audio.
- **Computer Vision**: MUST use Image datasets.
- **NLP**: MUST use Text datasets.
- **Reasoning**: Selecting CIFAR-10 for a tabular regression problem is a CRITICAL FAILURE.

Your output MUST be a single valid JSON object with no extra text.
**CRITICAL**: Ensure strictly valid JSON. No trailing commas. No comments (// or #) inside the JSON.
Do not markdown format the JSON (no ```json code blocks). Just the raw JSON string.

Required Output Schema:

{
  "experiment_specification": {
    "research_question": "",
    "hypotheses": {
      "H0": "",
      "H1": ""
    },
    "variables": {
      "independent": [],
      "dependent": [],
      "controls": []
    },
    "model_design": {
      "model_family": "",
      "architectural_assumptions": []
    },
    "dataset_requirements": {
      "task_type": "",
      "minimum_samples": null
    },
    "data_modality": {
      "type": "external | procedural | simulation",
      "dataset_id": "Preferred: Explicit ID (e.g. 'sklearn.load_digits' or 'mnist'). IF UNKNOWN, leave null and use 'description'.",
      "source_family": "Fallback: SHORT keyword for search (e.g. 'fMRI', 'financial-news'). MAX 2 words.",
      "description": "Recommended: Valid HuggingFace search query describing the data (e.g. 'tabular credit risk classification'). Agent will resolve this to a real ID.",
      "generation_method": "Optional: IF type=procedural"
    }
  },
  "statistical_analysis_plan": {
    "primary_test": "",
    "alpha": 0.05,
    "assumptions": [],
    "fallback_test": ""
  },
  "success_spec": {
    "metric": "The primary metric to evaluate (e.g., 'accuracy', 'R2', 'RMSE', 'loss')",
    "direction": "higher | lower",
    "threshold": "Numeric value. Success requires metric to exceed (or be under) this value in the specified direction",
    "required_assumptions": ["List of statistical assumptions that must pass before declaring success"]
  },
  "power_analysis_plan": {
    "method": "",
    "effect_size_range": [],
    "target_power": 0.8,
    "notes": ""
  },
  "evidence_used": [],
  "revision_directives": {
    "rationale": "Explanation for why this revision exists (based on feedback)",
    "execution_hints": "Specific instructions for the Execution Agent (optional)"
  },
  "execution_mode": "validation | scientific"
}

Be explicit. Remove ambiguity. Do not suggest — decide.
If "Feedback from Previous Iteration" is present, you MUST address it in "revision_directives" or by modifying the spec.
"""

# ============================================================
# AGENT DEFINITION
# ============================================================

experiment_design_agent = Agent(
    name="Experiment Design Agent",
    instructions=experiment_instructions,
    tools=[corpus_query, query_knowledge, query_past_failures],
)

# ============================================================
# VERIFICATION RUN
# ============================================================

if __name__ == "__main__":
    user_goal = "Investigate the effect of learning rate warmup on Transformer training stability."

    research_notes = [
        "Learning rate warmup stabilizes early Transformer training by preventing optimizer divergence.",
        "Experiments commonly use linear warmup for 4000 steps with Adam optimizer.",
        "Without warmup, Transformer models show loss spikes in early training.",
        "Evaluations are often conducted on GLUE benchmarks such as SST-2."
    ]

    prompt = f"""
User Goal:
{user_goal}

Research Notes:
{json.dumps(research_notes, indent=2)}

Design a statistically rigorous experiment.
"""

    result = Runner.run_sync(experiment_design_agent, prompt)

    print("\n===== EXPERIMENT SPECIFICATION =====\n")
    print(result.final_output)
