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

@function_tool
def corpus_query(query: str, notes_content: list[str]) -> str:
    """
    Evidence-grounded retrieval over research notes.
    Used ONLY to justify experimental design choices.
    """
    logger.info(f"corpus_query called with query: {query}")

    if not notes_content:
        return json.dumps([])

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(notes_content + [query])

    query_vec = tfidf[-1]
    similarities = (tfidf[:-1] * query_vec.T).toarray().flatten()

    top_indices = similarities.argsort()[-3:][::-1]

    results = []
    for idx in top_indices:
        if similarities[idx] > 0:
            results.append({
                "note_index": int(idx),
                "similarity": float(similarities[idx]),
                "excerpt": notes_content[idx][:200]
            })

    return json.dumps(results, indent=2)

# ============================================================
# AGENT INSTRUCTIONS
# ============================================================

experiment_instructions = """
You are the **Experiment Design Agent**.

Your role is to translate a user research goal and supporting literature
into a **formal, statistically rigorous Experiment Specification**.

You define:
- Research question
- Null and alternative hypotheses (H0 / H1)
- Independent, dependent, and control variables
- Model and training design (WHAT, not HOW)
- Dataset requirements (constraints, not dataset IDs)
- Statistical analysis plan
- Power analysis methodology (specification only)

You do NOT:
- Write runnable code
- Choose implementation details
- Execute experiments
- Compute statistics or p-values
- Interpret results

You may use tools ONLY to retrieve evidence from the research corpus.
All experimental and statistical specifications must be produced
through explicit reasoning and written declarations.

Your output MUST be a single valid JSON object with no extra text.

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
      "benchmark_family": "",
      "minimum_samples": null,
      "splits_required": []
    }
  },
  "statistical_analysis_plan": {
    "primary_test": "",
    "alpha": 0.05,
    "assumptions": [],
    "fallback_test": ""
  },
  "power_analysis_plan": {
    "method": "",
    "effect_size_range": [],
    "target_power": 0.8,
    "notes": ""
  },
  "evidence_used": []
}

Be explicit. Remove ambiguity. Do not suggest — decide.
"""

# ============================================================
# AGENT DEFINITION
# ============================================================

experiment_design_agent = Agent(
    name="Experiment Design Agent",
    instructions=experiment_instructions,
    tools=[corpus_query],  # minimal, correct
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
