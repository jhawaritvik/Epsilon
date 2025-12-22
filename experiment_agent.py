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
Inputs:
- User Goal
- Research Corpus
- Feedback from Previous Iteration (if any)

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
      "minimum_samples": null
    },
    "data_modality": {
      "type": "external | procedural | simulation",
      "dataset_id": "Optional: explicit resolvable dataset ID (e.g., cifar10)",
      "source_family": "Optional: IF type=external AND resolver choice allowed",
      "generation_method": "Optional: IF type=procedural",
      "description": ""
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
  "evidence_used": [],
  "revision_directives": {
    "rationale": "Explanation for why this revision exists (based on feedback)",
    "execution_hints": "Specific instructions for the Execution Agent (optional)"
  },
  "execution_mode": "validation | scientific"
}

Be explicit. Remove ambiguity. Do not suggest — decide.
If "Feedback from Previous Iteration" is present, you MUST address it in "revision_directives" or by modifying the spec.

EXECUTION MODES:
- "validation" (Default):
  - Purpose: Verify pipeline correctness, metric logging, and artifact generation.
  - Constraints: Short timeout (300s), small datasets (toy/subsampled), minimal epochs/seeds.
  - Use this for the FIRST iteration or when debugging.

- "scientific":
  - Purpose: Rigorous hypothesis testing and effect size estimation.
  - Constraints: Long timeout (1800s), full datasets, multiple seeds (N>=5).
  - Use this ONLY after a successful "validation" run has proven the code works but lacked statistical power.

DATA RESOLUTION CONSTRAINT (MANDATORY):

If you specify:
- data_modality.type = "external"

Then you MUST provide ONE of:
- dataset_id (explicit, resolvable)
- source_family with acceptance criteria explicitly allowing resolver choice

If neither is possible, you MUST choose:
- data_modality.type = "procedural" or "simulation"

Do NOT propose experiments that require external data without a resolvable acquisition path.

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
