import logging
import uuid
from typing import Dict, Any
from .supabase_client import SupabaseManager

logger = logging.getLogger("KnowledgeMemory")

class KnowledgeMemoryWriter:
    def __init__(self):
        self.manager = SupabaseManager()

    def record_knowledge(self,
                         run_id: str,
                         experiment_spec: Dict[str, Any],
                         evaluation_verdict: Dict[str, Any]):
        """
        Records a validated scientific conclusion to knowledge_memory.
        STRICT RULES:
        - Only if classification == 'robust'
        - Only if decision == 'Reject H0'
        """
        if not self.manager.is_enabled:
            return

        try:
            # 1. Validate Eligibility (Redundant safety check, controller should also check)
            classification = evaluation_verdict.get("outcome_classification")
            decision = evaluation_verdict.get("hypothesis_decision")

            if classification != "robust" or decision != "Reject H0":
                logger.warning(f"[KnowledgeMemory] Attempted to store non-robust result: {classification}/{decision}. Skipped.")
                return

            # 2. Extract Data
            spec_body = experiment_spec.get("experiment_specification", {})
            analysis_plan = experiment_spec.get("statistical_analysis_plan", {})
            execution_mode = experiment_spec.get("execution_mode", "validation")

            # Construct Statistical Summary
            # We assume evaluation_verdict contains 'statistical_result' or similar breakdown
            # If not, we store the whole verdict part
            stat_summary = {
                "test_used": analysis_plan.get("primary_test"),
                "alpha": analysis_plan.get("alpha"),
                "p_value": evaluation_verdict.get("p_value", "N/A"), # Assuming evaluator outputs this top-level or in rationale
                "rationale": evaluation_verdict.get("rationale")
            }

            # Construct Supporting Context
            supporting_context = {
                "model_design": spec_body.get("model_design"),
                "data_modality": spec_body.get("data_modality"),
                "dataset_requirements": spec_body.get("dataset_requirements"),
                "execution_mode": execution_mode
            }

            data = {
                "run_id": str(run_id),
                "research_question": spec_body.get("research_question", "Unknown"),
                "final_hypothesis": spec_body.get("hypotheses", {}).get("H1", "Unknown"),
                "decision": decision,
                "effect_summary": evaluation_verdict.get("rationale", "No summary provided"), # Evaluator rationale as summary
                "statistical_summary": stat_summary,
                "assumptions_status": {"status": "Not explicitly tracked in schema yet, assumed PASS due to robust verdict"},
                "execution_mode": execution_mode,
                "confidence_level": "high", # By definition of 'robust'
                "supporting_context": supporting_context
            }

            self.manager.client.table("knowledge_memory").insert(data).execute()
            logger.info(f"[KnowledgeMemory] 🧠 Knowledge crystallized for run {run_id}")

        except Exception as e:
            logger.error(f"[KnowledgeMemory] Failed to record knowledge: {e}")
