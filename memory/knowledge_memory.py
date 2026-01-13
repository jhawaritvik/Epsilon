import logging
import uuid
from typing import Dict, Any, List
from .supabase_client import SupabaseManager

logger = logging.getLogger("KnowledgeMemory")

class KnowledgeMemory:
    """
    Backend for Knowledge Memory.
    Manages validated scientific facts crystallized from successful runs.
    """
    def __init__(self):
        self.manager = SupabaseManager()

    def record_knowledge(self,
                         run_id: str,
                         user_id: str,
                         experiment_spec: Dict[str, Any],
                         evaluation_verdict: Dict[str, Any]):
        """
        Records a validated scientific conclusion to knowledge_memory.
        """
        if not self.manager.is_enabled:
            return

        try:
            # 1. Extract Data
            spec_body = experiment_spec.get("experiment_specification", {})
            analysis_plan = experiment_spec.get("statistical_analysis_plan", {})
            execution_mode = experiment_spec.get("execution_mode", "validation")
            
            # Allow flexible verdict structure
            outcome_classification = evaluation_verdict.get("outcome_classification")
            decision = evaluation_verdict.get("hypothesis_decision")
            
            stat_summary = {
                "test_used": analysis_plan.get("primary_test"),
                "alpha": analysis_plan.get("alpha"),
                "p_value": evaluation_verdict.get("p_value", "N/A"),
                "rationale": evaluation_verdict.get("rationale")
            }
            
            # Extract actual assumption check results (not hardcoded)
            assumption_checks = evaluation_verdict.get("assumption_checks", [])
            assumptions_status = {
                "checks": assumption_checks,
                "all_passed": all(
                    check.get("status") == "PASS" 
                    for check in assumption_checks
                ) if assumption_checks else True  # Default True if no checks recorded
            }

            supporting_context = {
                "model_design": spec_body.get("model_design"),
                "data_modality": spec_body.get("data_modality"),
                "dataset_requirements": spec_body.get("dataset_requirements"),
                "execution_mode": execution_mode
            }

            data = {
                "run_id": str(run_id),
                "user_id": str(user_id),
                "research_question": spec_body.get("research_question", "Unknown"),
                "final_hypothesis": spec_body.get("hypotheses", {}).get("H1", "Unknown"),
                "decision": decision,
                "effect_summary": evaluation_verdict.get("rationale", "No summary provided"),
                "statistical_summary": stat_summary,
                "assumptions_status": assumptions_status,
                "execution_mode": execution_mode,
                "confidence_level": "high",
                "supporting_context": supporting_context
            }

            self.manager.client.table("knowledge_memory").insert(data).execute()
            logger.info(f"[KnowledgeMemory] 🧠 Knowledge crystallized for run {run_id}")

        except Exception as e:
            logger.error(f"[KnowledgeMemory] Failed to record knowledge: {e}")

    def query_knowledge(self, user_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves validated facts relevant to the query.
        """
        if not self.manager.is_enabled:
            return []
            
        try:
            response = self.manager.client.table("knowledge_memory")\
                .select("*")\
                .eq("user_id", str(user_id))\
                .ilike("research_question", f"%{query}%")\
                .limit(limit)\
                .execute()
                
            return response.data if response.data else []
            
        except Exception as e:
            logger.error(f"[KnowledgeMemory] Failed to query knowledge: {e}")
            return []
