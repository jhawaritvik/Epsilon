import logging
import uuid
from typing import Dict, Any, List, Optional
from .types import FailureType
from .supabase_client import SupabaseManager

logger = logging.getLogger("RunMemory")

class RunMemory:
    """
    Backend for Run Memory.
    Manages audit logs of all iterations, including failures.
    """
    def __init__(self):
        self.manager = SupabaseManager()

    def record_iteration(self,
                         run_id: str,
                         iteration: int,
                         research_goal: str,
                         experiment_spec: Dict[str, Any],
                         evaluation_verdict: Dict[str, Any],
                         feedback_passed: str = None):
        """
        Records a single iteration to the run_memory table.
        """
        if not self.manager.is_enabled:
            return

        try:
            execution_mode = experiment_spec.get("execution_mode", "validation")
            
            # Validate execution_mode before insertion
            if execution_mode not in ["validation", "scientific"]:
                logger.warning(f"Invalid execution_mode: {execution_mode}. Defaulting to 'validation'.")
                execution_mode = "validation"
            
            # Map legacy strings to Enum if possible, else keep string
            classification = evaluation_verdict.get("outcome_classification", "failed")
            
            # Try to map issue_type to FailureType enum if possible
            issue_idx = evaluation_verdict.get("issue_type", "execution")
            if issue_idx == "none":
                 # If success, issue_type might be None or 'none'
                 issue_type = "none"
            else:
                 # Ensure it matches schema constraints ('design','data','execution','none')
                 issue_type = issue_idx

            data = {
                "run_id": str(run_id),
                "iteration": iteration,
                "research_goal": research_goal,
                "execution_mode": execution_mode,
                "experiment_spec": experiment_spec,
                "evaluation_verdict": evaluation_verdict,
                "classification": classification,
                "issue_type": issue_type,
                "feedback_passed": feedback_passed
            }

            self.manager.client.table("run_memory").insert(data).execute()
            logger.info(f"[RunMemory] Recorded iteration {iteration} for run {run_id}")

        except Exception as e:
            logger.error(f"[RunMemory] Failed to record iteration: {e}")

    def query_run_memory(self, query: str = None, issue_type: Optional[FailureType] = None, run_id: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves past runs. 
        useful for finding past failures to avoid repetition.
        """
        if not self.manager.is_enabled:
            return []
            
        try:
            builder = self.manager.client.table("run_memory").select("*")
            
            if query:
                builder = builder.ilike("research_goal", f"%{query}%")
                
            if issue_type:
                builder = builder.eq("issue_type", issue_type.value)

            if run_id:
                builder = builder.eq("run_id", str(run_id))
                
            response = builder.order("created_at", desc=True).limit(limit).execute()
            return response.data if response.data else []
            
        except Exception as e:
            logger.error(f"[RunMemory] Failed to query run memory: {e}")
            return []
