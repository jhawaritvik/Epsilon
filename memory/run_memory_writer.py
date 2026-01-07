import logging
import uuid
from typing import Dict, Any
from .supabase_client import SupabaseManager

logger = logging.getLogger("RunMemory")

class RunMemoryWriter:
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
        This is an append-only audit log.
        """
        if not self.manager.is_enabled:
            return

        try:
            # Extract key fields for column mapping
            execution_mode = experiment_spec.get("execution_mode", "validation")
            
            # Allow fallback if verdict is incomplete (failed run), default to 'failed'/'execution'
            classification = evaluation_verdict.get("outcome_classification", "failed")
            issue_type = evaluation_verdict.get("issue_type", "execution")
            
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
