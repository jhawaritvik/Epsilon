import logging
from typing import List, Dict, Any, Optional
from .types import FailureType
from .policies import CrystallizationPolicy
from .evidence_memory import EvidenceMemory
from .knowledge_memory import KnowledgeMemory
from .run_memory import RunMemory
from core.identity import ExecutionIdentity

logger = logging.getLogger("MemoryService")

class MemoryService:
    """
    Facade for all memory operations.
    Enforces policies, scoping, and abstraction layers.
    Agents MUST use this service instead of calling writers directly.
    """
    def __init__(self, crystallization_policy: Optional[CrystallizationPolicy] = None):
        # Backends
        self._evidence = EvidenceMemory()
        self._knowledge = KnowledgeMemory()
        self._run_memory = RunMemory()
        
        # Policies
        self.crystallization_policy = crystallization_policy or CrystallizationPolicy()
        
    def _get_user(self, user_id: Optional[str] = None) -> str:
        """Helper to resolve user_id from Identity if not explicitly provided."""
        if user_id:
            return user_id
        return ExecutionIdentity.get_user_id()

    # --- Evidence Operations ---
    
    def get_evidence(self, goal: str, user_id: str = None, scope: str = "global", limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves evidence relevant to the goal.
        Scope: Currently basic text search support. Future: Handle 'local' vs 'global'.
        """
        uid = self._get_user(user_id)
        return self._evidence.query_evidence(user_id=uid, query=goal, limit=limit)

    def get_evidence_count(self, run_id: str) -> int:
        """
        Returns the number of evidence items saved for a specific run.
        Used for verification of persistence.
        """
        return self._evidence.count_evidence(run_id=run_id)

    def write_evidence(self,
                       run_id: str,
                       source_type: str,
                       source_url: str,
                       extracted_claim: str,
                       supporting_text: str,
                       confidence: str,
                       user_id: str = None,
                       paper_title: Optional[str] = None,
                       section: Optional[str] = None) -> str:
        """
        Writes evidence with a READ-BEFORE-WRITE duplication check.
        Uses fuzzy matching to detect semantically similar claims.
        """
        uid = self._get_user(user_id)
        
        # Guard: Check for duplicates (scoped to user)
        existing = self._evidence.query_evidence(user_id=uid, query=extracted_claim, limit=5)
        if existing:
            # Use fuzzy matching for semantic similarity
            from difflib import SequenceMatcher
            
            def is_similar(a: str, b: str, threshold: float = 0.85) -> bool:
                """Check if two strings are semantically similar."""
                return SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold
            
            for doc in existing:
                existing_claim = doc.get("extracted_claim", "")
                if existing_claim == extracted_claim:
                    logger.warning(f"[MemoryService] Exact duplicate detected. Skipping: '{extracted_claim[:50]}...'")
                    return "Duplicate evidence skipped."
                elif is_similar(existing_claim, extracted_claim):
                    logger.warning(f"[MemoryService] Similar evidence detected. Skipping: '{extracted_claim[:50]}...'")
                    return f"Similar evidence already exists: '{existing_claim[:50]}...'"

        self._evidence.record_evidence(
            run_id=run_id,
            user_id=uid,
            source_type=source_type,
            source_url=source_url,
            paper_title=paper_title,
            section=section,
            extracted_claim=extracted_claim,
            supporting_text=supporting_text,
            confidence=confidence
        )
        return "Evidence saved."

    # --- Knowledge Operations ---

    def get_knowledge(self, goal: str, user_id: str = None, scope: str = "global", limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves validated facts. 
        INVARIANT: Knowledge is APPEND-ONLY and high-precision.
        """
        uid = self._get_user(user_id)
        return self._knowledge.query_knowledge(user_id=uid, query=goal, limit=limit)

    def write_knowledge(self,
                        run_id: str,
                        experiment_spec: Dict[str, Any],
                        evaluation_verdict: Dict[str, Any],
                        iteration_count: int,
                        user_id: str = None) -> str:
        """
        Writes to Knowledge Memory ONLY if the result satisfies the Crystallization Policy.
        """
        uid = self._get_user(user_id)
        
        if self.crystallization_policy.is_eligible(evaluation_verdict, iteration_count):
            self._knowledge.record_knowledge(
                run_id=run_id,
                user_id=uid,
                experiment_spec=experiment_spec,
                evaluation_verdict=evaluation_verdict
            )
            return "Knowledge crystallized."
        else:
            logger.info(f"[MemoryService] Result from run {run_id} did not meet crystallization criteria.")
            return "Result not crystallized (criteria not met)."

    # --- Run Memory Operations (Audit) ---

    def get_past_runs(self, goal: str = None, failure_type: Optional[FailureType] = None, run_id: str = None, limit: int = 10, user_id: str = None) -> List[Dict[str, Any]]:
        """
        Retrieves past runs, optionally filtering by specific failure types (e.g. DESIGN failures).
        """
        uid = self._get_user(user_id)
        return self._run_memory.query_run_memory(user_id=uid, query=goal, issue_type=failure_type, run_id=run_id, limit=limit)

    def write_run(self,
                  run_id: str,
                  iteration: int,
                  research_goal: str,
                  experiment_spec: Dict[str, Any],
                  evaluation_verdict: Dict[str, Any],
                  user_id: str = None,
                  feedback_passed: str = None):
        """
        Logs a run iteration. Append-only.
        """
        uid = self._get_user(user_id)
        
        self._run_memory.record_iteration(
            run_id=run_id,
            user_id=uid,
            iteration=iteration,
            research_goal=research_goal,
            experiment_spec=experiment_spec,
            evaluation_verdict=evaluation_verdict,
            feedback_passed=feedback_passed
        )
