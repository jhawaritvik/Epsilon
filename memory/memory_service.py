import logging
from typing import List, Dict, Any, Optional
from .types import FailureType
from .policies import CrystallizationPolicy
from .evidence_memory import EvidenceMemory
from .knowledge_memory import KnowledgeMemory
from .run_memory import RunMemory

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
        
    # --- Evidence Operations ---
    
    def get_evidence(self, goal: str, scope: str = "global", limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves evidence relevant to the goal.
        Scope: Currently basic text search support. Future: Handle 'local' vs 'global'.
        """
        return self._evidence.query_evidence(query=goal, limit=limit)

    def write_evidence(self,
                       run_id: str,
                       source_type: str,
                       source_url: str,
                       extracted_claim: str,
                       supporting_text: str,
                       confidence: str,
                       paper_title: Optional[str] = None,
                       section: Optional[str] = None) -> str:
        """
        Writes evidence with a READ-BEFORE-WRITE duplication check.
        Uses fuzzy matching to detect semantically similar claims.
        """
        # Guard: Check for duplicates
        existing = self._evidence.query_evidence(query=extracted_claim, limit=5)
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

    def get_knowledge(self, goal: str, scope: str = "global", limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves validated facts. 
        INVARIANT: Knowledge is APPEND-ONLY and high-precision.
        
        Scope Semantics:
        - scope="global": Retrieves from entire knowledge base (fuzzy search).
        - scope="goal": STRICT string match on the research goal (if supported).
        """
        return self._knowledge.query_knowledge(query=goal, limit=limit)

    def write_knowledge(self,
                        run_id: str,
                        experiment_spec: Dict[str, Any],
                        evaluation_verdict: Dict[str, Any],
                        iteration_count: int) -> str:
        """
        Writes to Knowledge Memory ONLY if the result satisfies the Crystallization Policy.
        """
        if self.crystallization_policy.is_eligible(evaluation_verdict, iteration_count):
            self._knowledge.record_knowledge(
                run_id=run_id,
                experiment_spec=experiment_spec,
                evaluation_verdict=evaluation_verdict
            )
            return "Knowledge crystallized."
        else:
            logger.info(f"[MemoryService] Result from run {run_id} did not meet crystallization criteria.")
            return "Result not crystallized (criteria not met)."

    # --- Run Memory Operations (Audit) ---

    def get_past_runs(self, goal: str = None, failure_type: Optional[FailureType] = None, run_id: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves past runs, optionally filtering by specific failure types (e.g. DESIGN failures).
        """
        return self._run_memory.query_run_memory(query=goal, issue_type=failure_type, run_id=run_id, limit=limit)

    def write_run(self,
                  run_id: str,
                  iteration: int,
                  research_goal: str,
                  experiment_spec: Dict[str, Any],
                  evaluation_verdict: Dict[str, Any],
                  feedback_passed: str = None):
        """
        Logs a run iteration. Append-only.
        """
        self._run_memory.record_iteration(
            run_id=run_id,
            iteration=iteration,
            research_goal=research_goal,
            experiment_spec=experiment_spec,
            evaluation_verdict=evaluation_verdict,
            feedback_passed=feedback_passed
        )
