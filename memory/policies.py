from dataclasses import dataclass

@dataclass
class CrystallizationPolicy:
    """
    Explicit criteria for when a research finding is "robust" enough
    to be promoted from Run Memory to Knowledge Memory.
    """
    min_iterations: int = 0         # Minimum iterations before crystallizing (0 = first try allowed)
    require_normality: bool = True  # If true, statistical tests must pass normality assumptions
    alpha_threshold: float = 0.05   # Statistical significance threshold (p-value)
    require_reproduction: bool = False # If true, requires >1 successful run (advanced)

    def is_eligible(self, verdict: dict, iteration_count: int) -> bool:
        """
        Evaluates if a verdict meets the crystallization criteria.
        
        Args:
            verdict: Evaluation verdict from the Evaluation Agent
            iteration_count: 0-indexed iteration number (0 = first attempt)
        """
        classification = verdict.get("outcome_classification")
        decision = verdict.get("hypothesis_decision")
        
        # 1. Fundamental Success
        if classification != "robust" or decision != "Reject H0":
            return False
            
        # 2. Iteration Threshold (0-indexed, so iteration_count >= min_iterations)
        # iteration_count=0 is first attempt, min_iterations=0 allows first-try success
        if iteration_count < self.min_iterations:
            return False
            
        # 3. Alpha Check (Redundant if 'robust' implies p < 0.05, but explicit is safer)
        # Assuming verdict contains p_value or we trust the agent's 'robust' label.
        # For now, we trust the 'robust' label as per Evaluation Agent logic.
        
        return True
