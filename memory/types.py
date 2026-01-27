from enum import Enum

class FailureType(str, Enum):
    """
    Standardized taxonomy for research failures.
    Used for analytics, adaptive control flow, and automated retries.
    
    Categories:
    - Operational: Issues with the pipeline mechanics
    - Epistemic: Issues with the research question itself
    """
    # Operational Failures (existing)
    DESIGN = "design"           # Flawed hypothesis, invalid variables, logical contradictions
    DATA = "data"               # Dataset missing, quality issues, empty retrieval
    EXECUTION = "execution"     # Code crashes, library errors, timeout constraints
    STATISTICAL = "statistical" # Evaluation anomalies (deprecated, use specific types below)
    
    # Epistemic Failures (new - first-class research failure modes)
    EPISTEMIC = "epistemic"                     # Question ill-posed or fundamentally unanswerable
    DATA_INSUFFICIENCY = "data_insufficiency"   # Insufficient samples for statistical power
    ASSUMPTION_VIOLATION = "assumption_violation"  # Normality, independence, homoscedasticity failed
    NON_CONVERGENCE = "non_convergence"         # Model/optimization did not converge
    RESOURCE_EXHAUSTION = "resource_exhaustion" # Token/time/compute budget exceeded

class ExecutionStatus(str, Enum):
    """
    Explicit status of the Code Execution Agent.
    Controller uses this to 'fail fast' if execution crashed,
    preventing Evaluation Agent from hallucinating on stale artifacts.
    """
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
