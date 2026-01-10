from enum import Enum

class FailureType(str, Enum):
    """
    Standardized taxonomy for research failures.
    Used for analytics, adaptive control flow, and automated retries.
    """
    DESIGN = "design"           # Flawed hypothesis, invalid variables, logical contradictions
    DATA = "data"               # Dataset missing, quality issues, empty retrieval
    EXECUTION = "execution"     # Code crashes, library errors, timeout constraints
    STATISTICAL = "statistical" # Evaluation anomalies, assumption violations (non-robust results)

class ExecutionStatus(str, Enum):
    """
    Explicit status of the Code Execution Agent.
    Controller uses this to 'fail fast' if execution crashed,
    preventing Evaluation Agent from hallucinating on stale artifacts.
    """
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
