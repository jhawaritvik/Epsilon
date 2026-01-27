"""
Core invariant validators for the Epsilon research pipeline.

These validators enforce the fundamental "laws" of the system that must never be violated:
1. Design authority separation (agents respect boundaries)
2. Data modality consistency (no silent switching)
3. Artifact completeness (all required outputs)
4. Memory correctness (proper insertion rules)

Invariant violations should trigger immediate pipeline failure.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ViolationType(Enum):
    """Types of invariant violations."""
    DESIGN_AUTHORITY = "design_authority"
    EXECUTION_AUTHORITY = "execution_authority"
    EVALUATION_AUTHORITY = "evaluation_authority"
    DATA_MODALITY = "data_modality"
    ARTIFACT_COMPLETENESS = "artifact_completeness"
    MEMORY_CORRECTNESS = "memory_correctness"


@dataclass
class InvariantViolation:
    """Represents a single invariant violation."""
    violation_type: ViolationType
    agent_name: str
    description: str
    severity: str  # "critical", "error", "warning"
    context: Optional[Dict[str, Any]] = None


class InvariantValidator:
    """Base class for all invariant validators."""
    
    def __init__(self):
        self.violations: List[InvariantViolation] = []
    
    def validate(self, *args, **kwargs) -> bool:
        """
        Validates the invariant. Returns True if valid, False if violated.
        Subclasses must implement this.
        """
        raise NotImplementedError
    
    def get_violations(self) -> List[InvariantViolation]:
        """Returns all recorded violations."""
        return self.violations
    
    def reset(self):
        """Clears all violations."""
        self.violations.clear()


class DesignAuthorityValidator(InvariantValidator):
    """
    Ensures the Design Agent never writes executable code.
    
    The Design Agent's role is to create specifications, not implementations.
    This validator checks that Design Agent outputs contain no executable code.
    """
    
    FORBIDDEN_PATTERNS = [
        "def ",
        "class ",
        "import ",
        "from ",
        "if __name__",
        "exec(",
        "eval(",
    ]
    
    def validate(self, agent_name: str, agent_output: str) -> bool:
        """
        Validates that Design Agent output contains no executable code.
        
        Args:
            agent_name: Name of the agent
            agent_output: The agent's output
            
        Returns:
            True if valid, False if violated
        """
        if "design" not in agent_name.lower():
            return True  # Not a design agent, skip
        
        # Check for executable code patterns
        for pattern in self.FORBIDDEN_PATTERNS:
            if pattern in agent_output:
                violation = InvariantViolation(
                    violation_type=ViolationType.DESIGN_AUTHORITY,
                    agent_name=agent_name,
                    description=f"Design Agent produced executable code (pattern: {pattern})",
                    severity="critical",
                    context={"pattern": pattern, "output_snippet": agent_output[:200]}
                )
                self.violations.append(violation)
                logger.error(f"INVARIANT VIOLATION: {violation.description}")
                return False
        
        return True


class ExecutionAuthorityValidator(InvariantValidator):
    """
    Ensures the Execution Agent never modifies hypotheses or experimental designs.
    
    The Execution Agent executes experiments but must not change the research question
    or hypothesis. This validator checks that execution outputs preserve the original
    hypothesis and experimental design.
    """
    
    def validate(self, original_spec: Dict[str, Any], execution_output: Dict[str, Any]) -> bool:
        """
        Validates that execution didn't modify the hypothesis or core design.
        
        Args:
            original_spec: The original experiment specification
            execution_output: The execution agent's output
            
        Returns:
            True if valid, False if violated
        """
        # Check if hypothesis was changed
        original_hypothesis = original_spec.get("hypothesis", "")
        
        # If execution output contains a different hypothesis, that's a violation
        if "hypothesis" in execution_output:
            if execution_output["hypothesis"] != original_hypothesis:
                violation = InvariantViolation(
                    violation_type=ViolationType.EXECUTION_AUTHORITY,
                    agent_name="Execution Agent",
                    description="Execution Agent modified the hypothesis",
                    severity="critical",
                    context={
                        "original": original_hypothesis,
                        "modified": execution_output["hypothesis"]
                    }
                )
                self.violations.append(violation)
                logger.error(f"INVARIANT VIOLATION: {violation.description}")
                return False
        
        # Check if data modality was changed
        if "data_modality" in original_spec and "data_modality" in execution_output:
            if original_spec["data_modality"] != execution_output["data_modality"]:
                violation = InvariantViolation(
                    violation_type=ViolationType.EXECUTION_AUTHORITY,
                    agent_name="Execution Agent",
                    description="Execution Agent modified data modality",
                    severity="critical",
                    context={
                        "original": original_spec["data_modality"],
                        "modified": execution_output["data_modality"]
                    }
                )
                self.violations.append(violation)
                logger.error(f"INVARIANT VIOLATION: {violation.description}")
                return False
        
        return True


class EvaluationAuthorityValidator(InvariantValidator):
    """
    Ensures the Evaluation Agent never executes training code.
    
    The Evaluation Agent's role is analysis, not execution. This validator
    checks that evaluation outputs contain no evidence of code execution.
    """
    
    FORBIDDEN_EXECUTION_MARKERS = [
        "execute_experiment",
        "run_experiment.py",
        "training_loop",
        "model.fit",
        "torch.nn",
        "tensorflow",
    ]
    
    def validate(self, agent_name: str, agent_output: str, tool_calls: List[str]) -> bool:
        """
        Validates that Evaluation Agent did not execute training code.
        
        Args:
            agent_name: Name of the agent
            agent_output: The agent's output
            tool_calls: List of tools called by the agent
            
        Returns:
            True if valid, False if violated
        """
        if "evaluation" not in agent_name.lower():
            return True  # Not an evaluation agent, skip
        
        # Allowed tools for evaluation agent (these contain "run" but are not execution)
        ALLOWED_TOOLS = [
            "run_statistical_test",
            "run_analysis", 
            "verify_assumptions",
            "python_analysis_tool"
        ]
        
        # Check tool calls for forbidden execution tools
        for tool in tool_calls:
            # Skip allowed analysis tools
            if tool in ALLOWED_TOOLS:
                continue
            # Check for actual execution tools (but not analysis tools)
            if tool.lower() in ["execute_experiment", "execute_code", "run_experiment"]:
                violation = InvariantViolation(
                    violation_type=ViolationType.EVALUATION_AUTHORITY,
                    agent_name=agent_name,
                    description=f"Evaluation Agent called execution tool: {tool}",
                    severity="critical",
                    context={"tool": tool}
                )
                self.violations.append(violation)
                logger.error(f"INVARIANT VIOLATION: {violation.description}")
                return False
        
        # Check output for execution markers
        for marker in self.FORBIDDEN_EXECUTION_MARKERS:
            if marker in agent_output:
                violation = InvariantViolation(
                    violation_type=ViolationType.EVALUATION_AUTHORITY,
                    agent_name=agent_name,
                    description=f"Evaluation Agent output contains execution marker: {marker}",
                    severity="error",
                    context={"marker": marker}
                )
                self.violations.append(violation)
                logger.warning(f"INVARIANT VIOLATION: {violation.description}")
                return False
        
        return True


class DataModalityValidator(InvariantValidator):
    """
    Ensures data modality consistency throughout the pipeline.
    
    Tabular tasks must use tabular datasets, vision tasks must use image datasets.
    No silent modality switching is allowed.
    """
    
    MODALITY_MAPPINGS = {
        "tabular": ["csv", "sklearn", "pandas", "dataframe"],
        "vision": ["image", "cifar", "mnist", "imagenet", "torchvision"],
        "text": ["text", "nlp", "huggingface", "tokenizer"],
        "time_series": ["timeseries", "temporal", "sequential"],
    }
    
    def validate(self, 
                 task_modality: str, 
                 dataset_info: Dict[str, Any]) -> bool:
        """
        Validates that dataset matches task modality.
        
        Args:
            task_modality: Expected data modality (e.g., "tabular", "vision")
            dataset_info: Information about the dataset used
            
        Returns:
            True if valid, False if violated
        """
        dataset_type = dataset_info.get("type", "").lower()
        dataset_source = dataset_info.get("source", "").lower()
        dataset_name = dataset_info.get("name", "").lower()
        
        # Get expected keywords for this modality
        expected_keywords = self.MODALITY_MAPPINGS.get(task_modality.lower(), [])
        
        # Check if dataset matches expected modality
        dataset_str = f"{dataset_type} {dataset_source} {dataset_name}"
        
        # Any keyword match is good
        if any(keyword in dataset_str for keyword in expected_keywords):
            return True
        
        # Check for forbidden modality switches
        for modality, keywords in self.MODALITY_MAPPINGS.items():
            if modality != task_modality.lower():
                if any(keyword in dataset_str for keyword in keywords):
                    violation = InvariantViolation(
                        violation_type=ViolationType.DATA_MODALITY,
                        agent_name="Pipeline",
                        description=f"Modality mismatch: {task_modality} task using {modality} dataset",
                        severity="critical",
                        context={
                            "expected_modality": task_modality,
                            "detected_modality": modality,
                            "dataset_info": dataset_info
                        }
                    )
                    self.violations.append(violation)
                    logger.error(f"INVARIANT VIOLATION: {violation.description}")
                    return False
        
        # If we can't determine, log a warning but don't fail
        logger.warning(f"Could not verify modality for dataset: {dataset_info}")
        return True


class ArtifactCompletenessValidator(InvariantValidator):
    """
    Ensures all required artifacts are produced by each run.
    
    Required artifacts:
    - run_experiment.py
    - raw_results.json
    - dataset_used.json
    - execution.log
    - At least one .png plot
    - FINAL_REPORT.md (if run completed)
    """
    
    REQUIRED_ARTIFACTS = [
        "run_experiment.py",
        "raw_results.json",
        "dataset_used.json",
        "execution.log",
    ]
    
    def validate(self, experiment_dir: Path, run_completed: bool = False) -> bool:
        """
        Validates that all required artifacts exist.
        
        Args:
            experiment_dir: Directory containing experiment artifacts
            run_completed: Whether the full run completed (requires FINAL_REPORT.md)
            
        Returns:
            True if valid, False if violated
        """
        missing_artifacts = []
        
        # Check required files
        for artifact in self.REQUIRED_ARTIFACTS:
            artifact_path = experiment_dir / artifact
            if not artifact_path.exists():
                missing_artifacts.append(artifact)
        
        # Check for at least one plot
        plots = list(experiment_dir.glob("*.png"))
        if len(plots) == 0:
            missing_artifacts.append("*.png (no plots found)")
        
        # Check for final report if run completed
        if run_completed:
            report_path = experiment_dir / "FINAL_REPORT.md"
            if not report_path.exists():
                missing_artifacts.append("FINAL_REPORT.md")
        
        if missing_artifacts:
            violation = InvariantViolation(
                violation_type=ViolationType.ARTIFACT_COMPLETENESS,
                agent_name="Pipeline",
                description=f"Missing required artifacts: {', '.join(missing_artifacts)}",
                severity="critical",
                context={
                    "experiment_dir": str(experiment_dir),
                    "missing": missing_artifacts
                }
            )
            self.violations.append(violation)
            logger.error(f"INVARIANT VIOLATION: {violation.description}")
            return False
        
        # Validate JSON files are valid
        json_files = ["raw_results.json", "dataset_used.json"]
        for json_file in json_files:
            json_path = experiment_dir / json_file
            try:
                with open(json_path, 'r') as f:
                    json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                violation = InvariantViolation(
                    violation_type=ViolationType.ARTIFACT_COMPLETENESS,
                    agent_name="Pipeline",
                    description=f"Invalid JSON in {json_file}: {str(e)}",
                    severity="error",
                    context={"file": json_file, "error": str(e)}
                )
                self.violations.append(violation)
                logger.error(f"INVARIANT VIOLATION: {violation.description}")
                return False
        
        return True


class MemoryCorrectnessValidator(InvariantValidator):
    """
    Ensures memory system correctness.
    
    Rules:
    - Failed runs must be queryable
    - Crystallized knowledge only from robust verdicts
    - No duplicate evidence insertion
    """
    
    def validate_no_duplicates(self, evidence_records: List[Dict[str, Any]]) -> bool:
        """
        Validates that evidence records contain no duplicates.
        
        Args:
            evidence_records: List of evidence records
            
        Returns:
            True if valid, False if violated
        """
        # Create signature for each record (source_url + extracted_claim)
        signatures = []
        for record in evidence_records:
            sig = f"{record.get('source_url', '')}::{record.get('extracted_claim', '')}"
            if sig in signatures:
                violation = InvariantViolation(
                    violation_type=ViolationType.MEMORY_CORRECTNESS,
                    agent_name="Memory System",
                    description="Duplicate evidence insertion detected",
                    severity="error",
                    context={"duplicate_signature": sig}
                )
                self.violations.append(violation)
                logger.error(f"INVARIANT VIOLATION: {violation.description}")
                return False
            signatures.append(sig)
        
        return True
    
    def validate_knowledge_source(self, knowledge_record: Dict[str, Any]) -> bool:
        """
        Validates that crystallized knowledge only comes from robust verdicts.
        
        Args:
            knowledge_record: The knowledge record to validate
            
        Returns:
            True if valid, False if violated
        """
        verdict = knowledge_record.get("verdict", "").lower()
        
        if verdict not in ["robust", "very_robust"]:
            violation = InvariantViolation(
                violation_type=ViolationType.MEMORY_CORRECTNESS,
                agent_name="Memory System",
                description=f"Attempted to crystallize knowledge from non-robust verdict: {verdict}",
                severity="critical",
                context={"verdict": verdict, "knowledge": knowledge_record}
            )
            self.violations.append(violation)
            logger.error(f"INVARIANT VIOLATION: {violation.description}")
            return False
        
        return True


class InvariantChecker:
    """
    Main invariant checker that coordinates all validators.
    """
    
    def __init__(self, fail_fast: bool = True):
        """
        Args:
            fail_fast: If True, stop on first violation. If False, collect all violations.
        """
        self.fail_fast = fail_fast
        self.validators = {
            "design_authority": DesignAuthorityValidator(),
            "execution_authority": ExecutionAuthorityValidator(),
            "evaluation_authority": EvaluationAuthorityValidator(),
            "data_modality": DataModalityValidator(),
            "artifact_completeness": ArtifactCompletenessValidator(),
            "memory_correctness": MemoryCorrectnessValidator(),
        }
    
    def check_all(self) -> bool:
        """
        Checks all validators and returns True if all pass.
        """
        all_valid = True
        for name, validator in self.validators.items():
            if validator.get_violations():
                all_valid = False
                if self.fail_fast:
                    return False
        return all_valid
    
    def get_all_violations(self) -> List[InvariantViolation]:
        """Returns all violations from all validators."""
        violations = []
        for validator in self.validators.values():
            violations.extend(validator.get_violations())
        return violations
    
    def reset_all(self):
        """Resets all validators."""
        for validator in self.validators.values():
            validator.reset()
    
    def get_validator(self, name: str) -> Optional[InvariantValidator]:
        """Gets a specific validator by name."""
        return self.validators.get(name)
