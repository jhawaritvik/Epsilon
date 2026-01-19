"""
Tests for invariant validators.

These tests verify that our invariant validators correctly identify violations
and pass valid inputs.
"""

import pytest
import json
from pathlib import Path
from tests.invariants import (
    DesignAuthorityValidator,
    ExecutionAuthorityValidator,
    EvaluationAuthorityValidator,
    DataModalityValidator,
    ArtifactCompletenessValidator,
    MemoryCorrectnessValidator,
    InvariantChecker,
    ViolationType,
)


@pytest.mark.invariant
class TestDesignAuthorityValidator:
    """Tests for Design Authority Validator."""
    
    def test_valid_design_output(self):
        """Design output with only specifications should pass."""
        validator = DesignAuthorityValidator()
        
        design_output = """
        {
            "hypothesis": "L2 regularization reduces overfitting",
            "data_modality": {"type": "tabular"},
            "model_family": "neural_network"
        }
        """
        
        result = validator.validate("Design Agent", design_output)
        assert result is True
        assert len(validator.get_violations()) == 0
    
    def test_design_agent_writing_code_fails(self):
        """Design Agent producing executable code should fail."""
        validator = DesignAuthorityValidator()
        
        design_output = """
        def train_model():
            import torch
            model = torch.nn.Linear(10, 1)
            return model
        """
        
        result = validator.validate("Design Agent", design_output)
        assert result is False
        assert len(validator.get_violations()) == 1
        assert validator.get_violations()[0].violation_type == ViolationType.DESIGN_AUTHORITY
    
    def test_non_design_agent_can_write_code(self):
        """Non-design agents should be allowed to write code."""
        validator = DesignAuthorityValidator()
        
        code_output = """
        def train_model():
            import torch
            return torch.nn.Linear(10, 1)
        """
        
        result = validator.validate("Execution Agent", code_output)
        assert result is True
        assert len(validator.get_violations()) == 0


@pytest.mark.invariant
class TestExecutionAuthorityValidator:
    """Tests for Execution Authority Validator."""
    
    def test_execution_preserves_hypothesis(self):
        """Execution that preserves hypothesis should pass."""
        validator = ExecutionAuthorityValidator()
        
        original = {
            "hypothesis": "L2 reduces variance",
            "data_modality": {"type": "tabular"}
        }
        
        execution_output = {
            "execution_status": "success",
            "artifacts": ["run_experiment.py"]
        }
        
        result = validator.validate(original, execution_output)
        assert result is True
        assert len(validator.get_violations()) == 0
    
    def test_execution_modifying_hypothesis_fails(self):
        """Execution that modifies hypothesis should fail."""
        validator = ExecutionAuthorityValidator()
        
        original = {
            "hypothesis": "L2 reduces variance",
            "data_modality": {"type": "tabular"}
        }
        
        execution_output = {
            "hypothesis": "Dropout reduces variance",  # Changed!
            "execution_status": "success"
        }
        
        result = validator.validate(original, execution_output)
        assert result is False
        assert len(validator.get_violations()) == 1
        assert validator.get_violations()[0].violation_type == ViolationType.EXECUTION_AUTHORITY
    
    def test_execution_changing_modality_fails(self):
        """Execution that changes data modality should fail."""
        validator = ExecutionAuthorityValidator()
        
        original = {
            "hypothesis": "Test hypothesis",
            "data_modality": {"type": "tabular"}
        }
        
        execution_output = {
            "data_modality": {"type": "vision"},  # Changed!
            "execution_status": "success"
        }
        
        result = validator.validate(original, execution_output)
        assert result is False
        assert len(validator.get_violations()) == 1


@pytest.mark.invariant
class TestEvaluationAuthorityValidator:
    """Tests for Evaluation Authority Validator."""
    
    def test_evaluation_with_only_analysis_passes(self):
        """Evaluation doing only analysis should pass."""
        validator = EvaluationAuthorityValidator()
        
        output = """
        Statistical analysis:
        - T-statistic: 3.45
        - P-value: 0.002
        - Decision: Reject H0
        """
        
        tool_calls = ["run_statistical_test", "verify_assumptions"]
        
        result = validator.validate("Evaluation Agent", output, tool_calls)
        assert result is True
        assert len(validator.get_violations()) == 0
    
    def test_evaluation_executing_code_fails(self):
        """Evaluation Agent executing code should fail."""
        validator = EvaluationAuthorityValidator()
        
        output = "Running training loop..."
        tool_calls = ["execute_experiment"]  # Forbidden!
        
        result = validator.validate("Evaluation Agent", output, tool_calls)
        assert result is False
        assert len(validator.get_violations()) == 1
        assert validator.get_violations()[0].violation_type == ViolationType.EVALUATION_AUTHORITY
    
    def test_evaluation_with_execution_markers_fails(self):
        """Evaluation output with execution markers should fail."""
        validator = EvaluationAuthorityValidator()
        
        output = "I executed the run_experiment.py file and got results..."
        tool_calls = ["run_statistical_test"]
        
        result = validator.validate("Evaluation Agent", output, tool_calls)
        assert result is False
        assert len(validator.get_violations()) > 0


@pytest.mark.invariant
class TestDataModalityValidator:
    """Tests for Data Modality Validator."""
    
    def test_tabular_task_with_tabular_dataset_passes(self):
        """Tabular task using sklearn dataset should pass."""
        validator = DataModalityValidator()
        
        dataset_info = {
            "type": "tabular",
            "source": "sklearn",
            "name": "california_housing"
        }
        
        result = validator.validate("tabular", dataset_info)
        assert result is True
        assert len(validator.get_violations()) == 0
    
    def test_vision_task_with_vision_dataset_passes(self):
        """Vision task using image dataset should pass."""
        validator = DataModalityValidator()
        
        dataset_info = {
            "type": "image",
            "source": "torchvision",
            "name": "cifar10"
        }
        
        result = validator.validate("vision", dataset_info)
        assert result is True
        assert len(validator.get_violations()) == 0
    
    def test_modality_mismatch_fails(self):
        """Tabular task using vision dataset should fail."""
        validator = DataModalityValidator()
        
        dataset_info = {
            "type": "image",
            "source": "torchvision",
            "name": "cifar10"
        }
        
        result = validator.validate("tabular", dataset_info)
        assert result is False
        assert len(validator.get_violations()) == 1
        assert validator.get_violations()[0].violation_type == ViolationType.DATA_MODALITY


@pytest.mark.invariant
class TestArtifactCompletenessValidator:
    """Tests for Artifact Completeness Validator."""
    
    def test_all_artifacts_present_passes(self, artifact_files):
        """Complete set of artifacts should pass."""
        validator = ArtifactCompletenessValidator()
        
        result = validator.validate(artifact_files, run_completed=False)
        assert result is True
        assert len(validator.get_violations()) == 0
    
    def test_missing_artifact_fails(self, temp_experiment_dir):
        """Missing required artifact should fail."""
        validator = ArtifactCompletenessValidator()
        
        # Create only some artifacts
        (temp_experiment_dir / "run_experiment.py").write_text("# code")
        (temp_experiment_dir / "raw_results.json").write_text("{}")
        # Missing: dataset_used.json, execution.log, plots
        
        result = validator.validate(temp_experiment_dir, run_completed=False)
        assert result is False
        assert len(validator.get_violations()) == 1
        assert "dataset_used.json" in validator.get_violations()[0].description
    
    def test_no_plots_fails(self, temp_experiment_dir):
        """Missing plots should fail."""
        validator = ArtifactCompletenessValidator()
        
        # Create all required files except plots
        (temp_experiment_dir / "run_experiment.py").write_text("# code")
        (temp_experiment_dir / "raw_results.json").write_text("{}")
        (temp_experiment_dir / "dataset_used.json").write_text("{}")
        (temp_experiment_dir / "execution.log").write_text("log")
        
        result = validator.validate(temp_experiment_dir, run_completed=False)
        assert result is False
        assert "*.png" in validator.get_violations()[0].description
    
    def test_invalid_json_fails(self, temp_experiment_dir):
        """Invalid JSON in artifact should fail."""
        validator = ArtifactCompletenessValidator()
        
        # Create artifacts with invalid JSON
        (temp_experiment_dir / "run_experiment.py").write_text("# code")
        (temp_experiment_dir / "raw_results.json").write_text("invalid json {{{")
        (temp_experiment_dir / "dataset_used.json").write_text("{}")
        (temp_experiment_dir / "execution.log").write_text("log")
        
        # Create plot
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot([1, 2], [1, 2])
        plt.savefig(temp_experiment_dir / "plot.png")
        plt.close()
        
        result = validator.validate(temp_experiment_dir, run_completed=False)
        assert result is False
        assert "Invalid JSON" in validator.get_violations()[0].description


@pytest.mark.invariant
class TestMemoryCorrectnessValidator:
    """Tests for Memory Correctness Validator."""
    
    def test_no_duplicate_evidence_passes(self):
        """Unique evidence records should pass."""
        validator = MemoryCorrectnessValidator()
        
        records = [
            {"source_url": "paper1.pdf", "extracted_claim": "Claim A"},
            {"source_url": "paper2.pdf", "extracted_claim": "Claim B"},
            {"source_url": "paper3.pdf", "extracted_claim": "Claim C"},
        ]
        
        result = validator.validate_no_duplicates(records)
        assert result is True
        assert len(validator.get_violations()) == 0
    
    def test_duplicate_evidence_fails(self):
        """Duplicate evidence should fail."""
        validator = MemoryCorrectnessValidator()
        
        records = [
            {"source_url": "paper1.pdf", "extracted_claim": "Claim A"},
            {"source_url": "paper2.pdf", "extracted_claim": "Claim B"},
            {"source_url": "paper1.pdf", "extracted_claim": "Claim A"},  # Duplicate!
        ]
        
        result = validator.validate_no_duplicates(records)
        assert result is False
        assert len(validator.get_violations()) == 1
        assert validator.get_violations()[0].violation_type == ViolationType.MEMORY_CORRECTNESS
    
    def test_robust_verdict_passes(self):
        """Crystallizing knowledge from robust verdict should pass."""
        validator = MemoryCorrectnessValidator()
        
        knowledge = {
            "verdict": "robust",
            "claim": "L2 reduces variance"
        }
        
        result = validator.validate_knowledge_source(knowledge)
        assert result is True
        assert len(validator.get_violations()) == 0
    
    def test_non_robust_verdict_fails(self):
        """Crystallizing knowledge from non-robust verdict should fail."""
        validator = MemoryCorrectnessValidator()
        
        knowledge = {
            "verdict": "spurious",
            "claim": "Effect is spurious"
        }
        
        result = validator.validate_knowledge_source(knowledge)
        assert result is False
        assert len(validator.get_violations()) == 1


@pytest.mark.invariant
class TestInvariantChecker:
    """Tests for the main invariant checker."""
    
    def test_fail_fast_mode_stops_on_first_violation(self):
        """Fail-fast mode should stop on first violation."""
        checker = InvariantChecker(fail_fast=True)
        
        # Trigger violation in design authority
        design_validator = checker.get_validator("design_authority")
        design_validator.validate("Design Agent", "def code(): pass")
        
        # check_all should return False immediately
        result = checker.check_all()
        assert result is False
    
    def test_collect_mode_gathers_all_violations(self):
        """Collect mode should gather all violations."""
        checker = InvariantChecker(fail_fast=False)
        
        # Trigger multiple violations
        design_validator = checker.get_validator("design_authority")
        design_validator.validate("Design Agent", "def code(): pass")
        
        # Should collect all violations
        result = checker.check_all()
        assert result is False
        
        violations = checker.get_all_violations()
        assert len(violations) >= 1
    
    def test_reset_clears_violations(self):
        """Reset should clear all violations."""
        checker = InvariantChecker()
        
        # Trigger violation
        design_validator = checker.get_validator("design_authority")
        design_validator.validate("Design Agent", "import os")
        
        assert len(checker.get_all_violations()) > 0
        
        # Reset
        checker.reset_all()
        assert len(checker.get_all_violations()) == 0
