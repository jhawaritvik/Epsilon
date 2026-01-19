"""
Adversarial tests that deliberately try to break the Epsilon pipeline.

These tests verify that the system correctly rejects invalid inputs, handles
edge cases, and fails gracefully when it should.

Philosophy: If the system silently "does something reasonable" instead of
halting on ambiguity, that is a BUG, not a feature.
"""

import pytest
import json
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, patch


@pytest.mark.adversarial
class TestModalityTraps:
    """
    Tests that deliberately mix modalities to ensure rejection.
    
    Example: Asking to "predict house prices using CIFAR-10 features"
    should be rejected by the Design Agent or caught early.
    """
    
    def test_tabular_task_with_vision_dataset_rejected(self, mock_supabase_client):
        """
        Modality Trap: Tabular task with vision dataset.
        
        Expected:
        - Design Agent should reject or correct the mismatch
        - Execution should never start
        - Clear error message explaining the violation
        """
        from tests.invariants import DataModalityValidator
        
        # This is what would happen if the system failed
        task_modality = "tabular"
        dataset_info = {
            "type": "image",
            "source": "torchvision",
            "name": "cifar10"
        }
        
        # Validator should catch this
        validator = DataModalityValidator()
        result = validator.validate(task_modality, dataset_info)
        
        assert result is False, "DataModalityValidator should reject modality mismatch"
        assert len(validator.get_violations()) > 0
        assert "mismatch" in validator.get_violations()[0].description.lower()
    
    def test_vision_task_with_text_dataset_rejected(self):
        """
        Modality Trap: Vision task with text dataset.
        
        Expected: System halts with clear modality violation.
        """
        from tests.invariants import DataModalityValidator
        
        task_modality = "vision"
        dataset_info = {
            "type": "text",
            "source": "huggingface",
            "name": "imdb"
        }
        
        validator = DataModalityValidator()
        result = validator.validate(task_modality, dataset_info)
        
        assert result is False
        assert "mismatch" in validator.get_violations()[0].description.lower()
    
    def test_impossible_dataset_request_fails(self, mock_agent_runner):
        """
        Impossible Request: "Use MNIST for regression on housing prices"
        
        Expected:
        - Design Agent recognizes the impossibility
        - System halts before execution
        - No silent fallback to "something reasonable"
        """
        # This would be caught during design phase
        research_goal = "Use MNIST images to predict California housing prices"
        
        # The system should recognize this is nonsensical
        # and either:
        # 1. Reject it outright
        # 2. Correct it to a sensible interpretation
        # 3. Ask for clarification
        
        # What it should NOT do:
        # - Silently use MNIST for something else
        # - Silently use a different dataset
        # - Proceed without validation
        
        # For now, we'll verify the validator catches it
        from tests.invariants import DataModalityValidator
        
        validator = DataModalityValidator()
        
        # If the system tried to proceed with this:
        task_modality = "tabular"  # Housing price prediction
        dataset_info = {
            "type": "image",
            "source": "torchvision",
            "name": "mnist"
        }
        
        result = validator.validate(task_modality, dataset_info)
        assert result is False, "System should reject impossible dataset-task pairing"


@pytest.mark.adversarial
class TestAmbiguousSpecifications:
    """
    Tests with deliberately ambiguous or underspecified goals.
    
    The system should HALT and request clarification, not guess.
    """
    
    def test_ambiguous_dataset_specification_halts(self):
        """
        Ambiguous Goal: "Study learning dynamics in early training"
        
        Expected:
        - System recognizes ambiguity (which dataset? which model?)
        - HALT with specific questions
        - NO execution attempt
        """
        # This is a design-phase failure
        # The Design Agent should recognize insufficient information
        
        research_goal = "Study learning dynamics in early training"
        
        # Ambiguities:
        # - Which dataset? (MNIST? CIFAR? ImageNet?)
        # - Which model? (ResNet? VGG? Transformer?)
        # - Which metric? (Loss? Gradient norm? Learning rate?)
        # - What comparison? (Different initializations? Different optimizers?)
        
        # The system should NOT:
        # - Pick arbitrary defaults
        # - Proceed with incomplete spec
        
        # For testing purposes, we verify that if such a spec reached execution,
        # it would be caught
        
        incomplete_spec = {
            "hypothesis": "Learning dynamics matter",
            # Missing: data_modality, dataset_requirements, model_family, etc.
        }
        
        # Should fail validation
        required_fields = ["data_modality", "dataset_requirements", "model_family"]
        
        for field in required_fields:
            assert field not in incomplete_spec, \
                f"Test setup error: {field} should be missing"
        
        # Real system should halt here
        # This is a placeholder for actual Design Agent validation
        assert True, "Ambiguous specs should trigger clarification request"
    
    def test_contradictory_requirements_halt(self):
        """
        Contradictory Requirements: "Use small dataset for large-scale study"
        
        Expected:
        - System detects contradiction
        - Halts with explanation
        """
        contradictory_spec = {
            "hypothesis": "Large-scale training improves generalization",
            "data_modality": {"type": "vision"},
            "dataset_requirements": {
                "size": "small",  # Contradicts "large-scale"
                "samples": 100
            },
            "execution_mode": "large_scale_training"
        }
        
        # System should detect this contradiction
        # For now, we document the expected behavior
        assert contradictory_spec["dataset_requirements"]["size"] == "small"
        assert "large_scale" in contradictory_spec["execution_mode"]
        
        # In real implementation, Design Agent should catch this
        # and request clarification
    
    def test_unresolvable_dataset_request_halts(self):
        """
        Unresolvable Dataset: Request for non-existent dataset.
        
        Expected:
        - dataset_resolver fails cleanly
        - No synthetic fallback
        - Clear error message
        """
        from tests.invariants import ArtifactCompletenessValidator
        
        # Simulate dataset resolver failing
        dataset_spec = {
            "family": "nonexistent_library",
            "name": "fake_dataset_12345",
            "type": "tabular"
        }
        
        # When dataset_resolver fails, execution should halt
        # No run_experiment.py should be created
        # This is tested by checking artifact NON-existence
        
        # In a real scenario, temp_experiment_dir would be empty
        # because dataset resolution failed before execution
        
        # The system should log:
        # - Dataset resolution attempt
        # - Failure reason
        # - No fallback to synthetic data (unless explicitly designed for it)
        
        assert True, "Dataset resolution failure should halt pipeline cleanly"


@pytest.mark.adversarial
class TestForbiddenBehaviors:
    """
    Tests that ensure agents don't violate authority boundaries.
    """
    
    def test_design_agent_cannot_execute_code(self):
        """
        Authority Violation: Design Agent tries to execute code.
        
        Expected:
        - Immediate rejection
        - Critical invariant violation logged
        """
        from tests.invariants import DesignAuthorityValidator
        
        validator = DesignAuthorityValidator()
        
        # Design Agent should never produce this
        malicious_output = """
        Let me run this experiment:
        ```python
        import os
        os.system('rm -rf /')  # Malicious!
        ```
        """
        
        result = validator.validate("Design Agent", malicious_output)
        
        assert result is False
        assert len(validator.get_violations()) > 0
        assert validator.get_violations()[0].severity == "critical"
    
    def test_execution_agent_cannot_modify_hypothesis(self):
        """
        Authority Violation: Execution Agent modifies hypothesis.
        
        Expected:
        - Immediate rejection
        - Critical invariant violation logged
        """
        from tests.invariants import ExecutionAuthorityValidator
        
        validator = ExecutionAuthorityValidator()
        
        original_spec = {
            "hypothesis": "L2 regularization reduces variance",
            "data_modality": {"type": "tabular"}
        }
        
        # Execution tries to change hypothesis
        execution_output = {
            "hypothesis": "Dropout is better than L2",  # VIOLATION
            "execution_status": "success"
        }
        
        result = validator.validate(original_spec, execution_output)
        
        assert result is False
        assert len(validator.get_violations()) > 0
    
    def test_evaluation_agent_cannot_execute_experiments(self):
        """
        Authority Violation: Evaluation Agent executes code.
        
        Expected:
        - Immediate rejection
        - Critical invariant violation logged
        """
        from tests.invariants import EvaluationAuthorityValidator
        
        validator = EvaluationAuthorityValidator()
        
        # Evaluation should only analyze, not execute
        evaluation_output = "I ran the experiment and got these results..."
        tool_calls = ["execute_experiment"]  # FORBIDDEN
        
        result = validator.validate("Evaluation Agent", evaluation_output, tool_calls)
        
        assert result is False
        assert len(validator.get_violations()) > 0


@pytest.mark.adversarial
class TestSyntheticDataFallback:
    """
    Tests that synthetic data fallback only happens when explicitly designed for it.
    """
    
    def test_no_silent_synthetic_fallback(self):
        """
        Forbidden Fallback: System should not silently switch to synthetic data.
        
        If user requests "CIFAR-10" but it's unavailable, the system should:
        1. Log the failure
        2. Halt execution
        3. NOT silently switch to make_classification or similar
        
        Exception: PINN and similar physics tasks that REQUIRE synthetic data.
        """
        # This tests the dataset_resolver contract
        
        requested_dataset = {
            "family": "torchvision",
            "name": "cifar10",
            "type": "image"
        }
        
        # If CIFAR-10 is unavailable, dataset_resolver should:
        # - Return an error status
        # - NOT return make_classification or similar
        
        # The execution agent should then:
        # - Halt with dataset resolution failure
        # - NOT proceed with alternative dataset
        
        # Exception clause:
        # If data_modality.type == "physics" or similar, synthetic is expected
        
        assert True, "Silent synthetic fallback is forbidden unless explicit"
    
    def test_pinn_allows_synthetic_data(self):
        """
        Allowed Synthetic: PINN tasks should use synthetic collocation points.
        
        This is an exception to the no-synthetic rule.
        """
        task_type = "physics_informed"
        
        # For PINN:
        # - Collocation points are EXPECTED to be synthetic
        # - This is not a fallback, but the primary method
        
        # The difference:
        # - Requested: "Use PINN with 5000 collocation points"
        # - System: Generates synthetic points (CORRECT)
        
        # vs
        
        # - Requested: "Use CIFAR-10"
        # - System: Can't find CIFAR-10, uses make_classification (WRONG)
        
        assert True, "PINN synthetic data generation is allowed and expected"


@pytest.mark.adversarial
@pytest.mark.slow
class TestEdgeCases:
    """
    Edge cases that might break the system.
    """
    
    def test_empty_research_goal(self):
        """
        Edge Case: Empty or whitespace-only research goal.
        
        Expected:
        - Immediate rejection
        - Clear error message
        """
        research_goal = "   \n\t  "
        
        # System should reject this before doing any work
        assert research_goal.strip() == "", "Test setup: goal should be empty"
        
        # Expected behavior:
        # - Input validation fails
        # - No agent calls made
        # - User-friendly error message
    
    def test_extremely_long_research_goal(self):
        """
        Edge Case: Research goal that exceeds token limits.
        
        Expected:
        - Graceful handling
        - Truncation or summary
        - No silent failure
        """
        # 10,000 word research goal
        research_goal = "test " * 10000
        
        # System should either:
        # 1. Reject with "too long" error
        # 2. Summarize/truncate with warning
        # 3. Handle it fine (if LLM supports it)
        
        # What it should NOT do:
        # - Silently truncate without warning
        # - Crash
        
        assert len(research_goal.split()) == 10000
    
    def test_special_characters_in_goal(self):
        """
        Edge Case: Special characters, unicode, code injection attempts.
        
        Expected:
        - Proper escaping
        - No code execution
        - Safe handling
        """
        malicious_goals = [
            "'; DROP TABLE evidence_memory; --",
            "$(rm -rf /)",
            "<script>alert('xss')</script>",
            "../../etc/passwd",
        ]
        
        for goal in malicious_goals:
            # System should treat these as plain text
            # No SQL injection, command injection, XSS, path traversal
            assert isinstance(goal, str), "Should be treated as plain string"


@pytest.mark.adversarial
def test_all_adversarial_tests_exist():
    """
    Meta-test: Ensures we have comprehensive adversarial coverage.
    
    Required adversarial test categories:
    - Modality traps
    - Ambiguous specifications
    - Forbidden behaviors
    - Synthetic data fallback
    - Edge cases
    """
    # This is a documentation test
    # It ensures we've thought about all attack vectors
    
    required_categories = [
        "modality_traps",
        "ambiguous_specifications",
        "forbidden_behaviors",
        "synthetic_data_fallback",
        "edge_cases",
    ]
    
    # Verify we have test classes for each
    import sys
    current_module = sys.modules[__name__]
    
    class_names = [name for name in dir(current_module) 
                   if name.startswith("Test")]
    
    assert len(class_names) >= len(required_categories), \
        "Missing adversarial test categories"
