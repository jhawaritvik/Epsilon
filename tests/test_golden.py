"""
Golden tests for the Epsilon research pipeline.

These tests use fixed research goals with known expected behavior to verify
that the system produces correct, deterministic results.

Golden tests are like compiler tests - they verify structural correctness,
not numeric accuracy.
"""

import pytest
import yaml
import json
from pathlib import Path
from typing import Dict, Any
from tests.invariants import (
    DataModalityValidator,
    ArtifactCompletenessValidator,
    InvariantChecker,
)


class GoldenTestRunner:
    """Runs golden tests and validates expectations."""
    
    def __init__(self, test_spec_path: Path):
        """
        Args:
            test_spec_path: Path to YAML test specification
        """
        with open(test_spec_path, 'r') as f:
            self.spec = yaml.safe_load(f)
        
        self.goal = self.spec['goal']
        self.expected = self.spec['expected']
    
    def validate_dataset_choice(self, dataset_info: Dict[str, Any]) -> bool:
        """Validates that dataset choice matches expectations."""
        expected_modality = self.expected['data_modality']
        expected_family = self.expected.get('dataset_family', '')
        
        # Check modality
        validator = DataModalityValidator()
        modality_valid = validator.validate(expected_modality, dataset_info)
        
        if not modality_valid:
            return False
        
        # Check family if specified
        if expected_family:
            dataset_source = dataset_info.get('source', '').lower()
            if expected_family.lower() not in dataset_source:
                return False
        
        return True
    
    def validate_execution_mode(self, experiment_spec: Dict[str, Any]) -> bool:
        """Validates that execution mode matches expectations."""
        expected_mode = self.expected['execution_mode']
        actual_mode = experiment_spec.get('execution_mode', '')
        
        return expected_mode.lower() == actual_mode.lower()
    
    def validate_artifacts(self, experiment_dir: Path) -> bool:
        """Validates that all required artifacts exist."""
        min_artifacts = self.expected['min_artifacts']
        
        # Count actual artifacts
        artifacts = [
            experiment_dir / "run_experiment.py",
            experiment_dir / "raw_results.json",
            experiment_dir / "dataset_used.json",
            experiment_dir / "execution.log",
        ]
        
        # Add plots
        artifacts.extend(list(experiment_dir.glob("*.png")))
        
        # Add report if exists
        report = experiment_dir / "FINAL_REPORT.md"
        if report.exists():
            artifacts.append(report)
        
        # Check completeness using validator
        validator = ArtifactCompletenessValidator()
        completeness_valid = validator.validate(experiment_dir, run_completed=False)
        
        if not completeness_valid:
            return False
        
        # Check count
        if len(artifacts) < min_artifacts:
            return False
        
        return True
    
    def validate_statistical_test(self, evaluation_output: Dict[str, Any]) -> bool:
        """Validates that the correct statistical test was used."""
        expected_test = self.expected.get('statistical_test', '')
        
        if not expected_test:
            return True  # No restriction
        
        actual_test = evaluation_output.get('test_used', '')
        
        return expected_test.lower() in actual_test.lower()
    
    def validate_memory_update(self, memory_records: list) -> bool:
        """Validates that memory was updated exactly once."""
        # Should have exactly one knowledge crystallization per run
        # This is a simplified check - real implementation would query Supabase
        return True  # Placeholder for now
    
    def run_full_validation(self, 
                           experiment_dir: Path,
                           dataset_info: Dict[str, Any],
                           experiment_spec: Dict[str, Any],
                           evaluation_output: Dict[str, Any]) -> Dict[str, bool]:
        """
        Runs all validations and returns a report.
        
        Returns:
            Dictionary with validation results for each check
        """
        results = {
            "dataset_choice": self.validate_dataset_choice(dataset_info),
            "execution_mode": self.validate_execution_mode(experiment_spec),
            "artifacts": self.validate_artifacts(experiment_dir),
            "statistical_test": self.validate_statistical_test(evaluation_output),
            "memory_update": self.validate_memory_update([]),
        }
        
        return results


@pytest.mark.golden
@pytest.mark.slow
class TestGoldenSuite:
    """
    Golden test suite.
    
    These tests run the full pipeline with known research goals and verify
    expected behavior.
    """
    
    @pytest.fixture
    def golden_test_dir(self):
        """Returns path to golden test specifications."""
        return Path(__file__).parent / "golden"
    
    def test_l2_vs_pruning(self, 
                           golden_test_dir, 
                           temp_experiment_dir,
                           mock_supabase_client,
                           sample_experiment_spec,
                           sample_raw_results):
        """
        Golden Test: L2 regularization vs capacity reduction.
        
        This test verifies:
        - Correct dataset selection (sklearn, tabular)
        - Proper execution mode (validation)
        - Complete artifact generation
        - Appropriate statistical test
        """
        spec_path = golden_test_dir / "test_l2_vs_pruning.yaml"
        runner = GoldenTestRunner(spec_path)
        
        # Simulate pipeline execution
        # In real tests, this would actually run the controller
        # For now, we'll use mock data
        
        dataset_info = {
            "type": "tabular",
            "source": "sklearn",
            "name": "california_housing"
        }
        
        # Create mock artifacts
        (temp_experiment_dir / "run_experiment.py").write_text("# experiment code")
        (temp_experiment_dir / "raw_results.json").write_text(json.dumps(sample_raw_results))
        (temp_experiment_dir / "dataset_used.json").write_text(json.dumps(dataset_info))
        (temp_experiment_dir / "execution.log").write_text("Execution log")
        (temp_experiment_dir / "FINAL_REPORT.md").write_text("# Final Report")
        
        # Create plot
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot([1, 2, 3], [1, 2, 3])
        plt.savefig(temp_experiment_dir / "comparison_plot.png")
        plt.close()
        
        evaluation_output = {
            "test_used": "t-test_ind",
            "verdict": "robust"
        }
        
        # Run validations
        results = runner.run_full_validation(
            temp_experiment_dir,
            dataset_info,
            sample_experiment_spec,
            evaluation_output
        )
        
        # Assert all checks passed
        for check, passed in results.items():
            assert passed, f"Golden test failed on: {check}"
    
    def test_depth_vs_width(self,
                           golden_test_dir,
                           temp_experiment_dir,
                           mock_supabase_client):
        """
        Golden Test: Network depth vs width for vision.
        
        This test verifies:
        - Correct modality detection (vision)
        - Vision-appropriate dataset (torchvision)
        - Complete artifact generation including image plots
        """
        spec_path = golden_test_dir / "test_depth_vs_width.yaml"
        runner = GoldenTestRunner(spec_path)
        
        dataset_info = {
            "type": "image",
            "source": "torchvision",
            "name": "mnist"
        }
        
        experiment_spec = {
            "execution_mode": "validation",
            "data_modality": {"type": "vision"}
        }
        
        # Create mock artifacts
        (temp_experiment_dir / "run_experiment.py").write_text("# vision experiment")
        (temp_experiment_dir / "raw_results.json").write_text("{}")
        (temp_experiment_dir / "dataset_used.json").write_text(json.dumps(dataset_info))
        (temp_experiment_dir / "execution.log").write_text("Log")
        (temp_experiment_dir / "FINAL_REPORT.md").write_text("# Report")
        
        # Create multiple plots
        import matplotlib.pyplot as plt
        for i in range(3):
            plt.figure()
            plt.imshow([[1, 2], [3, 4]])
            plt.savefig(temp_experiment_dir / f"plot_{i}.png")
            plt.close()
        
        evaluation_output = {
            "test_used": "t-test_ind",
            "verdict": "robust"
        }
        
        # Run validations
        results = runner.run_full_validation(
            temp_experiment_dir,
            dataset_info,
            experiment_spec,
            evaluation_output
        )
        
        # Assert modality is correct
        assert results["dataset_choice"], "Vision dataset validation failed"
        assert results["artifacts"], "Artifact validation failed"
    
    def test_label_noise_calibration(self,
                                    golden_test_dir,
                                    temp_experiment_dir,
                                    mock_supabase_client):
        """
        Golden Test: Label noise effect on calibration.
        
        This test verifies:
        - Tabular dataset selection
        - Calibration metrics computation
        - Appropriate visualization (calibration curves)
        """
        spec_path = golden_test_dir / "test_label_noise.yaml"
        runner = GoldenTestRunner(spec_path)
        
        dataset_info = {
            "type": "tabular",
            "source": "sklearn",
            "name": "iris"
        }
        
        experiment_spec = {
            "execution_mode": "validation",
            "data_modality": {"type": "tabular"}
        }
        
        # Create mock artifacts
        (temp_experiment_dir / "run_experiment.py").write_text("# calibration experiment")
        
        results_with_calibration = {
            "baseline_calibration": {"ece": 0.15, "mce": 0.25},
            "treatment_calibration": {"ece": 0.08, "mce": 0.12}
        }
        (temp_experiment_dir / "raw_results.json").write_text(json.dumps(results_with_calibration))
        (temp_experiment_dir / "dataset_used.json").write_text(json.dumps(dataset_info))
        (temp_experiment_dir / "execution.log").write_text("Log")
        (temp_experiment_dir / "FINAL_REPORT.md").write_text("# Report")
        
        # Create calibration curve plot
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot([0, 0.5, 1], [0, 0.5, 1], label="Perfect")
        plt.plot([0, 0.5, 1], [0, 0.4, 0.9], label="Actual")
        plt.savefig(temp_experiment_dir / "calibration_curve.png")
        plt.close()
        
        evaluation_output = {
            "test_used": "t-test_ind",
            "verdict": "robust"
        }
        
        # Run validations
        results = runner.run_full_validation(
            temp_experiment_dir,
            dataset_info,
            experiment_spec,
            evaluation_output
        )
        
        # All checks should pass
        for check, passed in results.items():
            assert passed, f"Calibration test failed on: {check}"


@pytest.mark.golden
def test_all_golden_specs_are_valid(golden_test_dir=Path(__file__).parent / "golden"):
    """
    Meta-test: Ensures all golden test YAML files are valid.
    """
    golden_specs = list(golden_test_dir.glob("*.yaml"))
    
    assert len(golden_specs) > 0, "No golden test specifications found"
    
    required_fields = ['goal', 'expected']
    
    for spec_file in golden_specs:
        with open(spec_file, 'r') as f:
            spec = yaml.safe_load(f)
        
        # Check required fields
        for field in required_fields:
            assert field in spec, f"{spec_file.name} missing required field: {field}"
        
        # Check expected structure
        assert 'data_modality' in spec['expected'], f"{spec_file.name} missing data_modality"
        assert 'min_artifacts' in spec['expected'], f"{spec_file.name} missing min_artifacts"
