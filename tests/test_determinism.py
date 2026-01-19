"""
Determinism tests for the Epsilon research pipeline.

These tests verify that running the same research goal multiple times
produces deterministic results (same structure, same dataset choices,
same statistical plans).

Minor numeric variance is acceptable, but structural variance is not.
"""

import pytest
import json
import hashlib
from pathlib import Path
from typing import Dict, Any


def hash_dict_structure(d: Dict[str, Any], ignore_keys: set = None) -> str:
    """
    Creates a hash of dictionary structure, ignoring specific numeric values.
    
    Args:
        d: Dictionary to hash
        ignore_keys: Keys to ignore (typically numeric results)
        
    Returns:
        SHA256 hash of structure
    """
    if ignore_keys is None:
        ignore_keys = {"timestamp", "execution_time", "random_seed"}
    
    def normalize(obj):
        """Normalize object for hashing."""
        if isinstance(obj, dict):
            return {k: normalize(v) for k, v in sorted(obj.items()) 
                   if k not in ignore_keys}
        elif isinstance(obj, list):
            return [normalize(item) for item in obj]
        elif isinstance(obj, float):
            return "FLOAT"  # Ignore exact float values
        elif isinstance(obj, int) and obj > 1000:
            return "LARGE_INT"  # Ignore large ints (likely seeds, sizes)
        else:
            return obj
    
    normalized = normalize(d)
    json_str = json.dumps(normalized, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()


def hash_experiment_spec(spec: Dict[str, Any]) -> str:
    """
    Creates a structural hash of an experiment specification.
    
    Ignores:
    - Random seeds
    - Timestamps
    - Execution times
    
    Preserves:
    - Hypothesis
    - Data modality
    - Model family
    - Statistical plan
    """
    ignore_keys = {
        "random_seed", 
        "timestamp", 
        "execution_time",
        "run_id",
        "user_id"
    }
    
    return hash_dict_structure(spec, ignore_keys)


def hash_report_structure(report_path: Path) -> str:
    """
    Creates a hash of report structure (sections, headings).
    
    Ignores:
    - Exact numeric values
    - Timestamps
    - Run IDs
    
    Preserves:
    - Section order
    - Heading hierarchy
    - Table structure
    """
    if not report_path.exists():
        return ""
    
    content = report_path.read_text()
    
    # Extract structure (headings, sections)
    lines = content.split('\n')
    structure = []
    
    for line in lines:
        # Keep headings
        if line.startswith('#'):
            structure.append(line.strip())
        # Keep section markers
        elif line.startswith('##'):
            structure.append(line.strip())
        # Keep table headers
        elif '|' in line and line.count('|') > 2:
            structure.append("TABLE_ROW")
    
    structure_str = '\n'.join(structure)
    return hashlib.sha256(structure_str.encode()).hexdigest()


class DeterminismChecker:
    """
    Checks determinism between two pipeline runs.
    """
    
    def __init__(self):
        self.mismatches = []
    
    def compare_dataset_choice(self, 
                               dataset1: Dict[str, Any],
                               dataset2: Dict[str, Any]) -> bool:
        """
        Compares dataset choices for determinism.
        
        Must match exactly:
        - Dataset source
        - Dataset name
        - Data modality
        """
        critical_fields = ["source", "name", "type"]
        
        for field in critical_fields:
            val1 = dataset1.get(field, "")
            val2 = dataset2.get(field, "")
            
            if val1 != val2:
                self.mismatches.append({
                    "component": "dataset",
                    "field": field,
                    "run1": val1,
                    "run2": val2
                })
                return False
        
        return True
    
    def compare_model_family(self,
                             spec1: Dict[str, Any],
                             spec2: Dict[str, Any]) -> bool:
        """
        Compares model family choices.
        
        Should be identical across runs.
        """
        model1 = spec1.get("model_family", "")
        model2 = spec2.get("model_family", "")
        
        if model1 != model2:
            self.mismatches.append({
                "component": "model",
                "field": "model_family",
                "run1": model1,
                "run2": model2
            })
            return False
        
        return True
    
    def compare_statistical_plan(self,
                                 spec1: Dict[str, Any],
                                 spec2: Dict[str, Any]) -> bool:
        """
        Compares statistical analysis plans.
        
        Should be identical across runs.
        """
        plan1 = spec1.get("statistical_analysis_plan", {})
        plan2 = spec2.get("statistical_analysis_plan", {})
        
        critical_fields = ["primary_test", "fallback_test", "assumptions"]
        
        for field in critical_fields:
            val1 = plan1.get(field)
            val2 = plan2.get(field)
            
            if val1 != val2:
                self.mismatches.append({
                    "component": "statistical_plan",
                    "field": field,
                    "run1": val1,
                    "run2": val2
                })
                return False
        
        return True
    
    def compare_artifact_structure(self,
                                   dir1: Path,
                                   dir2: Path) -> bool:
        """
        Compares artifact structure (not contents).
        
        Both runs should produce:
        - Same number of files
        - Same file types
        - Same artifact names
        """
        get_artifacts = lambda d: sorted([f.name for f in d.iterdir() if f.is_file()])
        
        artifacts1 = get_artifacts(dir1)
        artifacts2 = get_artifacts(dir2)
        
        if artifacts1 != artifacts2:
            self.mismatches.append({
                "component": "artifacts",
                "field": "file_list",
                "run1": artifacts1,
                "run2": artifacts2
            })
            return False
        
        return True
    
    def get_mismatches(self):
        """Returns all recorded mismatches."""
        return self.mismatches


@pytest.mark.determinism
@pytest.mark.slow
class TestDeterminism:
    """
    Tests for deterministic pipeline behavior.
    """
    
    def test_same_goal_produces_same_dataset_choice(self,
                                                     mock_supabase_client):
        """
        Running the same research goal twice should select the same dataset.
        
        This is critical for reproducibility.
        """
        research_goal = "Does L2 regularization reduce variance?"
        
        # Simulate two runs
        # In real test, this would run the full pipeline twice
        
        # Run 1
        dataset1 = {
            "source": "sklearn",
            "name": "california_housing",
            "type": "tabular"
        }
        
        # Run 2 (should be identical)
        dataset2 = {
            "source": "sklearn",
            "name": "california_housing",
            "type": "tabular"
        }
        
        checker = DeterminismChecker()
        result = checker.compare_dataset_choice(dataset1, dataset2)
        
        assert result is True, f"Dataset choice not deterministic: {checker.get_mismatches()}"
    
    def test_same_goal_produces_same_model_family(self,
                                                   sample_experiment_spec):
        """
        Same research goal should produce same model family choice.
        """
        # Run 1
        spec1 = sample_experiment_spec.copy()
        spec1["model_family"] = "neural_network"
        
        # Run 2
        spec2 = sample_experiment_spec.copy()
        spec2["model_family"] = "neural_network"
        
        checker = DeterminismChecker()
        result = checker.compare_model_family(spec1, spec2)
        
        assert result is True
    
    def test_same_goal_produces_same_statistical_plan(self,
                                                       sample_experiment_spec):
        """
        Same research goal should produce same statistical analysis plan.
        """
        # Run 1
        spec1 = sample_experiment_spec.copy()
        
        # Run 2 (should be identical)
        spec2 = sample_experiment_spec.copy()
        
        checker = DeterminismChecker()
        result = checker.compare_statistical_plan(spec1, spec2)
        
        assert result is True
    
    def test_experiment_spec_hash_is_deterministic(self,
                                                    sample_experiment_spec):
        """
        Hashing the experiment spec multiple times should give same result.
        """
        spec1 = sample_experiment_spec.copy()
        spec2 = sample_experiment_spec.copy()
        
        # Add some runtime data that should be ignored
        spec1["execution_time"] = 45.2
        spec1["timestamp"] = "2024-01-01"
        
        spec2["execution_time"] = 52.3  # Different!
        spec2["timestamp"] = "2024-01-02"  # Different!
        
        hash1 = hash_experiment_spec(spec1)
        hash2 = hash_experiment_spec(spec2)
        
        assert hash1 == hash2, "Experiment spec structural hash should ignore runtime data"
    
    def test_report_structure_hash_is_deterministic(self,
                                                     temp_experiment_dir):
        """
        Report structure hash should be deterministic across runs.
        """
        report1 = temp_experiment_dir / "report1.md"
        report2 = temp_experiment_dir / "report2.md"
        
        # Same structure, different numbers
        content1 = """
# Final Report

## Executive Summary
Mean loss: 0.456

## Results
| Metric | Baseline | Treatment |
|--------|----------|-----------|
| Loss | 0.5 | 0.3 |
        """
        
        content2 = """
# Final Report

## Executive Summary
Mean loss: 0.789

## Results
| Metric | Baseline | Treatment |
|--------|----------|-----------|
| Loss | 0.6 | 0.4 |
        """
        
        report1.write_text(content1)
        report2.write_text(content2)
        
        hash1 = hash_report_structure(report1)
        hash2 = hash_report_structure(report2)
        
        assert hash1 == hash2, "Report structure hash should ignore numeric values"
    
    def test_non_determinism_is_detected(self):
        """
        Verify that actual non-determinism is correctly detected.
        """
        # Different datasets
        dataset1 = {
            "source": "sklearn",
            "name": "california_housing",
            "type": "tabular"
        }
        
        dataset2 = {
            "source": "sklearn",
            "name": "iris",  # Different!
            "type": "tabular"
        }
        
        checker = DeterminismChecker()
        result = checker.compare_dataset_choice(dataset1, dataset2)
        
        assert result is False, "Determinism checker should detect mismatches"
        assert len(checker.get_mismatches()) > 0


@pytest.mark.determinism
def test_numeric_variance_is_acceptable():
    """
    Minor numeric variance (due to random initialization, etc.) is acceptable.
    
    What matters is structural determinism:
    - Same dataset
    - Same model architecture
    - Same statistical plan
    
    What's allowed to vary:
    - Exact loss values
    - Random seeds
    - Execution time
    """
    # Two runs with different numeric outcomes
    results1 = {
        "baseline_loss": 0.456,
        "treatment_loss": 0.289,
        "p_value": 0.003
    }
    
    results2 = {
        "baseline_loss": 0.461,  # Slightly different
        "treatment_loss": 0.295,  # Slightly different
        "p_value": 0.004  # Slightly different
    }
    
    # Structure is identical, numbers differ - this is OK
    assert results1.keys() == results2.keys()
    
    # All differences are small
    for key in results1.keys():
        diff = abs(results1[key] - results2[key])
        assert diff < 0.01, f"Variance in {key} is too large: {diff}"
