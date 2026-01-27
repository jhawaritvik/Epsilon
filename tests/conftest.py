"""
Pytest configuration and shared fixtures for Epsilon testing.
"""

import pytest
import tempfile
import shutil
import os
import json
from pathlib import Path
from typing import Dict, Any, Generator
from unittest.mock import Mock, patch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tcl/Tk errors


# =============================================================================
# PETRI FUZZ TESTING HOOKS
# =============================================================================

def pytest_addoption(parser):
    """Add command line options for Petri fuzz tests."""
    parser.addoption(
        "--petri-run",
        action="store_true",
        default=False,
        help="Run Petri fuzz tests (requires API keys, incurs costs)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip Petri tests unless --petri-run is specified."""
    if config.getoption("--petri-run"):
        return
    
    skip_petri = pytest.mark.skip(reason="needs --petri-run option to run")
    for item in items:
        # Check for actual @pytest.mark.petri marker, not just 'petri' keyword
        # (keyword would match directory name tests/petri/)
        if item.get_closest_marker("petri") is not None:
            item.add_marker(skip_petri)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_experiment_dir() -> Generator[Path, None, None]:
    """
    Creates a temporary experiment directory for isolated testing.
    Automatically cleans up after the test.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="epsilon_test_"))
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_run_id() -> str:
    """Returns a consistent mock run ID for testing."""
    return "test-run-00000000-0000-0000-0000-000000000001"


@pytest.fixture
def mock_user_id() -> str:
    """Returns a consistent mock user ID for testing."""
    return "test-user-00000000-0000-0000-0000-000000000001"


@pytest.fixture
def clean_experiment_dir(temp_experiment_dir: Path):
    """
    Sets up a clean experiment directory with expected structure.
    """
    # Create subdirectories
    (temp_experiment_dir / "artifacts").mkdir(exist_ok=True)
    (temp_experiment_dir / "logs").mkdir(exist_ok=True)
    return temp_experiment_dir


@pytest.fixture
def mock_supabase_client():
    """
    Mocks the Supabase client to avoid external dependencies in tests.
    """
    with patch('memory.supabase_client.SupabaseManager') as MockManager:
        mock_instance = Mock()
        mock_client = Mock()
        mock_client.table.return_value.insert.return_value.execute.return_value = Mock()
        mock_client.table.return_value.select.return_value.execute.return_value = Mock(data=[])
        
        # Configure manager instance to return mock client
        mock_instance.client = mock_client
        mock_instance.is_enabled = True
        
        # Configure MockManager constructor to return the instance
        MockManager.return_value = mock_instance
        
        yield mock_client


@pytest.fixture
def sample_experiment_spec() -> Dict[str, Any]:
    """
    Returns a valid experiment specification for testing.
    """
    return {
        "hypothesis": "L2 regularization reduces variance compared to capacity reduction",
        "data_modality": {
            "type": "tabular",
            "task": "regression"
        },
        "dataset_requirements": {
            "family": "sklearn",
            "name": "california_housing",
            "type": "synthetic"
        },
        "model_family": "neural_network",
        "execution_mode": "validation",
        "hyperparameters": {
            "baseline": {"lambda": 0.0, "hidden_units": 64},
            "treatment": {"lambda": 0.01, "hidden_units": 64}
        },
        "statistical_analysis_plan": {
            "primary_test": "t-test_ind",
            "fallback_test": "mannwhitneyu",
            "alpha": 0.05,
            "assumptions": ["normality"]
        }
    }


@pytest.fixture
def sample_raw_results() -> Dict[str, Any]:
    """
    Returns sample raw results for testing.
    """
    import numpy as np
    np.random.seed(42)
    
    return {
        "baseline_metrics": {
            "losses": np.random.normal(0.5, 0.1, 30).tolist(),
            "mean_loss": 0.5,
            "std_loss": 0.1
        },
        "treatment_metrics": {
            "losses": np.random.normal(0.3, 0.08, 30).tolist(),
            "mean_loss": 0.3,
            "std_loss": 0.08
        },
        "execution_time": 45.2,
        "dataset_info": {
            "name": "california_housing",
            "samples": 1000,
            "features": 8
        }
    }


@pytest.fixture
def artifact_files(temp_experiment_dir: Path, sample_raw_results: Dict[str, Any]):
    """
    Creates expected artifact files in the temp directory.
    """
    artifacts = {
        "run_experiment.py": "# Mock experiment code\nprint('Running experiment')",
        "raw_results.json": json.dumps(sample_raw_results, indent=2),
        "dataset_used.json": json.dumps({
            "source": "sklearn",
            "name": "california_housing",
            "type": "tabular"
        }, indent=2),
        "execution.log": "INFO: Starting experiment\nINFO: Experiment completed\n"
    }
    
    # Write files
    for filename, content in artifacts.items():
        (temp_experiment_dir / filename).write_text(content)
    
    # Create a mock plot
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot([1, 2, 3], [1, 2, 3])
    plt.savefig(temp_experiment_dir / "comparison_plot.png")
    plt.close()
    
    return temp_experiment_dir


@pytest.fixture(autouse=True)
def reset_environment_variables():
    """
    Resets environment variables before each test.
    """
    # Store original values
    original_run_id = os.environ.get("CURRENT_RUN_ID")
    
    yield
    
    # Restore
    if original_run_id:
        os.environ["CURRENT_RUN_ID"] = original_run_id
    elif "CURRENT_RUN_ID" in os.environ:
        del os.environ["CURRENT_RUN_ID"]


@pytest.fixture
def mock_agent_runner():
    """
    Mocks the Agent Runner to avoid actual LLM calls during testing.
    """
    with patch('agents.Runner') as mock:
        mock_result = Mock()
        mock_result.final_output = json.dumps({
            "execution_status": "success",
            "artifacts": ["run_experiment.py", "raw_results.json"],
            "summary": "Test execution completed"
        })
        mock.run_sync.return_value = mock_result
        yield mock


@pytest.fixture
def invariant_violation_logger(temp_experiment_dir: Path):
    """
    Creates a logger for invariant violations.
    """
    log_file = temp_experiment_dir / "invariant_violations.log"
    return log_file


# Helper functions for assertions

def assert_artifact_exists(directory: Path, filename: str):
    """Helper to assert an artifact file exists."""
    filepath = directory / filename
    assert filepath.exists(), f"Expected artifact {filename} not found in {directory}"


def assert_valid_json_file(filepath: Path):
    """Helper to assert a file contains valid JSON."""
    assert filepath.exists(), f"File {filepath} does not exist"
    try:
        with open(filepath, 'r') as f:
            json.load(f)
    except json.JSONDecodeError as e:
        pytest.fail(f"File {filepath} does not contain valid JSON: {e}")


def assert_all_required_artifacts(directory: Path):
    """Helper to assert all required artifacts exist."""
    required = [
        "run_experiment.py",
        "raw_results.json",
        "dataset_used.json",
        "execution.log"
    ]
    
    for artifact in required:
        assert_artifact_exists(directory, artifact)
    
    # Check for at least one plot
    plots = list(directory.glob("*.png"))
    assert len(plots) > 0, f"No .png plots found in {directory}"
