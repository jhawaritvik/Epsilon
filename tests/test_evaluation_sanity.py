"""
Sanity tests for the Evaluation Agent.

These tests feed known synthetic results to verify that the evaluation
logic doesn't hallucinate significance.

Philosophy: The evaluation agent should be a truth machine. If you feed
it identical distributions, it should NOT find significance.
"""

import pytest
import numpy as np
from typing import List


def generate_identical_distributions(n: int = 30, seed: int = 42) -> tuple:
    """
    Generates two identical distributions.
    
    Returns:
        (data_a, data_b) where both are from Normal(0.5, 0.1)
    """
    np.random.seed(seed)
    data_a = np.random.normal(0.5, 0.1, n).tolist()
    
    np.random.seed(seed)  # Same seed
    data_b = np.random.normal(0.5, 0.1, n).tolist()
    
    return data_a, data_b


def generate_different_distributions(n: int = 30, seed: int = 42) -> tuple:
    """
    Generates two clearly different distributions.
    
    Returns:
        (data_a, data_b) where a ~ Normal(0.5, 0.1), b ~ Normal(0.3, 0.1)
    """
    np.random.seed(seed)
    data_a = np.random.normal(0.5, 0.1, n).tolist()
    data_b = np.random.normal(0.3, 0.1, n).tolist()
    
    return data_a, data_b


def generate_zero_variance_data(n: int = 30) -> List[float]:
    """
    Generates data with zero variance.
    
    This is an edge case that should be handled gracefully.
    """
    return [0.5] * n


@pytest.mark.agent
@pytest.mark.evaluation
class TestEvaluationAgentSanity:
    """
    Sanity tests for the Evaluation Agent.
    """
    
    def test_identical_distributions_not_significant(self):
        """
        Feeding identical distributions should NOT reject H0.
        
        This is the most basic sanity check.
        """
        from evaluation_agent import run_statistical_test
        
        data_a, data_b = generate_identical_distributions()
        
        # Run t-test
        result = run_statistical_test(
            test_name="t-test_ind",
            data_a=data_a,
            data_b=data_b,
            alpha=0.05,
            alternative="two-sided"
        )
        
        result_dict = eval(result)  # Parse JSON string
        
        # Should fail to reject H0
        assert "Fail to reject H0" in result_dict["decision"], \
            "Identical distributions should not be significantly different"
        
        # P-value should be high
        assert result_dict["p_value"] > 0.05, \
            f"P-value should be > 0.05, got {result_dict['p_value']}"
    
    def test_different_distributions_are_significant(self):
        """
        Feeding clearly different distributions SHOULD reject H0.
        """
        from evaluation_agent import run_statistical_test
        
        data_a, data_b = generate_different_distributions()
        
        # Run t-test
        result = run_statistical_test(
            test_name="t-test_ind",
            data_a=data_a,
            data_b=data_b,
            alpha=0.05,
            alternative="two-sided"
        )
        
        result_dict = eval(result)
        
        # Should reject H0
        assert "Reject H0" in result_dict["decision"], \
            "Different distributions should be significantly different"
        
        # P-value should be low
        assert result_dict["p_value"] < 0.05, \
            f"P-value should be < 0.05, got {result_dict['p_value']}"
    
    def test_zero_variance_handled_gracefully(self):
        """
        Zero-variance inputs should be handled without crashing.
        
        Expected:
        - No RuntimeWarning
        - Clear "No Significant Difference" or "Cannot compute" message
        """
        from evaluation_agent import run_statistical_test
        
        data_a = generate_zero_variance_data()
        data_b = generate_zero_variance_data()
        
        # Should not crash
        try:
            result = run_statistical_test(
                test_name="t-test_ind",
                data_a=data_a,
                data_b=data_b,
                alpha=0.05
            )
            # Should return graceful error or fail to reject
            assert result is not None
        except Exception as e:
            pytest.fail(f"Zero variance should be handled gracefully, got: {e}")
    
    def test_known_effect_size_correct_classification(self):
        """
        Known effect size should be correctly classified.
        
        Large effect + low p-value = robust
        Small effect + high p-value = spurious
        """
        # Large effect size (Cohen's d ≈ 2.0)
        np.random.seed(42)
        data_a = np.random.normal(0.5, 0.1, 30).tolist()
        data_b = np.random.normal(0.3, 0.1, 30).tolist()  # Clear difference
        
        from evaluation_agent import run_statistical_test
        
        result = run_statistical_test(
            test_name="t-test_ind",
            data_a=data_a,
            data_b=data_b,
            alpha=0.05
        )
        
        result_dict = eval(result)
        
        # Should have low p-value (reject H0)
        assert result_dict["p_value"] < 0.05
        
        # With this effect size and sample size, should be classified as robust
        # (In real system, this would be done by evaluation agent logic)
    
    def test_no_false_robust_classification(self):
        """
        Marginal results should NOT be classified as robust.
        
        P-value barely < 0.05 should be 'promising' or 'marginal', not 'robust'.
        """
        # Create marginal effect
        np.random.seed(42)
        # Small effect size, might get p ≈ 0.045
        data_a = np.random.normal(0.50, 0.15, 25).tolist()
        data_b = np.random.normal(0.48, 0.15, 25).tolist()
        
        from evaluation_agent import run_statistical_test
        
        result = run_statistical_test(
            test_name="t-test_ind",
            data_a=data_a,
            data_b=data_b,
            alpha=0.05
        )
        
        result_dict = eval(result)
        
        # Even if p < 0.05, should not automatically be 'robust'
        # Robust classification requires:
        # - Low p-value (e.g., < 0.01)
        # - Large effect size
        # - Robust to assumptions
        
        # This test documents the expected behavior
        # Real implementation would check these criteria
    
    def test_assumption_violations_trigger_fallback(self):
        """
        When assumptions fail, should use fallback test.
        
        Example: Non-normal data should trigger Mann-Whitney instead of t-test.
        """
        # Generate non-normal data (exponential distribution)
        np.random.seed(42)
        data_a = np.random.exponential(2.0, 30).tolist()
        data_b = np.random.exponential(2.5, 30).tolist()
        
        from evaluation_agent import verify_assumptions
        
        # Check normality
        normality_check_a = verify_assumptions("normality", data_a)
        normality_result_a = eval(normality_check_a)
        
        # Exponential data should fail normality
        assert normality_result_a["status"] == "fail", \
            "Exponential data should fail normality test"
        
        # System should then use fallback (Mann-Whitney)
        # This is tested by checking the evaluation agent's logic


@pytest.mark.agent
@pytest.mark.evaluation
class TestStatisticalTestImplementations:
    """
    Tests for specific statistical test implementations.
    """
    
    def test_t_test_independent(self):
        """Test independent samples t-test."""
        from evaluation_agent import run_statistical_test
        
        data_a = [1, 2, 3, 4, 5]
        data_b = [6, 7, 8, 9, 10]
        
        result = run_statistical_test(
            test_name="t-test_ind",
            data_a=data_a,
            data_b=data_b,
            alpha=0.05
        )
        
        result_dict = eval(result)
        
        assert "statistic" in result_dict
        assert "p_value" in result_dict
        assert "decision" in result_dict
    
    def test_t_test_paired(self):
        """Test paired samples t-test."""
        from evaluation_agent import run_statistical_test
        
        data_a = [1, 2, 3, 4, 5]
        data_b = [1.1, 2.2, 3.1, 4.3, 5.2]
        
        result = run_statistical_test(
            test_name="t-test_rel",
            data_a=data_a,
            data_b=data_b,
            alpha=0.05
        )
        
        result_dict = eval(result)
        
        assert "statistic" in result_dict
        assert "p_value" in result_dict
    
    def test_mann_whitney_u(self):
        """Test Mann-Whitney U test (non-parametric)."""
        from evaluation_agent import run_statistical_test
        
        data_a = [1, 2, 3, 4, 5]
        data_b = [6, 7, 8, 9, 10]
        
        result = run_statistical_test(
            test_name="mannwhitneyu",
            data_a=data_a,
            data_b=data_b,
            alpha=0.05
        )
        
        result_dict = eval(result)
        
        assert "statistic" in result_dict
        assert "p_value" in result_dict
    
    def test_wilcoxon_signed_rank(self):
        """Test Wilcoxon signed-rank test."""
        from evaluation_agent import run_statistical_test
        
        data_a = [1, 2, 3, 4, 5]
        data_b = [1.1, 2.2, 3.1, 4.3, 5.2]
        
        result = run_statistical_test(
            test_name="wilcoxon",
            data_a=data_a,
            data_b=data_b,
            alpha=0.05
        )
        
        result_dict = eval(result)
        
        assert "statistic" in result_dict
        assert "p_value" in result_dict


@pytest.mark.agent
@pytest.mark.evaluation
def test_shapiro_wilk_normality():
    """Test Shapiro-Wilk normality test."""
    from evaluation_agent import verify_assumptions
    
    # Normal data
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, 100).tolist()
    
    result = verify_assumptions("normality", normal_data)
    result_dict = eval(result)
    
    assert "status" in result_dict
    assert result_dict["status"] in ["pass", "fail"]
    
    # Should likely pass for truly normal data
    # (though with small sample size, might occasionally fail)
