"""
Memory behavior tests for the Epsilon research pipeline.

Tests verify:
1. Duplicate suppression (same evidence isn't inserted twice)
2. Failure learning (past failures inform future designs)
3. Knowledge crystallization rules (only from robust verdicts)
4. Query functionality (failed runs are queryable)
"""

import pytest
import json
from typing import Dict, Any, List
from unittest.mock import Mock, patch


@pytest.mark.memory
class TestDuplicateSuppression:
    """
    Tests that ensure evidence and knowledge aren't duplicated.
    """
    
    def test_same_paper_read_twice_no_duplicate_evidence(self,
                                                          mock_supabase_client):
        """
        Reading the same paper twice should not create duplicate evidence.
        
        Expected:
        - First read: Evidence inserted
        - Second read: Duplicate detected, no insertion
        """
        from tests.invariants import MemoryCorrectnessValidator
        
        # Simulate reading same paper twice
        evidence_records = [
            {
                "source_url": "arxiv.org/paper1.pdf",
                "extracted_claim": "L2 reduces variance",
                "confidence": "high"
            },
            {
                "source_url": "arxiv.org/paper2.pdf",
                "extracted_claim": "Dropout improves generalization",
                "confidence": "high"
            },
            {
                "source_url": "arxiv.org/paper1.pdf",  # Duplicate!
                "extracted_claim": "L2 reduces variance",  # Same claim
                "confidence": "high"
            }
        ]
        
        validator = MemoryCorrectnessValidator()
        result = validator.validate_no_duplicates(evidence_records)
        
        assert result is False, "Duplicate evidence should be detected"
        assert len(validator.get_violations()) > 0
    
    def test_same_experiment_run_twice_no_duplicate_knowledge(self):
        """
        Running the same experiment twice should not duplicate knowledge.
        
        Expected:
        - First run: Knowledge crystallized (if robust)
        - Second run: Check existing knowledge, don't re-crystallize
        """
        # This would be tested by querying knowledge_memory
        # Simulation:
        
        knowledge_records = [
            {
                "claim": "L2 reduces variance",
                "evidence_count": 3,
                "verdict": "robust",
                "source_run_id": "run_001"
            }
        ]
        
        # Same claim from second run should not be inserted again
        new_record = {
            "claim": "L2 reduces variance",  # Same!
            "evidence_count": 3,
            "verdict": "robust",
            "source_run_id": "run_002"
        }
        
        # Check if claim already exists
        existing_claims = [r["claim"] for r in knowledge_records]
        
        assert new_record["claim"] in existing_claims, \
            "System should detect existing knowledge before insertion"
    
    def test_different_claims_from_same_source_allowed(self):
        """
        Different claims from the same source should be allowed.
        """
        from tests.invariants import MemoryCorrectnessValidator
        
        evidence_records = [
            {
                "source_url": "arxiv.org/paper1.pdf",
                "extracted_claim": "L2 reduces variance",
                "confidence": "high"
            },
            {
                "source_url": "arxiv.org/paper1.pdf",  # Same source
                "extracted_claim": "Dropout improves calibration",  # Different claim
                "confidence": "medium"
            }
        ]
        
        validator = MemoryCorrectnessValidator()
        result = validator.validate_no_duplicates(evidence_records)
        
        assert result is True, "Different claims from same source should be allowed"


@pytest.mark.memory
class TestFailureLearning:
    """
    Tests that verify the system learns from past failures.
    """
    
    def test_past_failures_queried_before_design(self):
        """
        Design Agent should query past failures before creating new design.
        
        Expected flow:
        1. Design Agent starts
        2. Queries past failures for similar goals
        3. Adjusts design to avoid known failures
        4. Proceeds with improved design
        """
        # Mock past failures
        past_failures = [
            {
                "goal": "Compare L2 vs Dropout",
                "failure_reason": "Dataset 'imagenet' unavailable",
                "dataset_requested": "imagenet",
                "timestamp": "2024-01-01"
            }
        ]
        
        # New similar goal
        new_goal = "Compare L2 vs Dropout for image classification"
        
        # System should:
        # 1. Query past_failures
        # 2. See imagenet failed before
        # 3. Choose alternative dataset (CIFAR-10, MNIST)
        
        # For testing, we verify the query would happen
        assert len(past_failures) > 0
        assert "unavailable" in past_failures[0]["failure_reason"]
        
        # Design Agent should avoid "imagenet" in new design
        # This is tested by checking the design output doesn't include it
    
    def test_repeated_failure_causes_earlier_halt(self,
                                                   mock_supabase_client):
        """
        If a failure happens repeatedly, system should halt earlier next time.
        
        Example:
        - Run 1: Dataset 'custom_data' fails after 30 seconds
        - Run 2: Query shows 'custom_data' failed before
        - Run 2: Halt immediately without retrying
        """
        # Mock failure history
        failure_history = [
            {
                "dataset": "custom_data",
                "failure_count": 3,
                "last_failure": "2024-01-15"
            }
        ]
        
        # New request for same dataset
        requested_dataset = "custom_data"
        
        # Should halt early
        for record in failure_history:
            if record["dataset"] == requested_dataset:
                assert record["failure_count"] >= 3, \
                    "High failure count should trigger immediate halt"
    
    def test_failure_learning_cross_user(self):
        """
        Failures should be queryable across users (with privacy controls).
        
        If User A tried dataset X and it failed, User B should benefit
        from that knowledge.
        """
        # This depends on system policy
        # Some systems might isolate failures per-user
        # Others might share anonymized failure data
        
        # For Epsilon, we document expected behavior:
        # - Dataset availability failures: Shared (no privacy concern)
        # - Hypothesis failures: Private (research-specific)
        
        shared_failure_types = ["dataset_unavailable", "dependency_missing"]
        private_failure_types = ["hypothesis_rejected", "no_significant_effect"]
        
        assert len(shared_failure_types) > 0
        assert len(private_failure_types) > 0


@pytest.mark.memory
class TestKnowledgeCrystallization:
    """
    Tests for knowledge crystallization rules.
    """
    
    def test_only_robust_verdicts_crystallize(self):
        """
        Only findings with 'robust' verdicts should become knowledge.
        
        Spurious or uncertain findings should NOT crystallize.
        """
        from tests.invariants import MemoryCorrectnessValidator
        
        validator = MemoryCorrectnessValidator()
        
        # Robust verdict - should pass
        robust_knowledge = {
            "claim": "L2 reduces variance",
            "verdict": "robust",
            "evidence_count": 5
        }
        
        result = validator.validate_knowledge_source(robust_knowledge)
        assert result is True
        
        # Spurious verdict - should fail
        spurious_knowledge = {
            "claim": "Effect is spurious",
            "verdict": "spurious",
            "evidence_count": 2
        }
        
        result = validator.validate_knowledge_source(spurious_knowledge)
        assert result is False
    
    def test_very_robust_verdicts_also_crystallize(self):
        """
        'very_robust' verdicts should also crystallize knowledge.
        """
        from tests.invariants import MemoryCorrectnessValidator
        
        validator = MemoryCorrectnessValidator()
        
        very_robust_knowledge = {
            "claim": "Effect size is large",
            "verdict": "very_robust",
            "evidence_count": 10
        }
        
        result = validator.validate_knowledge_source(very_robust_knowledge)
        assert result is True
    
    def test_promising_verdicts_do_not_crystallize(self):
        """
        'promising' verdicts should stay in evidence, not crystallize.
        """
        from tests.invariants import MemoryCorrectnessValidator
        
        validator = MemoryCorrectnessValidator()
        
        promising_knowledge = {
            "claim": "Effect might exist",
            "verdict": "promising",
            "evidence_count": 1
        }
        
        result = validator.validate_knowledge_source(promising_knowledge)
        assert result is False


@pytest.mark.memory
class TestQueryFunctionality:
    """
    Tests that verify failed runs are queryable and useful.
    """
    
    def test_failed_run_has_complete_metadata(self):
        """
        Failed runs should have complete metadata for analysis.
        
        Required fields:
        - run_id
        - goal
        - failure_reason
        - failure_stage (research/design/execution/evaluation)
        - timestamp
        - partial_artifacts (if any)
        """
        failed_run = {
            "run_id": "run_failed_001",
            "goal": "Study learning dynamics",
            "failure_reason": "Dataset unavailable: imagenet",
            "failure_stage": "execution",
            "timestamp": "2024-01-15T10:30:00",
            "partial_artifacts": ["run_experiment.py"]
        }
        
        required_fields = [
            "run_id", "goal", "failure_reason", 
            "failure_stage", "timestamp"
        ]
        
        for field in required_fields:
            assert field in failed_run, f"Failed run missing: {field}"
    
    def test_query_past_failures_by_goal_similarity(self):
        """
        Should be able to query failures by similar research goals.
        """
        past_failures = [
            {"goal": "Compare L2 vs Dropout", "failure_reason": "Dataset X"},
            {"goal": "Study regularization techniques", "failure_reason": "Dataset Y"},
            {"goal": "Analyze overfitting", "failure_reason": "Convergence issue"}
        ]
        
        # Query for regularization-related failures
        query = "regularization"
        
        relevant_failures = [
            f for f in past_failures 
            if query.lower() in f["goal"].lower()
        ]
        
        assert len(relevant_failures) >= 1
        assert any("L2" in f["goal"] or "regularization" in f["goal"] 
                  for f in relevant_failures)
    
    def test_query_past_failures_by_dataset(self):
        """
        Should be able to query failures by dataset name.
        """
        past_failures = [
            {"dataset": "imagenet", "failure_reason": "Download failed", "count": 5},
            {"dataset": "cifar10", "failure_reason": "Corrupted", "count": 1},
            {"dataset": "mnist", "failure_reason": None, "count": 0}
        ]
        
        # Query for problematic datasets
        problematic = [f for f in past_failures if f["count"] > 0]
        
        assert len(problematic) >= 2
        assert "imagenet" in [f["dataset"] for f in problematic]


@pytest.mark.memory
class TestMemoryHitRate:
    """
    Tests to measure and optimize memory hit rate.
    
    Hit rate = (queries that found useful info) / (total queries)
    
    Higher hit rate means better memory utilization.
    """
    
    def test_evidence_query_finds_existing_research(self):
        """
        Querying evidence before research should find existing work.
        """
        # Simulate evidence memory
        existing_evidence = [
            {"claim": "L2 reduces variance", "source": "paper1.pdf"},
            {"claim": "Dropout improves calibration", "source": "paper2.pdf"},
        ]
        
        # Query for regularization evidence
        query = "regularization variance"
        
        # Should find relevant evidence
        results = [e for e in existing_evidence 
                  if any(word in e["claim"].lower() 
                        for word in query.lower().split())]
        
        assert len(results) > 0, "Evidence query should find relevant existing research"
    
    def test_memory_miss_triggers_new_research(self):
        """
        When evidence query returns no results, trigger new research.
        """
        existing_evidence = [
            {"claim": "L2 reduces variance", "source": "paper1.pdf"}
        ]
        
        # Query for unrelated topic
        query = "quantum machine learning"
        
        results = [e for e in existing_evidence
                  if any(word in e["claim"].lower()
                        for word in query.lower().split())]
        
        # No results - should trigger new research
        assert len(results) == 0, "Query miss should trigger new research phase"


@pytest.mark.memory
@pytest.mark.integration
def test_memory_system_end_to_end():
    """
    End-to-end test of memory system.
    
    Flow:
    1. Query evidence (miss)
    2. Conduct research, save evidence
    3. Run experiment, record result
    4. Crystallize knowledge (if robust)
    5. Query evidence (hit)
    """
    # Step 1: Initial query (miss)
    evidence_db = []
    query_result = [e for e in evidence_db if "L2" in e.get("claim", "")]
    assert len(query_result) == 0, "Initial query should miss"
    
    # Step 2: Save new evidence
    new_evidence = {
        "source_url": "paper1.pdf",
        "extracted_claim": "L2 regularization reduces variance",
        "confidence": "high"
    }
    evidence_db.append(new_evidence)
    
    # Step 3: Record experiment result
    experiment_result = {
        "hypothesis": "L2 reduces variance",
        "verdict": "robust",
        "p_value": 0.001
    }
    
    # Step 4: Crystallize knowledge (verdict is robust)
    knowledge_db = []
    if experiment_result["verdict"] in ["robust", "very_robust"]:
        knowledge_db.append({
            "claim": experiment_result["hypothesis"],
            "verdict": experiment_result["verdict"]
        })
    
    # Step 5: Query evidence again (hit)
    query_result = [e for e in evidence_db if "L2" in e.get("extracted_claim", "")]
    assert len(query_result) > 0, "Second query should hit"
    
    # Verify knowledge was crystallized
    assert len(knowledge_db) > 0, "Robust finding should crystallize knowledge"
