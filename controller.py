import json
import logging
import time
import os
import uuid
from typing import Dict, Any, Optional, Callable

from agents import Runner
from research_agent import research_agent
from experiment_agent import experiment_design_agent
from execution_agent import code_execution_agent, EXPERIMENT_DIR
from evaluation_agent import evaluation_agent

# Memory Service & Types
# Memory Writers removed - replaced by MemoryService
from memory.memory_service import MemoryService
from memory.types import FailureType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Controller")

# Type for event callback
EventCallback = Callable[[str, Dict[str, Any]], None]

class ResearchController:
    def __init__(self, max_iterations: int = 5, event_callback: Optional[EventCallback] = None):
        self.max_iterations = max_iterations
        self.history = []
        self.research_corpus = [] # List of strings/notes
        self.current_iteration = 0
        
        # Event callback for live updates (API/WebSocket integration)
        self.event_callback = event_callback
        
        # Identity & Memory Service
        self.run_id = uuid.uuid4()
        logger.info(f"Initialized Run ID: {self.run_id}")
        
        self.memory_service = MemoryService()
    
    def emit(self, event_type: str, data: Optional[Dict[str, Any]] = None):
        """Emit an event if callback is registered."""
        if self.event_callback:
            payload = {"type": event_type, "run_id": str(self.run_id), **(data or {})}
            try:
                self.event_callback(event_type, payload)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")
        
    def run(self, research_goal: str):
        logger.info(f"Starting Research Process for Goal: {research_goal}")
        self.emit("run_started", {"goal": research_goal})
        
        # --- PHASE 1: EXPLORATION ---
        logger.info("--- PHASE 1: RESEARCH EXPLORATION ---")
        self.emit("agent_started", {"agent": "Research"})
        
        # Inject Run ID so Research Agent can save evidence
        os.environ["CURRENT_RUN_ID"] = str(self.run_id)
        
        # --- CLEANUP: Ensure fresh state for Contract Enforcement ---
        self._cleanup_artifacts()
        
        try:
            research_result = Runner.run_sync(research_agent, research_goal, max_turns=30)
        finally:
            del os.environ["CURRENT_RUN_ID"] # Cleanup


        
        # Validate corpus is not empty
        corpus_content = research_result.final_output.strip() if research_result.final_output else ""
        if not corpus_content:
            logger.error("Research Agent produced no output. Cannot proceed with experimentation.")
            self.emit("run_error", {"error": "No research corpus generated"})
            print("\n❌ ERROR: No research corpus generated.")
            print("The Research Agent failed to gather information. Please verify:")
            print("  - TAVILY_API_KEY is correctly configured")
            print("  - The research goal is clear and searchable")
            print("  - Network connectivity is available")
            return
        
        self.research_corpus = [corpus_content]
        logger.info("Research Corpus Constructed.")
        self.emit("agent_completed", {"agent": "Research"})
        
        feedback = None
        
        # --- PHASE 2: EXPERIMENTATION LOOP ---
        while self.current_iteration < self.max_iterations:
            logger.info(f"--- ITERATION {self.current_iteration}/{self.max_iterations} (0-indexed) ---")
            
            # 0. ADAPTIVE LOGIC: Check for consecutive design failures
            # We query the service for recent failures in this run
            # Since run_memory stores everything, we rely on filtering
            past_failures = self.memory_service.get_past_runs(
                goal=research_goal, 
                failure_type=FailureType.DESIGN, 
                limit=3
            )
            # Filter to current run_id if desired, or all runs (adaptive iteration across runs)
            # Here we just look at total recent failures to suggest simplifying.
            
            adaptive_hint = ""
            if len(past_failures) >= 2:
                adaptive_hint = "\n[ADAPTIVE HINT]: You have failed on design multiple times. Please SIMPLIFY the hypothesis or reduce variables."

            # 0.5. HISTORY INJECTION (Prevent Amnesia)
            # Fetch all previous iterations for THIS run
            run_history = self.memory_service.get_past_runs(run_id=self.run_id, limit=20)
            run_history.sort(key=lambda x: x.get("iteration", 0))
            
            history_summary = ""
            if run_history:
                history_summary = "HISTORY OF PREVIOUS ATTEMPTS (DO NOT REPEAT MISTAKES):\n"
                for entry in run_history:
                    itr = entry.get("iteration")
                    failed_type = entry.get("issue_type", "unknown")
                    rationale = entry.get("evaluation_verdict", {}).get("rationale", "No rationale")
                    history_summary += f"- Iteration {itr}: Failed ({failed_type}). Reason: {rationale}\n"

            # 1. DESIGN
            logger.info("Running Experiment Design Agent...")
            self.emit("agent_started", {"agent": "Design"})
            self.emit("iteration_started", {"iteration": self.current_iteration})
            design_prompt = f"""
User Goal: {research_goal}

Research Corpus:
{json.dumps([c[:15000] + "... (truncated)" if len(c) > 15000 else c for c in self.research_corpus])}

{history_summary}

Feedback from Previous Iteration (Immediate):
{feedback if feedback else "None (First Iteration)"}
{adaptive_hint}

Design a statistically rigorous experiment.
"""
            design_result = Runner.run_sync(experiment_design_agent, design_prompt)
            experiment_spec_str = design_result.final_output
            
            try:
                # Attempt to parse to ensure valid JSON before passing to next agent
                clean_spec = experiment_spec_str.strip().replace("```json", "").replace("```", "")
                experiment_spec = json.loads(clean_spec)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Experiment Spec. JSON Error: {e}")
                feedback = f"""Critical Error: Your previous output was not valid JSON.
Error: {str(e)}
Please ensure your output is ONLY a valid JSON object with no extra text, comments, or markdown formatting.

Your output that failed (first 500 chars):
{experiment_spec_str[:500]}...
"""
                
                # Record Failed Iteration (Design Failure)
                self.memory_service.write_run(
                    run_id=self.run_id,
                    iteration=self.current_iteration,
                    research_goal=research_goal,
                    experiment_spec={}, 
                    evaluation_verdict={"outcome_classification": "failed", "issue_type": FailureType.DESIGN, "rationale": "Invalid JSON output from Design Agent"},
                    feedback_passed=feedback
                )
                self.emit("iteration_failed", {"iteration": self.current_iteration, "message": "Design Agent produced invalid JSON"})
                self.current_iteration += 1  # Increment after recording
                continue

            # 1.5 Clean artifacts to prevent stale reads
            if os.path.exists(f"{EXPERIMENT_DIR}/raw_results.json"):
                os.remove(f"{EXPERIMENT_DIR}/raw_results.json")

            # 2. EXECUTION
            logger.info("Running Code & Execution Agent...")
            self.emit("agent_started", {"agent": "Execute"})
            self.emit("agent_completed", {"agent": "Design"})
            execution_prompt = f"""
Experiment Specification:
{json.dumps({
    "experiment_specification": experiment_spec.get("experiment_specification"),
    "statistical_analysis_plan": experiment_spec.get("statistical_analysis_plan"),
    "data_modality": experiment_spec.get("experiment_specification", {}).get("data_modality"),
    "revision_directives": experiment_spec.get("revision_directives"),
    "execution_mode": experiment_spec.get("execution_mode", "validation")
}, indent=2)}

Implement and execute this experiment.
"""

            execution_result = Runner.run_sync(code_execution_agent, execution_prompt)
            logger.info(f"Execution Output: {execution_result.final_output}")
            
            # Parse Execution Output for Status
            try:
                exec_json_str = execution_result.final_output.strip().replace("```json", "").replace("```", "")
                exec_data = json.loads(exec_json_str)
                exec_status = exec_data.get("execution_status", "unknown")
            except json.JSONDecodeError:
                logger.warning("Could not parse Execution Agent JSON. Assuming FAILURE.")
                exec_status = "failed"
            
            # FAIL-FAST: If execution failed, skip evaluation
            if exec_status != "success":
                logger.warning(f"Execution Status is {exec_status}. SKIP EVALUATION.")
                
                # Clean stale artifacts to prevent accidental evaluation on old data
                stale_file = f"{EXPERIMENT_DIR}/raw_results.json"
                if os.path.exists(stale_file):
                    os.remove(stale_file)
                    logger.info("Removed stale raw_results.json")
                
                feedback = f"Execution failed (Status: {exec_status}). Fix code errors."
                
                self.memory_service.write_run(
                    run_id=self.run_id,
                    iteration=self.current_iteration,
                    research_goal=research_goal,
                    experiment_spec=experiment_spec,
                    evaluation_verdict={"outcome_classification": "failed", "issue_type": FailureType.EXECUTION, "rationale": "Execution Agent reported failure"},
                    feedback_passed=feedback
                )
                self.emit("iteration_failed", {"iteration": self.current_iteration, "message": f"Execution failed: {exec_status}"})
                self.current_iteration += 1  # Increment after recording
                continue

            # 3. EVALUATION
            logger.info("Running Evaluation Agent...")
            self.emit("agent_started", {"agent": "Evaluate"})
            self.emit("agent_completed", {"agent": "Execute"})
            
            # Verify file exists but DO NOT load it into memory/prompt
            data_path = f"{EXPERIMENT_DIR}/raw_results.json"
            if not os.path.exists(data_path):
                # Fallback: check artifacts subdirectory (common agent availability)
                fallback_path = f"{EXPERIMENT_DIR}/artifacts/raw_results.json"
                if os.path.exists(fallback_path):
                    logger.info("Found raw_results.json in artifacts subdirectory.")
                    data_path = fallback_path
                else:
                    logger.error("raw_results.json not found despite 'success' status.")
                    feedback = "Execution reported success but 'raw_results.json' is missing."
                    
                    self.memory_service.write_run(
                        run_id=self.run_id,
                        iteration=self.current_iteration,
                        research_goal=research_goal,
                        experiment_spec=experiment_spec,
                        evaluation_verdict={"outcome_classification": "failed", "issue_type": FailureType.EXECUTION, "rationale": "Missing raw_results.json"},
                        feedback_passed=feedback
                    )
                    self.current_iteration += 1
                    continue
            
            # Use RELATIVE path for the agent tool (cleaner)
            relative_data_path = os.path.relpath(data_path, start=EXPERIMENT_DIR)
            if ".." in relative_data_path: 
                # Safety fallback if relpath is weird
                relative_data_path = data_path

            evaluation_input = {
                "experiment_specification": experiment_spec.get("experiment_specification"),
                "analysis_protocol": experiment_spec.get("statistical_analysis_plan"),
                "data_path": relative_data_path 
            }
            
            evaluation_prompt = f"""
Perform the evaluation based on the following context:
{json.dumps(evaluation_input, indent=2)}
"""
            evaluation_result = Runner.run_sync(evaluation_agent, evaluation_prompt)
            eval_output_str = evaluation_result.final_output
            
            try:
                 clean_eval = eval_output_str.strip().replace("```json", "").replace("```", "")
                 eval_json = json.loads(clean_eval)
            except json.JSONDecodeError:
                logger.error("Failed to parse Evaluation Output.")
                feedback = "Evaluation output was not valid JSON."
                
                # Record Failed Iteration (Eval Failure)
                self.memory_service.write_run(
                    run_id=self.run_id,
                    iteration=self.current_iteration,
                    research_goal=research_goal,
                    experiment_spec=experiment_spec,
                    evaluation_verdict={"outcome_classification": "failed", "issue_type": FailureType.EXECUTION, "rationale": "Invalid JSON from Eval Agent"},
                    feedback_passed=feedback
                )
                self.current_iteration += 1  # Increment after recording
                continue
            
            # --- PHASE 3: CONTROL LOGIC & MEMORY ---
            
            verdict = eval_json.get("verdict", {})
            decision = verdict.get("hypothesis_decision")
            classification = verdict.get("outcome_classification")
            issue_type = verdict.get("issue_type", "none") 
            rationale = verdict.get("rationale")
            
            # SCHEMA VALIDATION: Ensure valid values
            VALID_CLASSIFICATIONS = ["robust", "promising", "spurious", "failed"]
            VALID_ISSUE_TYPES = ["design", "execution", "data", "none"]
            
            if classification not in VALID_CLASSIFICATIONS:
                logger.warning(f"Invalid classification '{classification}'. Defaulting to 'failed'.")
                classification = "failed"
                verdict["outcome_classification"] = classification
            
            if issue_type not in VALID_ISSUE_TYPES:
                logger.warning(f"Invalid issue_type '{issue_type}'. Defaulting to 'execution'.")
                issue_type = "execution"
                verdict["issue_type"] = issue_type
            
            logger.info(f"Routing decision → classification={classification}, issue_type={issue_type}")
            
            # Attempt Crystallization via Service (Policy Enforced)
            # Attempt Crystallization via Service (Policy Enforced)
            if classification == "robust":
                result = self.memory_service.write_knowledge(
                    run_id=self.run_id,
                    experiment_spec=experiment_spec,
                    evaluation_verdict=verdict,
                    iteration_count=self.current_iteration
                )
                
                if "crystallized" in result:
                    logger.info("SUCCESS: Robust experimental evidence found and crystallized.")
                    print("\n\n🎉 ROBUST CONCLUSION REACHED!")
                    print(f"Decision: {decision}")
                    print("Final Spec:", json.dumps(experiment_spec, indent=2))
                    print("Evaluation:", json.dumps(eval_json, indent=2))
                    
                    self.emit("agent_completed", {"agent": "Evaluate"})
                    self.emit("iteration_completed", {
                        "iteration": self.current_iteration, 
                        "classification": "robust",
                        "experiment_spec": experiment_spec.get("experiment_specification", {}),
                        "evaluation": eval_json,
                        "p_value": eval_json.get("verdict", {}).get("p_value"),
                        "effect_size": eval_json.get("verdict", {}).get("effect_size")
                    })
                    
                    # Log success run and exit
                    self.memory_service.write_run(
                        run_id=self.run_id,
                        iteration=self.current_iteration,
                        research_goal=research_goal,
                        experiment_spec=experiment_spec,
                        evaluation_verdict=verdict,
                        feedback_passed="Success - Termination"
                    )
                    self.emit("run_completed", {"status": "success", "classification": "robust"})
                    return
                else:
                    logger.warning("Result was robust but failed crystallization policy (e.g. strictness). Continuing...")

            # Fallback Feedback Logic
            if issue_type == "execution" or issue_type == FailureType.EXECUTION:
                feedback = f"Execution error detected. {rationale}. Fix code/execution logic."
            
            elif issue_type == "design" or issue_type == FailureType.DESIGN:
                feedback = f"Design issue detected ({classification}). {rationale}. Refine experiment design."
            
            elif issue_type == "data" or issue_type == FailureType.DATA:
                feedback = f"Data issue detected. {rationale}. Revisit research corpus or constraints."
                
            else:
                feedback = f"Iterate. Classification: {classification}. Rationale: {rationale}."
            
            # 1. Record Iteration to Run Memory
            self.memory_service.write_run(
                run_id=self.run_id,
                iteration=self.current_iteration,
                research_goal=research_goal,
                experiment_spec=experiment_spec,
                evaluation_verdict=verdict,
                feedback_passed=feedback
            )
            
            self.emit("agent_completed", {"agent": "Evaluate"})
            self.emit("iteration_completed", {
                "iteration": self.current_iteration, 
                "classification": classification,
                "experiment_spec": experiment_spec.get("experiment_specification", {}),
                "evaluation": eval_json,
                "p_value": eval_json.get("verdict", {}).get("p_value"),
                "effect_size": eval_json.get("verdict", {}).get("effect_size")
            })
            
            # Increment iteration counter after completing full iteration
            self.current_iteration += 1

        logger.info("Max iterations reached without robust success.")
        self.emit("run_completed", {"status": "max_iterations", "iterations": self.max_iterations})

    def _cleanup_artifacts(self):
        """Removes previous experiment artifacts to enforce strict contract checking."""
        artifacts = ["dataset_used.json", "run_experiment.py", "execution.log", "raw_results.json"]
        for art in artifacts:
            path = os.path.join(EXPERIMENT_DIR, art)
            if os.path.exists(path):
                try:
                    os.remove(path)
                    logger.info(f"Cleaned up old artifact: {art}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {art}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", type=str, required=True, help="Research goal")
    parser.add_argument("--max_iters", type=int, default=3, help="Max iterations")
    args = parser.parse_args()
    
    controller = ResearchController(max_iterations=args.max_iters)
    controller.run(args.goal)
