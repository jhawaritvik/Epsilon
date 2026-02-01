import json
import logging
import time
import os
import uuid
from typing import Dict, Any, Optional, Callable

from agents import Runner
from research_agent import research_agent
from experiment_agent import experiment_design_agent
from execution_agent import code_execution_agent
from evaluation_agent import evaluation_agent

# Memory Service & Types
# Memory Writers removed - replaced by MemoryService
from memory.memory_service import MemoryService
from memory.types import FailureType
from core.identity import ExecutionIdentity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Controller")

# Type for event callback
EventCallback = Callable[[str, Dict[str, Any]], None]

class ResearchController:
    def __init__(self, user_id: str, max_iterations: int = 5, event_callback: Optional[EventCallback] = None):
        self.user_id = user_id
        self.max_iterations = max_iterations
        self.history = []
        self.research_corpus = [] # List of strings/notes
        self.current_iteration = 0
        
        # Event callback for live updates (API/WebSocket integration)
        self.event_callback = event_callback
        
        # Identity & Memory Service
        self.memory_service = MemoryService()
        
        # Resource Tracking (for audit appendix, NOT scientific conclusions)
        self.run_metrics = {
            "start_time": None,
            "end_time": None,
            "phase_times": {
                "research": 0.0,
                "design": 0.0,
                "execute": 0.0,
                "evaluate": 0.0
            },
            "iterations_completed": 0,
            "final_failure_type": None  # Explicit failure mode if failed
        }
    
    def emit(self, event_type: str, data: Optional[Dict[str, Any]] = None):
        """Emit an event if callback is registered."""
        if self.event_callback:
            payload = {"type": event_type, "run_id": str(self.run_id), **(data or {})}
            try:
                logger.info(f"[EMIT] Emitting event: {event_type}")
                self.event_callback(event_type, payload)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")
        
    def run(self, research_goal: str):
        # [IDENTITY] Initialize Run & Identity Context FIRST
        self.run_id = uuid.uuid4()
        ExecutionIdentity.set_identity(self.user_id)
        
        # [METRICS] Start tracking
        self.run_metrics["start_time"] = time.time()
        
        logger.info(f"Starting Research Process for Goal: {research_goal}")
        logger.info(f"Initialized Run ID: {self.run_id}")
        
        self.emit("run_started", {"goal": research_goal})
        self.emit("log", {"message": f"[INIT] Research pipeline started for: {research_goal[:100]}..."})
        
        # Inject Run ID globally (Transient) - Identity is handled by ExecutionIdentity
        os.environ["CURRENT_RUN_ID"] = str(self.run_id)

        
        try:
            # --- PHASE 1: EXPLORATION ---
            success = self._run_research_phase(research_goal)
            if not success:
                return

            # --- PHASE 2: EXPERIMENTATION LOOP ---
            self._run_experiment_loop(research_goal)

        except Exception as e:
            logger.error(f"Critical System Error: {e}")
            import traceback
            traceback.print_exc()
            self.emit("run_error", {"error": str(e)})
            
        finally:
            # [METRICS] Finalize and save
            self.run_metrics["end_time"] = time.time()
            self.run_metrics["iterations_completed"] = self.current_iteration
            self._save_run_metrics()
            
            # Global cleanup of environment variables for this run
            if "CURRENT_RUN_ID" in os.environ:
                del os.environ["CURRENT_RUN_ID"] 
            if "CURRENT_USER_ID" in os.environ:
                del os.environ["CURRENT_USER_ID"]

    def _get_experiment_dir(self):
        """Returns the isolated directory for this run."""
        base = os.path.join(os.path.dirname(__file__), "experiments")
        return os.path.join(base, str(self.run_id))

    def _save_run_metrics(self):
        """Saves run metrics to JSON file for audit appendix."""
        exp_dir = self._get_experiment_dir()
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir, exist_ok=True)
        
        # Calculate total run time
        if self.run_metrics["start_time"] and self.run_metrics["end_time"]:
            self.run_metrics["total_time_seconds"] = round(
                self.run_metrics["end_time"] - self.run_metrics["start_time"], 2
            )
        
        metrics_path = os.path.join(exp_dir, "run_metrics.json")
        try:
            with open(metrics_path, "w") as f:
                json.dump(self.run_metrics, f, indent=2)
            logger.info(f"Run metrics saved to {metrics_path}")
        except Exception as e:
            logger.warning(f"Failed to save run metrics: {e}")

    def _cleanup_artifacts(self):
        """Removes previous experiment artifacts to enforce strict contract checking."""
        # Use isolated directory
        exp_dir = self._get_experiment_dir()
        if not os.path.exists(exp_dir):
            return
            
        artifacts = ["dataset_used.json", "run_experiment.py", "execution.log", "raw_results.json"]
        for art in artifacts:
            path = os.path.join(exp_dir, art)
            if os.path.exists(path):
                try:
                    os.remove(path)
                    logger.info(f"Cleaned up old artifact: {art}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {art}: {e}")

    def _run_research_phase(self, research_goal: str) -> bool:
        logger.info("--- PHASE 1: RESEARCH EXPLORATION ---")
        self.emit("log", {"message": "[PHASE 1] Beginning research exploration..."})
        self.emit("agent_started", {"agent": "Research"})

        self._cleanup_artifacts()
        
        max_retries = 3
        attempt = 0
        research_feedback = ""

        while attempt < max_retries:
            # Inner Retry Loop for API/Runner Stability (Rate Limits)
            inner_attempt = 0
            while inner_attempt < 3:
                try:
                    # If retrying, append feedback to the goal to nudge the agent
                    current_prompt = research_goal
                    if research_feedback:
                        current_prompt += f"\n\n[SYSTEM FEEDBACK]: {research_feedback}"
                        logger.warning(f"Retrying Research Phase (Attempt {attempt+1}/{max_retries}) with feedback.")
                    
                    # Run the agent
                    research_result = Runner.run_sync(research_agent, current_prompt, max_turns=30)
                    # Break inner loop if successful
                    break 
                except Exception as e:
                    err_str = str(e).lower()
                    if "429" in err_str or "rate limit" in err_str:
                        logger.warning(f"Rate Limit Hit. Sleeping 20s... (Attempt {inner_attempt+1}/3)")
                        time.sleep(20)
                        inner_attempt += 1
                        if inner_attempt >= 3:
                            # If we exhausted inner retries, checking if we can proceed anyway
                            evidence_count = self.memory_service.get_evidence_count(str(self.run_id))
                            if evidence_count >= 3:
                                logger.warning("Research Agent crashed on Rate Limit, BUT sufficient evidence matches persistence check. Proceeding with partial data.")
                                self.research_corpus = ["(Partial Corpus due to Rate Limit Crash)"]
                                self.emit("log", {"message": "Rate Limit Encountered. Proceeding with partial evidence."})
                                return True
                            
                            self.emit("run_error", {"error": f"Research phase failed (Rate Limit): {e}"})
                            return False
                    else:
                        # Non-retryable error
                        # Fallback check for evidence
                        evidence_count = self.memory_service.get_evidence_count(str(self.run_id))
                        if evidence_count >= 3:
                             logger.warning(f"Research Agent failed ({e}), but evidence exists. Proceeding.")
                             self.research_corpus = ["(Partial Corpus due to Agent Error)"]
                             return True
                        logger.error(f"Research Agent failed: {e}")
                        self.emit("run_error", {"error": f"Research phase failed: {e}"})
                        return False
                
            corpus_content = research_result.final_output.strip() if research_result.final_output else ""
            if not corpus_content:
                # Fallback check: Did we get evidence?
                evidence_count = self.memory_service.get_evidence_count(str(self.run_id))
                if evidence_count >= 3:
                     logger.warning("Research Agent produced no summary, but evidence exists. Proceeding.")
                     self.research_corpus = ["(Corpus Missing - Evidence Only)"]
                     return True

                logger.error("Research Agent produced no output.")
                self.emit("run_error", {"error": "No research corpus generated"})
                return False
                
            # [VERIFICATION] Ensure evidence was actually persisted to DB
            evidence_count = self.memory_service.get_evidence_count(str(self.run_id))
            logger.info(f"[RESEARCH] Evidence Persistence Check: {evidence_count} items saved.")
            
            if evidence_count > 0:
                # Success!
                self.research_corpus = [corpus_content]
                logger.info("Research Corpus Constructed.")
                self.emit("log", {"message": f"[RESEARCH] Corpus constructed ({len(corpus_content)} chars). Evidence items: {evidence_count}"})
                self.emit("agent_completed", {"agent": "Research"})
                return True
            
            # Failure case (0 items)
            attempt += 1
            if attempt < max_retries:
                msg = f"WARNING: Research Agent found 0 items. Retrying ({attempt}/{max_retries})..."
                logger.warning(msg)
                self.emit("log", {"message": msg})
                research_feedback = "CRITICAL SYSTEM ALERT: You performed research but FAILED to call `save_evidence`. The system detects 0 items in memory. You MUST call `save_evidence` for at least 3 findings immediately to fix this. Do not just summarize; persist the data."
            else:
                # Max retries reached
                msg = "CRITICAL: Research Agent reported success but 0 evidence rows persisted in DB after multiple retries."
                logger.error(msg)
                self.emit("run_error", {"error": msg})
                print(f"\n❌ {msg}\n   (Check logs for 409 Conflict or User ID errors)\n")
                return False

    def _run_experiment_loop(self, research_goal: str):
        feedback = None
        
        # Dynamic path import check - prevent usage of static EXPERIMENT_DIR in execution agent logic override
        # We rely on execution_agent checking env var now.
        exp_dir = self._get_experiment_dir()
        
        while self.current_iteration < self.max_iterations:
            logger.info(f"--- ITERATION {self.current_iteration}/{self.max_iterations} (0-indexed) ---")
            
            # 0. ADAPTIVE LOGIC & HISTORY
            # (A) Get specific past failures (Design flaws) to warn the agent
            past_failures = self.memory_service.get_past_runs(
                goal=research_goal, 
                failure_type=FailureType.DESIGN, 
                limit=3
            )
            
            adaptive_hint = ""
            if len(past_failures) >= 2:
                adaptive_hint = "\n[ADAPTIVE HINT]: You have failed on design multiple times. Please SIMPLIFY the hypothesis."

            # (B) Get Current Run History (Short-term context)
            current_run_history = self.memory_service.get_past_runs(run_id=self.run_id, limit=20)
            current_run_history.sort(key=lambda x: x.get("iteration", 0))

            # (C) Get RELEVANT Past Run History (Long-term context) - Same goal, different run
            # We fetch more generic history to see what worked or failed before
            global_run_history = self.memory_service.get_past_runs(
                goal=research_goal, 
                limit=5
            )
            # Filter out current run items to avoid duplication if any
            global_run_history = [r for r in global_run_history if str(r.get("run_id")) != str(self.run_id)]

            history_summary = ""
            if current_run_history or global_run_history:
                history_summary = "HISTORY OF ATTEMPTS:\n"
                
                if global_run_history:
                    history_summary += "--- Previous Sessions (Long-Term Memory) ---\n"
                    for entry in global_run_history:
                        status = entry.get("classification", "unknown")
                        issue = entry.get("issue_type", "unknown")
                        rationale = entry.get("evaluation_verdict", {}).get("rationale", "No rationale")
                        history_summary += f"- [Past Run] Result: {status} ({issue}). Note: {rationale}\n"

                if current_run_history:
                    history_summary += "--- Current Session (Short-Term Memory) ---\n"
                    for entry in current_run_history:
                        itr = entry.get("iteration")
                        failed_type = entry.get("issue_type", "unknown")
                        rationale = entry.get("evaluation_verdict", {}).get("rationale", "No rationale")
                        history_summary += f"- Iteration {itr}: Failed ({failed_type}). Reason: {rationale}\n"

            # (D) Get Crystallized Knowledge (Proven Facts)
            validated_knowledge = self.memory_service.get_knowledge(
                goal=research_goal, 
                limit=3
            )
            
            knowledge_context = ""
            if validated_knowledge:
                knowledge_context = "ESTABLISHED KNOWLEDGE (PROVEN FACTS):\n"
                for k in validated_knowledge:
                    q = k.get("research_question", "Question")
                    decision = k.get("decision", "Result")
                    summary = k.get("effect_summary", "")
                    knowledge_context += f"- verified: {q} -> {decision}. Findings: {summary}\n"


            # 1. DESIGN
            logger.info("Running Experiment Design Agent...")
            self.emit("log", {"message": f"[ITERATION {self.current_iteration}] Starting experiment design..."})
            self.emit("agent_started", {"agent": "Design"})
            self.emit("iteration_started", {"iteration": self.current_iteration})
            
            # [NEW] Prepare design context - preserve successful designs for refinement
            refinement_context = ""
            if hasattr(self, '_last_successful_spec') and self._last_successful_spec:
                refinement_context = f"""
--- REFINEMENT MODE ---
The previous experiment design executed SUCCESSFULLY but did NOT meet the target.
You MUST refine this design, NOT redesign from scratch.
Previous Experiment Spec (to refine):
{json.dumps(self._last_successful_spec, indent=2)}

Focus your modifications on:
1. Improving the model/technique to better achieve the target
2. Adjusting hyperparameters or methodology
3. NOT changing the core hypothesis or research question
---
"""
            
            design_prompt = f"""
User Goal: {research_goal}
Research Corpus:
{json.dumps([c[:15000] + "... (truncated)" if len(c) > 15000 else c for c in self.research_corpus])}
{history_summary}
{knowledge_context}
{refinement_context}
Feedback from Previous Iteration:
{feedback if feedback else "None (First Iteration)"}
{adaptive_hint}
Design a statistically rigorous experiment.
"""
            design_result = Runner.run_sync(experiment_design_agent, design_prompt)
            experiment_spec_str = design_result.final_output
            
            try:
                clean_spec = experiment_spec_str.strip().replace("```json", "").replace("```", "")
                experiment_spec = json.loads(clean_spec)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Experiment Spec: {e}")
                feedback = f"Critical Error: Invalid JSON output. Error: {str(e)}"
                self.memory_service.write_run(
                    run_id=self.run_id, user_id=self.user_id, iteration=self.current_iteration,
                    research_goal=research_goal, experiment_spec={}, 
                    evaluation_verdict={"outcome_classification": "failed", "issue_type": FailureType.DESIGN, "rationale": "Invalid JSON"},
                    feedback_passed=feedback
                )
                self.emit("iteration_failed", {"iteration": self.current_iteration, "message": "Design Agent produced invalid JSON"})
                self.current_iteration += 1
                continue


            # 1.5 Clean artifacts - REMOVED: Don't delete raw_results.json prematurely
            # The new execution will overwrite it if successful. Deleting early causes
            # results to be lost when subsequent iterations fail (dataset resolution, etc.)
            # Previous behavior deleted raw_results.json here, causing benchmark failures.

            # 2. EXECUTION
            logger.info("Running Code & Execution Agent...")
            self.emit("log", {"message": "[DESIGN] Experiment design complete. Starting code execution..."})
            self.emit("agent_completed", {"agent": "Design"})
            self.emit("agent_started", {"agent": "Execute"})
            
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
            
            try:
                exec_json_str = execution_result.final_output.strip().replace("```json", "").replace("```", "")
                exec_data = json.loads(exec_json_str)
                exec_status = exec_data.get("execution_status", "unknown")
            except:
                exec_status = "failed"
            
            if exec_status != "success":
                logger.warning(f"Execution Status is {exec_status}. SKIP EVALUATION.")
                feedback = f"Execution failed (Status: {exec_status}). Fix code errors."
                self.memory_service.write_run(
                    run_id=self.run_id, user_id=self.user_id, iteration=self.current_iteration,
                    research_goal=research_goal, experiment_spec=experiment_spec,
                    evaluation_verdict={"outcome_classification": "failed", "issue_type": FailureType.EXECUTION, "rationale": "Execution Agent reported failure"},
                    feedback_passed=feedback
                )
                self.emit("iteration_failed", {"iteration": self.current_iteration, "message": f"Execution failed: {exec_status}"})
                self.current_iteration += 1
                continue

            # 3. EVALUATION
            logger.info("Running Evaluation Agent...")
            self.emit("log", {"message": "[EXECUTE] Code execution successful. Running statistical evaluation..."})
            self.emit("agent_completed", {"agent": "Execute"})
            self.emit("agent_started", {"agent": "Evaluate"})
            
            data_path = f"{exp_dir}/raw_results.json"
            if not os.path.exists(data_path):
                # Check subdir artifact
                fallback = f"{exp_dir}/artifacts/raw_results.json"
                if os.path.exists(fallback):
                    data_path = fallback
                else:
                    feedback = "Execution reported success but 'raw_results.json' is missing."
                    self.memory_service.write_run(
                        run_id=self.run_id, iteration=self.current_iteration,
                        research_goal=research_goal, experiment_spec=experiment_spec,
                        evaluation_verdict={"outcome_classification": "failed", "issue_type": FailureType.EXECUTION, "rationale": "Missing raw_results.json"},
                        feedback_passed=feedback
                    )
                    self.current_iteration += 1
                    continue
            
            # Relative path for tool
            relative_data_path = os.path.relpath(data_path, start=exp_dir) # Agent sees exp_dir as root? No, agent sees CWD.
            # Using absolute path is safer for the agent tool if it doesn't assume CWD. 
            # But the agent tool instructions say "usage: python_analysis_tool".
            
            evaluation_input = {
                "experiment_specification": experiment_spec.get("experiment_specification"),
                "analysis_protocol": experiment_spec.get("statistical_analysis_plan"),
                "data_path": data_path # Pass absolute path to be safe in this complex env
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
            except:
                feedback = "Evaluation output was not valid JSON."
                self.memory_service.write_run(
                    run_id=self.run_id, iteration=self.current_iteration,
                    research_goal=research_goal, experiment_spec=experiment_spec,
                    evaluation_verdict={"outcome_classification": "failed", "issue_type": FailureType.EXECUTION, "rationale": "Invalid JSON from Eval Agent"},
                    feedback_passed=feedback
                )
                self.current_iteration += 1
                continue
            
            verdict = eval_json.get("verdict", {})
            classification = verdict.get("outcome_classification", "failed")
            
            # Crystallization Check
            if classification == "robust":
                result = self.memory_service.write_knowledge(
                    run_id=self.run_id, experiment_spec=experiment_spec,
                    evaluation_verdict=verdict, iteration_count=self.current_iteration
                )
                if "crystallized" in result:
                    logger.info("SUCCESS: CRYSTALLIZED.")
                    
                    # [NEW] Generate Final Report
                    from core.report_generator import ReportGenerator
                    report_path = ReportGenerator.generate(str(self.run_id), research_goal)
                    
                    self.emit("log", {"message": f"🎉 ROBUST CONCLUSION REACHED! Report: {report_path}"})
                    print(f"\n🎉 ROBUST CONCLUSION REACHED!\n📄 Report generated: {report_path}")
                    
                    self.emit("agent_completed", {"agent": "Evaluate"})
                    self.emit("iteration_completed", {
                        "iteration": self.current_iteration, "classification": "robust",
                        "experiment_spec": experiment_spec.get("experiment_specification"), "evaluation": eval_json
                    })
                    self.memory_service.write_run(
                        run_id=self.run_id, iteration=self.current_iteration,
                        research_goal=research_goal, experiment_spec=experiment_spec,
                        evaluation_verdict=verdict, feedback_passed="Success"
                    )
                    self.emit("run_completed", {"status": "success", "classification": "robust", "report": report_path})
                    return

            # Feedback Generation
            issue_type = verdict.get("issue_type", "none")
            rationale = verdict.get("rationale", "")
            
            # [NEW] Preserve successful design for refinement if target not met
            # This allows the Design Agent to refine rather than completely redesign
            if classification in ["promising", "failed"] and issue_type in ["target_not_met", "design", "none"]:
                # Code ran successfully but target wasn't achieved
                self._last_successful_spec = experiment_spec
                feedback = f"REFINEMENT REQUIRED: {issue_type} ({classification}). {rationale}. The code executed successfully but the target was not met. Refine the approach, do not redesign from scratch."
            else:
                # Clear the preserved spec - need fresh design for execution errors
                self._last_successful_spec = None
                feedback = f"Issue: {issue_type} ({classification}). {rationale}."
            
            self.memory_service.write_run(
                run_id=self.run_id, iteration=self.current_iteration,
                research_goal=research_goal, experiment_spec=experiment_spec,
                evaluation_verdict=verdict, feedback_passed=feedback
            )
            
            self.emit("agent_completed", {"agent": "Evaluate"})
            self.emit("iteration_completed", {
                "iteration": self.current_iteration, "classification": classification,
                "experiment_spec": experiment_spec.get("experiment_specification"), "evaluation": eval_json
            })
            
            self.current_iteration += 1

        logger.info("Max iterations reached.")
        
        # [NEW] Generate Report even slightly if failed/max_iter? 
        # Typically we only want the final report if there's a result, but useful for debugging too.
        # Let's generate it anyway so user can see what happened.
        from core.report_generator import ReportGenerator
        try:
             report_path = ReportGenerator.generate(str(self.run_id), research_goal)
             print(f"\n📄 Final Report (Max Iters): {report_path}")
        except Exception as e:
             report_path = f"Failed to generate: {e}"

        self.emit("log", {"message": f"Max iterations reached. Report: {report_path}"})
        self.emit("run_completed", {"status": "max_iterations", "report": report_path})

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", type=str, required=True, help="Research goal")
    parser.add_argument("--max_iters", type=int, default=3, help="Max iterations")
    args = parser.parse_args()
    
    # Note: CLI usage here is mocked for simplicity. Ideally pass user_id via args too.
    controller = ResearchController(user_id="00000000-0000-0000-0000-000000000001", max_iterations=args.max_iters)
    controller.run(args.goal)
