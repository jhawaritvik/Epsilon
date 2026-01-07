import json
import logging
import time
import os
import uuid
from typing import Dict, Any, Optional

from agents import Runner
from research_agent import research_agent
from experiment_agent import experiment_design_agent
from execution_agent import code_execution_agent, EXPERIMENT_DIR
from evaluation_agent import evaluation_agent

# Memory Writers
from memory.run_memory_writer import RunMemoryWriter
from memory.knowledge_memory_writer import KnowledgeMemoryWriter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Controller")

class ResearchController:
    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.history = []
        self.research_corpus = [] # List of strings/notes
        self.current_iteration = 0
        
        # Identity & Memory
        self.run_id = uuid.uuid4()
        logger.info(f"Initialized Run ID: {self.run_id}")
        
        self.run_memory = RunMemoryWriter()
        self.knowledge_memory = KnowledgeMemoryWriter()
        
    def run(self, research_goal: str):
        logger.info(f"Starting Research Process for Goal: {research_goal}")
        
        # --- PHASE 1: EXPLORATION ---
        logger.info("--- PHASE 1: RESEARCH EXPLORATION ---")
        research_result = Runner.run_sync(research_agent, research_goal)
        
        # Parse research output to act as corpus
        # The research agent returns a "Research Corpus Index" and summary. 
        # For this implementation, we'll treat the whole text output as the corpus context 
        # but optimally we would parse the specific notes_store if accessible.
        # Since agents are isolated, we pass the textual summary + index.
        self.research_corpus = [research_result.final_output] 
        logger.info("Research Corpus Constructed.")
        
        feedback = None
        
        # --- PHASE 2: EXPERIMENTATION LOOP ---
        while self.current_iteration < self.max_iterations:
            self.current_iteration += 1
            logger.info(f"--- ATTEMPTING ITERATION (Current Count: {self.current_iteration}/{self.max_iterations}) ---")
            
            # 1. DESIGN
            logger.info("Running Experiment Design Agent...")
            design_prompt = f"""
User Goal: {research_goal}

Research Corpus:
{json.dumps(self.research_corpus)}

Feedback from Previous Iteration:
{feedback if feedback else "None (First Iteration)"}

Design a statistically rigorous experiment.
"""
            design_result = Runner.run_sync(experiment_design_agent, design_prompt)
            experiment_spec_str = design_result.final_output
            
            try:
                # Attempt to parse to ensure valid JSON before passing to next agent
                clean_spec = experiment_spec_str.strip().replace("```json", "").replace("```", "")
                experiment_spec = json.loads(clean_spec)
            except json.JSONDecodeError:
                logger.error("Failed to parse Experiment Spec. Retrying loop...")
                feedback = "Critical Error: Your previous output was not valid JSON. You must output ONLY valid JSON."
                
                # Record Failed Iteration (Design Failure)
                self.run_memory.record_iteration(
                    run_id=self.run_id,
                    iteration=self.current_iteration,
                    research_goal=research_goal,
                    experiment_spec={}, 
                    evaluation_verdict={"outcome_classification": "failed", "issue_type": "design", "rationale": "Invalid JSON output from Design Agent"},
                    feedback_passed=feedback
                )
                continue

            # 2. EXECUTION
            logger.info("Running Code & Execution Agent...")
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
            # Inject Execution Mode into Environment - REMOVED
            # mode = experiment_spec.get("execution_mode", "validation")
            # os.environ["EXECUTION_MODE"] = mode
            # logger.info(f"Setting EXECUTION_MODE={mode}")

            execution_result = Runner.run_sync(code_execution_agent, execution_prompt)
            logger.info(f"Execution Output: {execution_result.final_output}")
            
            # 3. EVALUATION
            logger.info("Running Evaluation Agent...")
            
            try:
                with open(f"{EXPERIMENT_DIR}/raw_results.json", "r") as f:
                    raw_results = json.load(f)
            except FileNotFoundError:
                logger.error("raw_results.json not found. Execution likely failed.")
                feedback = "Execution Agent failed to produce 'raw_results.json'. Check code generation."
                
                # Record Failed Iteration (Execution Failure)
                self.run_memory.record_iteration(
                    run_id=self.run_id,
                    iteration=self.current_iteration,
                    research_goal=research_goal,
                    experiment_spec=experiment_spec,
                    evaluation_verdict={"outcome_classification": "failed", "issue_type": "execution", "rationale": "Missing raw_results.json"},
                    feedback_passed=feedback
                )
                continue
                
            evaluation_input = {
                "experiment_specification": experiment_spec.get("experiment_specification"),
                "analysis_protocol": experiment_spec.get("statistical_analysis_plan"),
                "observations": raw_results
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
                self.run_memory.record_iteration(
                    run_id=self.run_id,
                    iteration=self.current_iteration,
                    research_goal=research_goal,
                    experiment_spec=experiment_spec,
                    evaluation_verdict={"outcome_classification": "failed", "issue_type": "execution", "rationale": "Invalid JSON from Eval Agent"},
                    feedback_passed=feedback
                )
                continue
            
            # --- PHASE 3: CONTROL LOGIC & MEMORY ---
            
            verdict = eval_json.get("verdict", {})
            decision = verdict.get("hypothesis_decision")
            classification = verdict.get("outcome_classification")
            issue_type = verdict.get("issue_type", "none") 
            rationale = verdict.get("rationale")
            
            logger.info(f"Routing decision → classification={classification}, issue_type={issue_type}")
            
            if classification == "robust" and decision == "Reject H0":
                logger.info("SUCCESS: Robust experimental evidence found.")
                print("\n\n🎉 DISCOVERY MADE!")
                print("Final Spec:", json.dumps(experiment_spec, indent=2))
                print("Evaluation:", json.dumps(eval_json, indent=2))
                
                # Record Success Iteration (No feedback needed as we stop)
                self.run_memory.record_iteration(
                    run_id=self.run_id,
                    iteration=self.current_iteration,
                    research_goal=research_goal,
                    experiment_spec=experiment_spec,
                    evaluation_verdict=verdict,
                    feedback_passed="Success - Termination"
                )

                # 2. Crystallize Knowledge
                self.knowledge_memory.record_knowledge(
                    run_id=self.run_id,
                    experiment_spec=experiment_spec,
                    evaluation_verdict=verdict
                )
                return

            elif issue_type == "execution":
                feedback = f"Execution error detected. {rationale}. Fix code/execution logic."
            
            elif issue_type == "design":
                feedback = f"Design issue detected ({classification}). {rationale}. Refine experiment design."
            
            elif issue_type == "data":
                feedback = f"Data issue detected. {rationale}. Revisit research corpus or constraints."
                
            else:
                feedback = f"Iterate. Classification: {classification}. Rationale: {rationale}."
            
            # 1. Record Iteration to Run Memory (Now with feedback)
            self.run_memory.record_iteration(
                run_id=self.run_id,
                iteration=self.current_iteration,
                research_goal=research_goal,
                experiment_spec=experiment_spec,
                evaluation_verdict=verdict,
                feedback_passed=feedback
            )

        logger.info("Max iterations reached without robust success.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", type=str, required=True, help="Research goal")
    parser.add_argument("--max_iters", type=int, default=3, help="Max iterations")
    args = parser.parse_args()
    
    controller = ResearchController(max_iterations=args.max_iters)
    controller.run(args.goal)
