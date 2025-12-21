import os
import sys
from controller import ResearchController
import logging

# Configure plain logging for the user interface
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Main")

PROMPT_FILE = "research_prompt.txt"

def main():
    print("==================================================")
    print("   AUTONOMOUS RESEARCH ENGINE - USER CONTROLLER   ")
    print("==================================================")
    
    research_goal = ""
    
    # 1. Try reading from prompt file
    if os.path.exists(PROMPT_FILE):
        with open(PROMPT_FILE, "r") as f:
            content = f.read().strip()
            # Ignore placeholder or empty
            if content and not content.startswith("Enter your research goal here"):
                research_goal = content
                print(f"\n[INFO] Loaded research goal from {PROMPT_FILE}")
    
    # 2. Fallback to interactive input
    if not research_goal:
        print(f"\n[INFO] No valid goal found in {PROMPT_FILE}.")
        print("Please enter your research goal below:")
        try:
            research_goal = input("> ").strip()
        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit(0)
            
    if not research_goal:
        print("Error: Research goal cannot be empty.")
        sys.exit(1)
        
    print(f"\nStarting Research on: '{research_goal}'")
    print("--------------------------------------------------\n")
    
    # 3. Initialize and Run Controller
    try:
        controller = ResearchController(max_iterations=5)
        controller.run(research_goal)
        print("\n==================================================")
        print("              RESEARCH COMPLETED                  ")
        print("==================================================")
    except Exception as e:
        logger.error(f"Critical System Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
