import os
import sys
from controller import ResearchController
import logging

# Configure plain logging for the user interface
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Main")

PROMPT_FILE = "research_prompt.txt"

def validate_environment():
    """
    Validates required environment variables are set.
    Exits with clear error message if any are missing.
    """
    required = {
        "OPENAI_API_KEY": "OpenAI API access for all agents",
        "TAVILY_API_KEY": "Web search functionality (Research Agent)",
        "SUPABASE_URL": "Memory persistence (optional but recommended)",
        "SUPABASE_KEY": "Memory persistence (optional but recommended)"
    }
    
    missing = []
    warnings = []
    
    for key, purpose in required.items():
        if not os.getenv(key):
            # Supabase is optional - warn but don't block
            if key.startswith("SUPABASE"):
                warnings.append(f"  ⚠️  {key} (for {purpose})")
            else:
                missing.append(f"  ❌ {key} (for {purpose})")
    
    if warnings:
        print("\n⚠️  WARNING: Optional environment variables not set:")
        print("\n".join(warnings))
        print("Memory persistence will be DISABLED. Set these in .env to enable.\n")
    
    if missing:
        print("\n" + "="*60)
        print("❌ ERROR: Missing REQUIRED environment variables:")
        print("="*60)
        print("\n".join(missing))
        print("\nPlease configure your .env file. See .env.example for template.")
        print("="*60 + "\n")
        sys.exit(1)


def main():
    # Validate environment before starting
    validate_environment()
    
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
        from memory.user_manager import UserManager
        import getpass
        
        print("\n--- LOGIN TO RESEARCH ENGINE ---")
        user_email = input("Email: ").strip()
        if not user_email:
            print("Email is required.")
            sys.exit(1)
            
        user_password = getpass.getpass("Password: ").strip()
        
        try:
            print(f"Logging in as {user_email}...")
            user_id = UserManager.login(user_email, user_password)
            print(f"✅ Success. Session ID: {user_id}")
        except Exception as e:
            print(f"❌ Login Failed: {e}")
            sys.exit(1)
        
        controller = ResearchController(user_id=user_id, max_iterations=5)
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
