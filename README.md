# Epsilon: Autonomous Research Engine

[![Epsilon Architecture](https://img.shields.io/badge/Status-Operational-green)](https://github.com/jhawaritvik/Epsilon)
[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://python.org)

Epsilon is an **Autonomous Research Engine** capable of conducting end-to-end scientific discovery. It takes a high-level research goal, breaks it down into hypotheses, designs methods, executes python code experiments, statistically validates results, and writes a comprehensive final report—all without human intervention.

---

## 🚀 Key Features

*   **Autonomous Agent Loop**: A multi-agent system (Design, Code, Execution, Evaluation) that iteratively solves research problems.
*   **Epistemic Integrity**: strict separation of concerns where the "Evaluation Agent" (Statistician) is isolated from the "Design Agent" (Researcher) to prevent p-hacking.
*   **Self-Correction**: The system learns from execution errors and failed hypotheses, automatically refining its approach.
*   **Report Generation**: Produces a full scientific report (`FINAL_REPORT.md` / `.html`) with embedded matplotlib visualizations and statistical analysis.

---

## 🧠 Architecture & Workflow

Epsilon uses a centralized **Controller** to orchestrate specialized agents.



### The "Epistemic Loop"
1.  **Research**: The `ResearchAgent` gathers background context and formulated hypotheses.
2.  **Design**: The `ExperimentAgent` creates a robust experimental design (variables, metrics, conditions).
3.  **Code**: The `ExecutionAgent` (and helper Code Agent) translates the design into a runnable Python script.
4.  **Execute**: The script runs in a sandboxed environment, producing data.
5.  **Evaluate**: The `EvaluationAgent` performs statistical tests (e.g., T-Test, ANOVA) to validate the hypothesis.
6.  **Iterate**: If the hypothesis fails or the code breaks, the Controller updates the `RunMemory` and retries with refined parameters.

---

## 📊 Example Output

Epsilon generates visual data analysis as part of its reporting pipeline.



The final output is a self-contained HTML report containing:
*   Executive Summary (LLM generated)
*   Methodology & Dataset details
*   Statistical Results & Discussion
*   Embedded Visualization Plots

---

## 🛠️ Getting Started

### Prerequisites
*   Python 3.9+
*   [Supabase](https://supabase.com) Account (for memory/vector store)
*   OpenAI API Key (for agent intelligence)

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/epsilon.git
    cd epsilon
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Setup**
    Copy the example env file and fill in your keys:
    ```bash
    cp .env.example .env
    ```
    
    **Required `.env` variables:**
    ```ini
    OPENAI_API_KEY=sk-...
    SUPABASE_URL=https://your-project.supabase.co
    SUPABASE_KEY=your-anon-key
    ```

### Usage

To start a new research session:

1. **Create a prompt file** with your research question:
   ```bash
   echo "Investigate the impact of learning rate on training variance in small MLPs" > research_prompt.txt
   ```

2. **Run the engine**:
   ```bash
   python main.py
   ```

   If no `research_prompt.txt` exists, you'll be prompted to enter your goal interactively.

The system will:
1.  Initialize the agents and authenticate you.
2.  Plan the experiment.
3.  Write `experiments/{run_id}/run_experiment.py`.
4.  Execute the code.
5.  Analyze results and generate `FINAL_REPORT.html` in the experiment directory.

### Docker Execution (Recommended)

For improved security and isolation, Epsilon can execute experiments inside Docker containers:

1.  **Build the Docker image**:
    ```bash
    docker build -f docker/Dockerfile.execution -t epsilon-executor:latest .
    ```

2.  **Enable Docker execution** in your `.env`:
    ```ini
    USE_DOCKER=true
    ```

3.  **Run as usual**:
    ```bash
    python main.py --goal "Your research question"
    ```

The system automatically falls back to local execution if Docker is unavailable.

---

## ⚠️ Security Notice

> [!CAUTION]
> **Code Execution Risk**: Epsilon internally generates and executes Python code. While it attempts to sandbox execution, **DO NOT** run this on sensitive production machines or with privileged access.
> *   Recommended: Run within a Docker container or VM.
> *   The `EvaluationAgent` uses `exec()` to run dynamic statistical analysis.

---

## 📄 License

MIT License. See `LICENSE` for details.
