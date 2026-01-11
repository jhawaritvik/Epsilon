# Epsilon: Autonomous Research Engine

## Memory Architecture

The Epsilon system uses a structured memory architecture to ensure agents operate with state, history, and scientific rigor.

### Memory Contract

| Memory Type | Purpose | Who Writes | Who Reads |
| :--- | :--- | :--- | :--- |
| **Evidence Memory** | Raw claims + sources (papers, web) | **Research Agent** | **Design Agent**, **Research Agent** (for dedup) |
| **Run Memory** | Audit trail of iterations & failures | **Controller** | **Controller**, **Design Agent** (to avoid repeats) |
| **Knowledge Memory** | Validated facts & robust discoveries | **Controller** (only upon Robust Success) | **All Agents** |

### Agent Responsibilities

- **Research Agent**: Gathers information. Must **read** existing evidence before searching to avoid duplication. Writes atomic claims to Evidence Memory.
- **Experiment Design Agent**: Designs experiments. **Must read** Evidence, Knowledge, and Run memory (past failures) to design scientifically valid and novel experiments.
- **Execution Agent**: Stateless executor. Reads spec, executes code, writes `raw_results.json`. **No direct memory access.**
- **Evaluation Agent**: Epistemically isolated statistician. Reads `raw_results` and spec. **Must NOT read** Knowledge or Evidence memory to remain unbiased.
- **Controller**: The orchestrator and epistemic authority. Routes data, manages the loop, and is the **only** entity allowed to write to Knowledge Memory ("Crystallization").

## Setup

1.  Ensure Supabase credentials are in `.env`.
2.  Run `python main.py --goal "Your Research Goal"`
