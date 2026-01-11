from agents import Agent, Runner, function_tool
from tavily import TavilyClient
import logging
import json
import os
from dotenv import load_dotenv
# from memory.evidence_memory_writer import EvidenceMemoryWriter # Removed in favor of MemoryService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Tavily Client
tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key)

# --- Mock Tools ---
import requests
import io
from pypdf import PdfReader

# ... imports ...

# --- Mock Tools ---
# - Tavily search (primary)
# - PDF reader
# - Notes save / retrieve
# - (Later) Firecrawl for specific sites


@function_tool
def web_search(query: str) -> str:
    """
    Searching the web for a given query using Tavily API.
    Returns the search results.
    """
    logger.info(f"Tool 'web_search' called with query: {query}")
    try:
        response = tavily_client.search(query=query, search_depth="advanced")
        return json.dumps(response, indent=2)
    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return f"Search failed: {e}"

@function_tool
def read_pdf(url: str, title: str = None) -> str:
    """
    Downloads and reads a PDF from a URL.
    extracts the text and saves it to the notes memory.
    Use this ONLY for relevant PDF links found during search.
    """
    logger.info(f"Tool 'read_pdf' called with url: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        f = io.BytesIO(response.content)
        reader = PdfReader(f)
        
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        # Return success - agent should use save_evidence for critical claims
        return f"PDF successfully read from {url}. {len(text)} characters extracted. Use save_evidence to persist key findings."
    except Exception as e:
        logger.error(f"Failed to read PDF {url}: {e}")
        return f"Failed to read PDF: {e}"

# notes_store REMOVED - violates memory architecture principles
# All research data must flow through Evidence Memory for:
# - Persistence across agent boundaries
# - Auditability and provenance tracking
# - Queryability by other agents (Design Agent needs access)
# Use save_evidence and query_evidence tools instead


from memory.memory_service import MemoryService

# Initialize the Memory Service
memory_service = MemoryService()

@function_tool
def query_evidence(query: str) -> str:
    """
    Queries the existing evidence memory for claims related to the topic.
    ALWAYS call this BEFORE starting a new web search or reading a PDF to see if the information already exists.
    """
    logger.info(f"Tool 'query_evidence' called with query: {query}")
    try:
        results = memory_service.get_evidence(goal=query, limit=5)
        if not results:
            return "No existing evidence found."
        
        formatted = ["Existing Evidence:"]
        for item in results:
            formatted.append(f"- [{item.get('confidence', 'medium')}] {item.get('extracted_claim')} (Source: {item.get('source_url')})")
        return "\n".join(formatted)
    except Exception as e:
        logger.error(f"Failed to query evidence: {e}")
        return f"Error querying evidence: {e}"

@function_tool
def save_evidence(
    source_type: str,
    source_url: str,
    extracted_claim: str,
    supporting_text: str,
    confidence: str,
    paper_title:str = None,
    section: str = None
) -> str:
    """
    Saves a specific piece of evidence to the permanent research memory.
    Use this for KEY findings, claims, or specifications that must survive the research phase.
    
    params:
      source_type: 'pdf' or 'web'
      confidence: 'high', 'medium', or 'low'
      extracted_claim: The core finding in your own words.
      supporting_text: The exact quote/excerpt from the source proving the claim.
    """
    logger.info(f"Tool 'save_evidence' called for: {extracted_claim[:50]}...")
    
    # 1. Get Run ID from Environment (Passed by Controller)
    run_id = os.environ.get("CURRENT_RUN_ID")
    if not run_id:
        # Fallback if run standalone (dev mode)
        logger.warning("CURRENT_RUN_ID not set. Evidence will not be linked to a run.")
        return "Error: CURRENT_RUN_ID not set in environment. Cannot save evidence."
    
    # 2. Save via Service
    try:
        result = memory_service.write_evidence(
            run_id=run_id,
            source_type=source_type,
            source_url=source_url,
            paper_title=paper_title,
            section=section,
            extracted_claim=extracted_claim,
            supporting_text=supporting_text,
            confidence=confidence
        )
        return result
    except Exception as e:
        logger.error(f"Failed to save evidence: {e}")
        return f"Failed to save evidence: {e}"


# --- Agent Definition ---

research_instructions = """
You are the **Research Exploration Agent**. Your primary and sole goal is to **CONSTRUCT A COMPREHENSIVE RESEARCH CORPUS**. You are an information gatherer, not a designer or executor.

**Primary Objectives**:
1.  **Check Existing Knowledge**: ALWAYS call `query_evidence` first to see what is already known about the topic. Avoid duplicating work.
2.  **Targeted Search**: Find the most *relevant* papers, technical reports, and documentation for the given topic. Filter out noise.
3.  **Corpus Construction (CRITICAL)**:
    -   Use `read_pdf` to ingest the full text of primary sources.
    -   Use `save_evidence` to persist ALL important claims with supporting text.
    -   Extract atomic, verifiable facts (not interpretations).
    -   Prioritize *relevant information density* over breadth.
4.  **No Inferencing**: Dedicated to building a lean, relevant, and searchable information base.

**Process**:
1.  **Memory Check**: Query existing evidence with `query_evidence`.
2.  **Analyze**: Identify key gaps in knowledge.
3.  **Search & Filter**: Use `web_search` for relevant sources.
4.  **Ingest & Persist**: Use `read_pdf` to extract content, then `save_evidence` for critical claims.
5.  **Final Summary**: Return research corpus summary.

**Outputs**:
-   **Research Corpus Index**
-   **Research Corpus Index** (List of sources/claims)
-   **Executive Summary** (Concise synthesis of findings, < 2000 words. DO NOT dump full paper text.)
"""

research_agent = Agent(
    name="Research Exploration Agent",
    instructions=research_instructions,
    tools=[query_evidence, web_search, read_pdf, save_evidence],
)

# --- Main Execution ---

if __name__ == "__main__":
    print("Initializing Research Exploration Agent...")
    
    user_request = "I want to research about the 'attention is all you need' paper."
    
    print(f"\nUser Request: {user_request}\n")
    
    # Run the agent
    result = Runner.run_sync(research_agent, user_request)
    
    print("\n--- Agent Output ---\n")
    print(result.final_output)
