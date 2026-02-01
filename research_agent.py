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

@function_tool
def read_webpage(url: str) -> str:
    """
    Reads the text content of a web page (HTML).
    Use this for blog posts, articles, and documentation that are NOT PDFs.
    """
    logger.info(f"Tool 'read_webpage' called with url: {url}")
    try:
        # Standard headers to avoid 403 blocks from some sites
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Simple text extraction (heuristic)
        # In a real scraper we'd use BeautifulSoup, but let's keep dependencies low if possible.
        # However, for research grade, we should probably install bs4 if needed, but let's try naive first?
        # No, naive regex is bad for HTML. Let's assume bs4 is available or install it.
        # Actually, let's use the 'install_package' philosophy or just rely on simple string stripping if bs4 is missing.
        
        # Let's import bs4 inside here to be safe, or fall back
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            # Kill javascript and style elements
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text()
            # break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
            # break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)
        except ImportError:
            return "Error: BeautifulSoup (bs4) not installed. Please install it to read webpages."
            
        return f"Webpage read successfully from {url}. {len(text)} characters extracted. Use save_evidence to persist key findings."

    except Exception as e:
        logger.error(f"Failed to read webpage {url}: {e}")
        return f"Failed to read webpage: {e}"

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
        # Identity is handled implicitly by MemoryService via ExecutionIdentity
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
    
    # 2. Save via Service (Identity handled implicitly)
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

**Critical Instruction**:
Reading a PDF or Webpage is **EPHEMERAL**. The system **DOES NOT** remember what you read unless you explicitly call `save_evidence`.
You **MUST** call `save_evidence` for every important finding. If you finish without saving at least 3 items, the system will **CRASH** and you will be penalized.

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
2.  **Search & Ingest**:
    - `web_search(query)`: Search for sources.
    - `read_pdf(url)` / `read_webpage(url)`: Read content. **(THIS DOES NOT SAVE ANYTHING)**.
    - **IMMEDIATELY** after reading, call `save_evidence(claim, source, ...)` for key facts.
3.  **Persistence**: You MUST save key findings using `save_evidence`.
    - If you find 0 relevant evidence, the pipeline will HALT.
    - Aim for at least 3 distinct pieces of evidence.
4.  **Adaptability**:
    - If `read_pdf` fails (e.g. "Stream ended unexpectedly" or 403), do **NOT** retry the same URL.
    - If the link is likely a blog or article (not ending in .pdf), use `read_webpage` instead.
    - Try alternative sources.

**Outputs**:
-   **Research Corpus Index** (List of sources/claims)
-   **Executive Summary** (Concise synthesis of findings, < 2000 words. DO NOT dump full paper text.)
"""

research_agent = Agent(
    name="Research Exploration Agent",
    model=os.getenv("MODEL_NAME", "gpt-5.2"),
    instructions=research_instructions,
    tools=[web_search, read_pdf, read_webpage, query_evidence, save_evidence],
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
