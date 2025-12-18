from agents import Agent, Runner, function_tool
from tavily import TavilyClient
import logging
import json
import os
from dotenv import load_dotenv

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
        
        # Save to notes store
        note_id = len(notes_store)
        notes_store[note_id] = {
            "id": note_id,
            "type": "paper",
            "source": url,
            "title": title or url.split("/")[-1],
            "content": text,
            "summary": text[:500],
            "tags": [],
            "created_by": "research_agent"
        }

        return f"PDF stored successfully. note_id={note_id}"
    except Exception as e:
        logger.error(f"Failed to read PDF {url}: {e}")
        return f"Failed to read PDF: {e}"

# Simple in-memory store for notes
notes_store = {}

@function_tool
def notes_memory_search(query: str) -> str:
    """
    Simulates searching the notes memory for a given query.
    """
    logger.info(f"Tool 'notes_memory_search' called with query: {query}")
    
    results = []

    for note in notes_store.values():
        if query.lower() in note["content"].lower():
            results.append(f"[{note['id']}] {note['title']}")

    return "\n".join(results) if results else "No relevant notes found."

@function_tool
def notes_memory_save(
    content: str,
    source: str = "",
    note_type: str = "note",
    tags: list[str] = None
) -> str:
    note_id = len(notes_store)

    notes_store[note_id] = {
        "id": note_id,
        "type": note_type,
        "source": source,
        "content": content,
        "tags": tags or [],
        "created_by": "research_agent"
    }

    return f"Note saved. note_id={note_id}"


# --- Agent Definition ---

research_instructions = """
You are the **Research Exploration Agent**. Your primary and sole goal is to **CONSTRUCT A COMPREHENSIVE RESEARCH CORPUS**. You are an information gatherer, not a designer or executor.

**Primary Objectives**:
1.  **Targeted Search**: Find the most *relevant* papers, technical reports, and documentation for the given topic. Filter out noise and tangential information.
2.  **Corpus Construction (CRITICAL)**:
    -   Use `read_pdf` to ingest the full text of primary sources that directly address the research question.
    -   Use `notes_memory_save` to capture *highly relevant* technical data: specific architecture diagrams descriptions, core formulas, key benchmark results, and implementation specifics that are essential for reproduction.
    -   Prioritize *relevant information density* over breadth. Ensure every item in the corpus has a clear link to the research objective.
3.  **No Inferencing**: Dedicated to building a lean, relevant, and searchable information base.

**Process**:
1.  **Analyze**: Identify the key keywords, core technical challenges, and relevant domains.
2.  **Search & Filter**: Use `web_search` to find high-impact resources. Critically evaluate search results for relevance before proceeding to a deep dive.
3.  **Filtered Ingest**: 
    -   `read_pdf` for identified *core* papers only.
    -   `notes_memory_save` for technical specifics found in blogs, docs, or READMEs.
4.  **Final Indexing**: 
    -   Provide a "Research Corpus Index" which is a structured list of every entity saved to memory.

**Outputs**:
-   **Research Corpus Index**: A detailed catalog of all stored papers and notes.
-   **Topic Summary**: A purely descriptive map of what information has been collected and where it can be found in the memory.
"""

research_agent = Agent(
    name="Research Exploration Agent",
    instructions=research_instructions,
    tools=[web_search, read_pdf, notes_memory_search, notes_memory_save],
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
