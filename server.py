from mcp.server.fastmcp import FastMCP, Context
import os
from dotenv import load_dotenv
load_dotenv()
from tavily import TavilyClient
from typing import Dict, Any
from contextlib import asynccontextmanager

# Create a server context class to hold the Tavily client
class ServerContext:
    def __init__(self, tavily_client: TavilyClient):
        self.tavily_client = tavily_client

# Define the lifespan for the MCP server
@asynccontextmanager
async def server_lifespan(server: FastMCP):
    """Initialize the Tavily client on server startup."""
    # Get API key from environment
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is required")
    
    # Create Tavily client
    tavily_client = TavilyClient(api_key=api_key)
    
    # Yield the context to be used by tools
    yield ServerContext(tavily_client)

# Create the MCP server
mcp = FastMCP(
    "Tavily Search", 
    dependencies=["tavily"],
    lifespan=server_lifespan
)

@mcp.tool()
def search(
    query: str,
    ctx: Context,
    search_depth: str = "advanced",
    max_results: int = 10,
    time_range: str = "year",
    include_answer: str = "advanced",
) -> Dict[str, Any]:
    """
    Search the web using Tavily's search API.
    
    Args:
        query: The search query to perform
        search_depth: Either "basic" or "advanced"
        max_results: Maximum number of results to return (1-10)
        time_range: Time range for search ("day", "week", "month", "year")
        include_answer: Whether to include an AI-generated answer ("basic", "advanced", or None)
        
    Returns:
        Search results including links, snippets, and potentially an AI answer
    """
    # Get the Tavily client from context
    server_ctx = ctx.request_context.lifespan_context
    tavily_client = server_ctx.tavily_client
    
    # Report progress
    ctx.info(f"Searching for: {query}")
    ctx.report_progress(50, 100)
    
    # Perform the search
    response = tavily_client.search(
        query=query,
        search_depth=search_depth,
        max_results=max_results,
        time_range=time_range,
        include_answer=include_answer,
    )
    
    # Complete progress
    ctx.report_progress(100, 100)
    ctx.info("Search complete")
    
    return response

# Global Tavily client for resource handler
# Note: We need this because resource handlers can't access the context with URI parameters
_tavily_client = None

def get_tavily_client():
    """Get or create a global Tavily client instance"""
    global _tavily_client
    if _tavily_client is None:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY environment variable is required")
        _tavily_client = TavilyClient(api_key=api_key)
    return _tavily_client

@mcp.resource("search://{query}")
def search_resource(query: str) -> str:
    """
    Search the web and return results as a resource.
    This is useful for getting search results directly into context.
    
    Args:
        query: The search query to perform
    """
    # Get the Tavily client
    tavily_client = get_tavily_client()
    
    # Perform a basic search
    response = tavily_client.search(
        query=query,
        search_depth="advanced",
        max_results=10,
        include_answer="advanced",
    )
    
    # Format the results as readable text
    result = f"# Search Results for: {query}\n\n"
    
    # Include the answer if available
    if "answer" in response and response["answer"]:
        result += f"## Answer\n{response['answer']}\n\n"
    
    # Include search results
    result += "## Sources\n"
    for i, item in enumerate(response.get("results", []), 1):
        result += f"{i}. [{item['title']}]({item['url']})\n"
        result += f"   {item['content'][:150]}...\n\n"
    
    return result

@mcp.prompt()
def search_prompt(query: str = "") -> str:
    """
    Create a prompt for searching the web.
    
    Args:
        query: Optional initial search query
    """
    if query:
        return f"""I'd like to search for information about: {query}

Please use the Tavily search tool to find relevant information and summarize what you find.
"""
    else:
        return """I'd like to search for information on the web.

What would you like to search for? Once you tell me, I'll use the Tavily search tool to find relevant information and summarize the results for you.
"""

# Allow direct execution of the server
if __name__ == "__main__":
    mcp.run()