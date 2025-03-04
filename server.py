from mcp.server.fastmcp import FastMCP, Context
from dotenv import load_dotenv
load_dotenv()
import os
import httpx
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
import asyncio

class TavilyClient:
    """Simple client for interacting with the Tavily Search API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tavily.com/v1"
        
    async def search(
        self,
        query: str,
        search_depth: str = "advanced",
        max_results: int = 10,
        time_range: str = "year",
        include_answer: str = "advanced",
    ) -> Dict[str, Any]:
        """
        Perform a search using the Tavily API.
        
        Args:
            query: The search query
            search_depth: Either "basic" or "advanced"
            max_results: Maximum number of results to return
            time_range: Time range for search (e.g., "day", "week", "month", "year")
            include_answer: Whether to include an AI-generated answer ("basic", "advanced", or None)
            
        Returns:
            The search results from Tavily
        """
        url = f"{self.base_url}/search"
        
        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": self.api_key
        }
        
        payload = {
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
            "time_range": time_range,
            "include_answer": include_answer
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()


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
    tavily_client = TavilyClient(api_key)
    
    # Yield the context to be used by tools
    yield ServerContext(tavily_client)

# Create the MCP server
mcp = FastMCP(
    "Tavily Search", 
    dependencies=["httpx"],
    lifespan=server_lifespan
)

@mcp.tool()
async def search(
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
    await ctx.report_progress(50, 100)
    
    # Perform the search
    response = await tavily_client.search(
        query=query,
        search_depth=search_depth,
        max_results=max_results,
        time_range=time_range,
        include_answer=include_answer,
    )
    
    # Complete progress
    await ctx.report_progress(100, 100)
    ctx.info("Search complete")
    
    return response

@mcp.resource("search://{query}")
async def search_resource(query: str) -> str:
    """
    Search the web and return results as a resource.
    This is useful for getting search results directly into context.
    
    Args:
        query: The search query to perform
    """
    # Create a Tavily client directly
    # Note: This is less ideal than using the context, but works for resource handlers
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is required")
    
    tavily_client = TavilyClient(api_key)
    
    # Perform a basic search
    response = await tavily_client.search(
        query=query,
        search_depth="basic",
        max_results=5,
        include_answer="basic",
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