from mcp.server.fastmcp import FastMCP

from tavily import TavilyClient
from dotenv import load_dotenv
load_dotenv()
import os

# Create the MCP server
mcp = FastMCP(
    "Tavily Search", 
    dependencies=["tavily"]
)

# Initialize the Tavily client once
tavily_api_key = os.environ.get("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY environment variable must be set")

tavily_client = TavilyClient(api_key=tavily_api_key)

@mcp.tool()
def search(
    query: str,
    search_depth: str = "advanced",
    max_results: int = 10,
    time_range: str = "year",
    include_answer: str = "advanced",
) -> dict:
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
    # Perform the search using the Tavily client
    response = tavily_client.search(
        query=query,
        search_depth=search_depth,
        max_results=max_results,
        time_range=time_range,
        include_answer=include_answer,
    )
    
    return response

@mcp.resource("search://{query}")
def search_resource(query: str) -> str:
    """
    Search the web and return results as a resource.
    This is useful for getting search results directly into context.
    
    Args:
        query: The search query to perform
    """
    # Perform a basic search
    response = tavily_client.search(
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