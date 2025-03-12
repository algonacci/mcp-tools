from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
from dotenv import load_dotenv
load_dotenv()
import os

# Configure FastMCP with dependencies
mcp = FastMCP(
    "Tavily Extract", 
    dependencies=["tavily-python"]
)

# Initialize global Tavily client
tavily_api_key = os.environ.get("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY environment variable must be set")

tavily_client = TavilyClient(api_key=tavily_api_key)

# Simple extraction tool that mirrors your test script
@mcp.tool()
def extract_url(url: str) -> dict:
    """
    Extract content from a URL using Tavily Extract API.
    
    Args:
        url: The URL to extract content from
        
    Returns:
        The extracted content
    """
    return tavily_client.extract(url)

if __name__ == "__main__":
    mcp.run()