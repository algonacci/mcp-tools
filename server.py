import os
from dotenv import load_dotenv

load_dotenv()
from tavily import TavilyClient
from mcp.server import FastMCP

tavily_client = TavilyClient(api_key=os.getenv("TVLY_API_KEY"))
mcp = FastMCP("MCP Tools")


@mcp.tool()
def tavily_search(query: str) -> str:
    return tavily_client.search(
        query=query,
        search_depth="advanced",
        max_results=10,
        time_range="year",
        include_answer="advanced",
    )
