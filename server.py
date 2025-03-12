from mcp.server.fastmcp import FastMCP, Context
from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass
from contextlib import asynccontextmanager
import os
import re
import PyPDF2
import gnews
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from tavily import TavilyClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define server contexts
@dataclass
class ServerContext:
    gnews_client: gnews.GNews
    tavily_client: TavilyClient

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[ServerContext]:
    """Initialize clients on startup"""
    # Check for Tavily API key
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY environment variable must be set")
    
    # Initialize clients
    default_gnews = gnews.GNews()
    tavily_client = TavilyClient(api_key=tavily_api_key)
    
    try:
        yield ServerContext(
            gnews_client=default_gnews,
            tavily_client=tavily_client
        )
    finally:
        pass  # No cleanup needed

# Configure FastMCP with dependencies and lifespan
mcp = FastMCP(
    "Multi-Tool Server", 
    dependencies=[
        "gnews", 
        "tavily-python", 
        "PyPDF2>=3.0.0",
        "python-dotenv",
        "sqlalchemy",
        "pandas",
        "pymysql",
        "psycopg2-binary"
    ],
    lifespan=app_lifespan
)

#
# SQL Database functionality
#

# Dictionary to store database connections for reuse
active_connections = {}

@mcp.tool()
def connect_database(
    connection_string: str,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Connect to a SQL database using SQLAlchemy.
    Automatically detects MySQL or PostgreSQL databases.
    
    Args:
        connection_string: Database connection string
            - MySQL format: "mysql+pymysql://user:password@host:port/database"
            - PostgreSQL format: "postgresql+psycopg2://user:password@host:port/database"
            
    Returns:
        Dictionary with connection status, database type, and available tables
    """
    try:
        # Log connection attempt (masking password for security)
        masked_connection = mask_password(connection_string)
        if ctx:
            ctx.info(f"Attempting to connect to database: {masked_connection}")
        
        # Check if connection string has the right format
        if not (connection_string.startswith('mysql+') or 
                connection_string.startswith('postgresql+') or
                connection_string.startswith('mysql://') or 
                connection_string.startswith('postgresql://')):
            
            # Try to auto-correct the connection string if possible
            if "mysql" in connection_string.lower():
                if not connection_string.startswith('mysql+pymysql://'):
                    connection_string = connection_string.replace('mysql://', 'mysql+pymysql://')
                    if not connection_string.startswith('mysql+'):
                        connection_string = 'mysql+pymysql://' + connection_string
            elif "postgre" in connection_string.lower():
                if not connection_string.startswith('postgresql+psycopg2://'):
                    connection_string = connection_string.replace('postgresql://', 'postgresql+psycopg2://')
                    if not connection_string.startswith('postgresql+'):
                        connection_string = 'postgresql+psycopg2://' + connection_string
            else:
                return {
                    "success": False,
                    "error": "Unsupported database type. Please use MySQL or PostgreSQL connection strings."
                }
        
        # Create engine and connect
        engine = create_engine(connection_string)
        connection = engine.connect()
        
        # Determine database type
        db_type = "MySQL" if "mysql" in connection_string.lower() else "PostgreSQL"
        
        # Get database inspector
        inspector = inspect(engine)
        
        # Get all tables
        tables = inspector.get_table_names()
        
        # Get schema information for each table
        schema_info = {}
        for table in tables:
            columns = inspector.get_columns(table)
            schema_info[table] = [
                {"name": col["name"], "type": str(col["type"])} 
                for col in columns
            ]
        
        # Store connection for future use
        conn_id = masked_connection
        active_connections[conn_id] = {
            "engine": engine,
            "connection": connection,
            "type": db_type,
            "tables": tables,
            "schema": schema_info
        }
        
        return {
            "success": True,
            "connection_id": conn_id,
            "database_type": db_type,
            "tables": tables,
            "schema": schema_info
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to connect: {str(e)}"
        }

@mcp.tool()
def execute_query(
    connection_id: str,
    query: str,
    params: Optional[Dict[str, Any]] = None,
    limit: int = 100,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Execute a SQL query on a previously connected database.
    
    Args:
        connection_id: Connection identifier returned from connect_database
        query: SQL query to execute
        params: Optional parameters for the query
        limit: Maximum number of rows to return (for SELECT queries)
        
    Returns:
        Dictionary with query results or affected row count
    """
    if connection_id not in active_connections:
        return {
            "success": False,
            "error": "Invalid connection ID. Please connect to the database first."
        }
    
    connection_info = active_connections[connection_id]
    connection = connection_info["connection"]
    
    try:
        if ctx:
            ctx.info(f"Executing query: {query[:100]}...")
        
        # Check if it's a SELECT query
        is_select = query.strip().lower().startswith("select")
        
        if is_select:
            # For SELECT queries, use pandas to get results as a DataFrame
            if params:
                df = pd.read_sql(text(query), connection, params=params)
            else:
                df = pd.read_sql(text(query), connection)
            
            # Limit the number of rows
            if limit > 0:
                df = df.head(limit)
            
            # Convert to dictionary format
            result = {
                "success": True,
                "is_select": True,
                "rows": df.to_dict(orient="records"),
                "columns": df.columns.tolist(),
                "row_count": len(df)
            }
        else:
            # For non-SELECT queries, execute directly
            if params:
                result_proxy = connection.execute(text(query), params)
            else:
                result_proxy = connection.execute(text(query))
            
            result = {
                "success": True,
                "is_select": False,
                "affected_rows": result_proxy.rowcount
            }
        
        return result
    except Exception as e:
        return {
            "success": False,
            "error": f"Query execution failed: {str(e)}"
        }

@mcp.tool()
def list_tables(
    connection_id: str,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    List all tables in the connected database.
    
    Args:
        connection_id: Connection identifier returned from connect_database
        
    Returns:
        Dictionary with list of tables and their schema information
    """
    if connection_id not in active_connections:
        return {
            "success": False,
            "error": "Invalid connection ID. Please connect to the database first."
        }
    
    connection_info = active_connections[connection_id]
    
    return {
        "success": True,
        "database_type": connection_info["type"],
        "tables": connection_info["tables"],
        "schema": connection_info["schema"]
    }

@mcp.tool()
def describe_table(
    connection_id: str,
    table_name: str,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Get detailed schema information for a specific table.
    
    Args:
        connection_id: Connection identifier returned from connect_database
        table_name: Name of the table to describe
        
    Returns:
        Dictionary with table schema information
    """
    if connection_id not in active_connections:
        return {
            "success": False,
            "error": "Invalid connection ID. Please connect to the database first."
        }
    
    connection_info = active_connections[connection_id]
    engine = connection_info["engine"]
    
    try:
        # Get database inspector
        inspector = inspect(engine)
        
        # Get column information
        columns = inspector.get_columns(table_name)
        
        # Get primary key information
        pk_columns = inspector.get_pk_constraint(table_name).get('constrained_columns', [])
        
        # Get foreign key information
        foreign_keys = inspector.get_foreign_keys(table_name)
        
        # Get index information
        indexes = inspector.get_indexes(table_name)
        
        # Format column information
        column_info = []
        for col in columns:
            column_info.append({
                "name": col["name"],
                "type": str(col["type"]),
                "nullable": col.get("nullable", True),
                "default": str(col.get("default", "None")),
                "is_primary_key": col["name"] in pk_columns
            })
        
        # Execute a sample query to get row count
        query = text(f"SELECT COUNT(*) as count FROM {table_name}")
        result = connection_info["connection"].execute(query).fetchone()
        row_count = result[0] if result else 0
        
        return {
            "success": True,
            "table_name": table_name,
            "columns": column_info,
            "primary_keys": pk_columns,
            "foreign_keys": foreign_keys,
            "indexes": indexes,
            "row_count": row_count
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to describe table: {str(e)}"
        }

@mcp.tool()
def disconnect_database(
    connection_id: str,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Close a database connection.
    
    Args:
        connection_id: Connection identifier returned from connect_database
        
    Returns:
        Dictionary with disconnection status
    """
    if connection_id not in active_connections:
        return {
            "success": False,
            "error": "Invalid connection ID. No active connection to close."
        }
    
    try:
        connection_info = active_connections[connection_id]
        connection = connection_info["connection"]
        
        # Close the connection
        connection.close()
        
        # Remove from active connections
        del active_connections[connection_id]
        
        return {
            "success": True,
            "message": f"Successfully disconnected from {connection_info['type']} database."
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to disconnect: {str(e)}"
        }

@mcp.resource("sql://schema/{connection_id}")
def schema_resource(connection_id: str) -> str:
    """
    Get the database schema as a formatted resource.
    
    Args:
        connection_id: Connection identifier returned from connect_database
    """
    if connection_id not in active_connections:
        return "# Error\n\nInvalid connection ID. Please connect to the database first."
    
    connection_info = active_connections[connection_id]
    
    # Format as markdown
    result = f"# {connection_info['type']} Database Schema\n\n"
    result += f"## Tables ({len(connection_info['tables'])})\n\n"
    
    for table_name in connection_info['tables']:
        result += f"### {table_name}\n\n"
        result += "| Column | Type | Description |\n"
        result += "|--------|------|-------------|\n"
        
        for column in connection_info['schema'][table_name]:
            result += f"| {column['name']} | {column['type']} | |\n"
        
        result += "\n"
    
    return result

@mcp.resource("sql://query/{connection_id}/{query}")
def query_resource(connection_id: str, query: str) -> str:
    """
    Execute a SQL query and return the results as a formatted resource.
    
    Args:
        connection_id: Connection identifier returned from connect_database
        query: SQL query to execute (URL-encoded)
    """
    if connection_id not in active_connections:
        return "# Error\n\nInvalid connection ID. Please connect to the database first."
    
    # URL-decode the query
    query = query.replace('%20', ' ').replace('%22', '"').replace('%27', "'")
    
    # Execute the query
    result = execute_query(connection_id, query, limit=20)
    
    if not result["success"]:
        return f"# Error Executing Query\n\n{result['error']}"
    
    # Format as markdown
    output = "# SQL Query Results\n\n"
    output += f"```sql\n{query}\n```\n\n"
    
    if result.get("is_select", False):
        # Format SELECT results as a table
        if result["row_count"] == 0:
            output += "No results returned.\n"
        else:
            # Create header row
            output += "| " + " | ".join(result["columns"]) + " |\n"
            output += "|" + "---|" * len(result["columns"]) + "\n"
            
            # Add data rows
            for row in result["rows"]:
                output += "| " + " | ".join(str(row.get(col, "")) for col in result["columns"]) + " |\n"
            
            if result["row_count"] >= 20:
                output += "\n*Query limited to 20 rows. Use the execute_query tool for more results.*\n"
    else:
        # Format non-SELECT results
        output += f"**Affected rows:** {result['affected_rows']}\n"
    
    return output

#
# GNews functionality
#

# Helper function to create a GNews client with specific parameters
def create_gnews_client(
    language: str = "en",
    country: str = "US",
    max_results: int = 10,
    period: str = None,
    proxy: str = None,
    exclude_websites: List[str] = None
) -> gnews.GNews:
    """
    Create a GNews client with the specified parameters.
    """
    return gnews.GNews(
        language=language,
        country=country,
        max_results=max_results,
        period=period,
        proxy=proxy,
        exclude_websites=exclude_websites
    )

@mcp.tool()
async def search_news(
    query: str,
    language: str = "en",
    country: str = "US",
    max_results: int = 10,
    period: str = None,
    proxy: str = None,
    exclude_websites: List[str] = None,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Search for news articles using GNews.
    
    Args:
        query: Search keywords or topic
        language: Language code (e.g., 'en'=English, 'id'=Indonesian, 'es'=Spanish, 'fr'=French)
        country: Country code (e.g., 'US'=USA, 'ID'=Indonesia, 'UK'=United Kingdom, 'CA'=Canada)
        max_results: Maximum number of results to return (1-100)
        period: Time period (None for all time, 'd' for day, 'h' for hour, 'm' for month)
        proxy: Optional proxy server to use for requests
        exclude_websites: Optional list of websites to exclude from results
        
    Returns:
        List of news articles matching the search criteria
    """
    # Create a new client with the specified parameters
    gn = create_gnews_client(
        language=language,
        country=country,
        max_results=max_results,
        period=period,
        proxy=proxy,
        exclude_websites=exclude_websites
    )
    
    # Report progress
    if ctx:
        ctx.info(f"Searching for news about: {query} in {language} ({country})")
        await ctx.report_progress(50, 100)
    
    try:
        # Get news articles
        articles = gn.get_news(query)
        
        # Format the results
        results = []
        for article in articles:
            formatted_article = {
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "publisher": article.get("publisher", {}).get("title", ""),
                "published_date": article.get("published date", ""),
                "description": article.get("description", "")
            }
            results.append(formatted_article)
        
        # Complete progress
        if ctx:
            await ctx.report_progress(100, 100)
            ctx.info(f"Found {len(results)} news articles")
        
        return {
            "success": True,
            "query": query,
            "language": language,
            "country": country,
            "period": period,
            "articles": results
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error searching for news: {str(e)}"
        }

@mcp.tool()
async def get_top_news(
    language: str = "en",
    country: str = "US",
    max_results: int = 10,
    proxy: str = None,
    exclude_websites: List[str] = None,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Get top headline news.
    
    Args:
        language: Language code (e.g., 'en'=English, 'id'=Indonesian, 'es'=Spanish, 'fr'=French)
        country: Country code (e.g., 'US'=USA, 'ID'=Indonesia, 'UK'=United Kingdom, 'CA'=Canada)
        max_results: Maximum number of results to return (1-100)
        proxy: Optional proxy server to use for requests
        exclude_websites: Optional list of websites to exclude from results
        
    Returns:
        List of top headline news articles
    """
    # Create a new client with the specified parameters
    gn = create_gnews_client(
        language=language,
        country=country,
        max_results=max_results,
        proxy=proxy,
        exclude_websites=exclude_websites
    )
    
    # Report progress
    if ctx:
        ctx.info(f"Fetching top headlines for {country} in {language}")
        await ctx.report_progress(50, 100)
    
    try:
        # Get top news articles
        articles = gn.get_top_news()
        
        # Format the results
        results = []
        for article in articles:
            formatted_article = {
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "publisher": article.get("publisher", {}).get("title", ""),
                "published_date": article.get("published date", ""),
                "description": article.get("description", "")
            }
            results.append(formatted_article)
        
        # Complete progress
        if ctx:
            await ctx.report_progress(100, 100)
            ctx.info(f"Found {len(results)} top news articles")
        
        return {
            "success": True,
            "language": language,
            "country": country,
            "articles": results
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error fetching top news: {str(e)}"
        }

@mcp.tool()
async def get_topic_news(
    topic: str,
    language: str = "en",
    country: str = "US",
    max_results: int = 10,
    proxy: str = None,
    exclude_websites: List[str] = None,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Get news for a specific topic category.
    
    Args:
        topic: News category (e.g., 'world', 'business', 'technology', 'sports', 'entertainment', 'science', 'health')
        language: Language code (e.g., 'en'=English, 'id'=Indonesian, 'es'=Spanish, 'fr'=French)
        country: Country code (e.g., 'US'=USA, 'ID'=Indonesia, 'UK'=United Kingdom, 'CA'=Canada)
        max_results: Maximum number of results to return (1-100)
        proxy: Optional proxy server to use for requests
        exclude_websites: Optional list of websites to exclude from results
        
    Returns:
        List of news articles for the specified topic
    """
    # Create a new client with the specified parameters
    gn = create_gnews_client(
        language=language,
        country=country,
        max_results=max_results,
        proxy=proxy,
        exclude_websites=exclude_websites
    )
    
    # Report progress
    if ctx:
        ctx.info(f"Fetching {topic} news for {country} in {language}")
        await ctx.report_progress(50, 100)
    
    try:
        # Get topic news articles
        articles = gn.get_news_by_topic(topic)
        
        # Format the results
        results = []
        for article in articles:
            formatted_article = {
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "publisher": article.get("publisher", {}).get("title", ""),
                "published_date": article.get("published date", ""),
                "description": article.get("description", "")
            }
            results.append(formatted_article)
        
        # Complete progress
        if ctx:
            await ctx.report_progress(100, 100)
            ctx.info(f"Found {len(results)} {topic} news articles")
        
        return {
            "success": True,
            "topic": topic,
            "language": language,
            "country": country,
            "articles": results
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error fetching {topic} news: {str(e)}"
        }

@mcp.resource("news://{query}/{language}/{country}")
async def news_resource_localized(query: str, language: str, country: str) -> str:
    """
    Get news about a specific query in the specified language and country.
    
    Args:
        query: Search keywords or topic
        language: Language code (e.g., 'en', 'id', 'es', 'fr')
        country: Country code (e.g., 'US', 'ID', 'UK', 'CA')
    """
    # Initialize GNews client with specified parameters
    gn = gnews.GNews(language=language, country=country, max_results=5)
    
    try:
        # Get news articles
        articles = gn.get_news(query)
        
        # Format as markdown
        result = f"# News Results for: {query}\n"
        result += f"## Language: {language} | Country: {country}\n\n"
        
        for i, article in enumerate(articles, 1):
            title = article.get("title", "No title")
            url = article.get("url", "")
            publisher = article.get("publisher", {}).get("title", "Unknown")
            date = article.get("published date", "")
            description = article.get("description", "No description available")
            
            result += f"### {i}. {title}\n"
            result += f"**Source:** {publisher} | **Date:** {date}\n\n"
            result += f"{description}\n\n"
            result += f"[Read more]({url})\n\n"
            result += "---\n\n"
        
        return result
    except Exception as e:
        return f"# Error Fetching News\n\nThere was a problem retrieving news articles for '{query}' in {language}/{country}: {str(e)}"

@mcp.resource("news://{query}")
async def news_resource(query: str) -> str:
    """
    Get news about a specific query in English (US).
    
    Args:
        query: Search keywords or topic
    """
    # Use the localized resource with default values
    return await news_resource_localized(query, "en", "US")

@mcp.resource("news://top/{language}/{country}")
async def top_news_resource_localized(language: str, country: str) -> str:
    """
    Get top headline news for the specified language and country.
    
    Args:
        language: Language code (e.g., 'en', 'id', 'es', 'fr')
        country: Country code (e.g., 'US', 'ID', 'UK', 'CA')
    """
    # Initialize GNews client with specified parameters
    gn = gnews.GNews(language=language, country=country, max_results=5)
    
    try:
        # Get top news articles
        articles = gn.get_top_news()
        
        # Format as markdown
        result = "# Top News Headlines\n"
        result += f"## Language: {language} | Country: {country}\n\n"
        
        for i, article in enumerate(articles, 1):
            title = article.get("title", "No title")
            url = article.get("url", "")
            publisher = article.get("publisher", {}).get("title", "Unknown")
            date = article.get("published date", "")
            description = article.get("description", "No description available")
            
            result += f"### {i}. {title}\n"
            result += f"**Source:** {publisher} | **Date:** {date}\n\n"
            result += f"{description}\n\n"
            result += f"[Read more]({url})\n\n"
            result += "---\n\n"
        
        return result
    except Exception as e:
        return f"# Error Fetching Top News\n\nThere was a problem retrieving top news articles for {language}/{country}: {str(e)}"

@mcp.resource("news://top")
async def top_news_resource() -> str:
    """Get top headline news for English (US)."""
    # Use the localized resource with default values
    return await top_news_resource_localized("en", "US")

#
# Tavily Search functionality
#

@mcp.tool()
def tavily_search(
    query: str,
    search_depth: str = "advanced",
    max_results: int = 10,
    time_range: str = "year",
    include_answer: str = "advanced",
    ctx: Context = None
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
    if ctx and hasattr(ctx.request_context.lifespan_context, 'tavily_client'):
        # Get Tavily client from context
        tavily_client = ctx.request_context.lifespan_context.tavily_client
    else:
        # Get Tavily API key from environment if context not available
        tavily_api_key = os.environ.get("TAVILY_API_KEY")
        if not tavily_api_key:
            return {"success": False, "error": "TAVILY_API_KEY environment variable not set"}
        tavily_client = TavilyClient(api_key=tavily_api_key)
    
    # Report progress
    if ctx:
        ctx.info(f"Searching for: {query}")
    
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
    # Get Tavily API key from environment
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_api_key:
        return "# Error: TAVILY_API_KEY environment variable not set"
    
    # Create a client just for this request
    tavily_client = TavilyClient(api_key=tavily_api_key)
    
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

#
# Tavily Extract functionality
#

@mcp.tool()
def extract_url(url: str, ctx: Context = None) -> dict:
    """
    Extract content from a URL using Tavily Extract API.
    
    Args:
        url: The URL to extract content from
        
    Returns:
        The extracted content
    """
    if ctx and hasattr(ctx.request_context.lifespan_context, 'tavily_client'):
        # Get Tavily client from context
        tavily_client = ctx.request_context.lifespan_context.tavily_client
    else:
        # Get Tavily API key from environment if context not available
        tavily_api_key = os.environ.get("TAVILY_API_KEY")
        if not tavily_api_key:
            return {"success": False, "error": "TAVILY_API_KEY environment variable not set"}
        tavily_client = TavilyClient(api_key=tavily_api_key)
        
    return tavily_client.extract(url)

@mcp.resource("extract://{url}")
def extract_resource(url: str) -> str:
    """
    Extract content from a URL and return as a formatted resource.
    
    Args:
        url: The URL to extract content from
    """
    try:
        # Get Tavily API key from environment
        tavily_api_key = os.environ.get("TAVILY_API_KEY")
        if not tavily_api_key:
            return "# Error: TAVILY_API_KEY environment variable not set"
        
        # Create a client just for this request
        tavily_client = TavilyClient(api_key=tavily_api_key)
        
        # Extract content
        extraction = tavily_client.extract(url)
        
        # Format as markdown
        result = f"# Content Extracted from URL\n\n"
        result += f"**Source:** [{url}]({url})\n\n"
        
        if "title" in extraction:
            result += f"## {extraction['title']}\n\n"
        
        if "text" in extraction:
            result += extraction["text"]
        
        return result
    except Exception as e:
        return f"# Error Extracting URL Content\n\nThere was a problem extracting content from '{url}': {str(e)}"

#
# PDF functionality
#

@mcp.tool()
def read_pdf(
    file_path: str,
    password: str = None,
    pages: Optional[List[int]] = None
) -> Dict:
    """
    Read a PDF file and extract its text. Works with both protected and unprotected PDFs.
    
    Args:
        file_path: Path to the PDF file
        password: Optional password to decrypt the PDF if it's protected
        pages: Optional list of specific page numbers to extract (1-indexed). If None, all pages are extracted.
        
    Returns:
        Dictionary containing the PDF content by page and metadata
    """
    # Check if file exists
    if not os.path.exists(file_path):
        return {
            "success": False,
            "error": f"File not found: {file_path}"
        }
    
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Check if PDF is encrypted
            is_encrypted = pdf_reader.is_encrypted
            
            # Try to decrypt if necessary
            decrypt_success = True
            if is_encrypted:
                if password is None:
                    return {
                        "success": False,
                        "error": "This PDF is password-protected. Please provide a password.",
                        "is_encrypted": True,
                        "password_required": True
                    }
                decrypt_success = pdf_reader.decrypt(password)
            
            # Return error if decryption failed
            if is_encrypted and not decrypt_success:
                return {
                    "success": False,
                    "error": "Incorrect password or PDF could not be decrypted",
                    "is_encrypted": True,
                    "password_required": True
                }
            
            # Extract metadata
            metadata = {}
            if pdf_reader.metadata:
                for key, value in pdf_reader.metadata.items():
                    if key.startswith('/'):
                        metadata[key[1:]] = value
                    else:
                        metadata[key] = value
            
            # Determine which pages to extract
            total_pages = len(pdf_reader.pages)
            pages_to_extract = pages or list(range(1, total_pages + 1))
            
            # Convert to 0-indexed for internal use
            zero_indexed_pages = [p - 1 for p in pages_to_extract if 1 <= p <= total_pages]
            
            # Extract content from requested pages
            content = {}
            for page_number in zero_indexed_pages:
                page = pdf_reader.pages[page_number]
                content[page_number + 1] = page.extract_text()
            
            return {
                "success": True,
                "is_encrypted": is_encrypted,
                "total_pages": total_pages,
                "extracted_pages": list(content.keys()),
                "metadata": metadata,
                "content": content
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Error processing PDF: {str(e)}"
        }

@mcp.resource("pdf://{file_path}")
def pdf_resource_no_password(file_path: str) -> str:
    """
    Read a PDF file and format its content as a resource.
    For unprotected PDFs.
    
    Args:
        file_path: Path to the PDF file
    """
    # Replace URL-encoded characters in file path
    file_path = file_path.replace('%20', ' ')
    
    result = read_pdf(file_path)
    
    if not result["success"]:
        if result.get("password_required", False):
            return f"# Password Required\n\nThis PDF is protected with a password. Please use the PDF resource with a password parameter: `pdf://{file_path}/YOUR_PASSWORD`"
        return f"# Error Reading PDF\n\n{result['error']}"
    
    # Format the PDF content as a Markdown document
    output = f"# PDF Content: {os.path.basename(file_path)}\n\n"
    
    if result["metadata"]:
        output += "## Metadata\n\n"
        for key, value in result["metadata"].items():
            output += f"- **{key}**: {value}\n"
        output += "\n"
    
    output += f"## Content ({result['total_pages']} pages total)\n\n"
    
    for page_num, page_text in result["content"].items():
        output += f"### Page {page_num}\n\n"
        output += page_text + "\n\n"
    
    return output

@mcp.resource("pdf://{file_path}/{password}")
def pdf_resource_with_password(file_path: str, password: str) -> str:
    """
    Read a password-protected PDF file and format its content as a resource.
    
    Args:
        file_path: Path to the PDF file
        password: Password to decrypt the PDF
    """
    # Replace URL-encoded characters in file path
    file_path = file_path.replace('%20', ' ')
    
    result = read_pdf(file_path, password)
    
    if not result["success"]:
        return f"# Error Reading PDF\n\n{result['error']}"
    
    # Format the PDF content as a Markdown document
    output = f"# PDF Content: {os.path.basename(file_path)}\n\n"
    
    if result["metadata"]:
        output += "## Metadata\n\n"
        for key, value in result["metadata"].items():
            output += f"- **{key}**: {value}\n"
        output += "\n"
    
    output += f"## Content ({result['total_pages']} pages total)\n\n"
    
    for page_num, page_text in result["content"].items():
        output += f"### Page {page_num}\n\n"
        output += page_text + "\n\n"
    
    return output

#
# Prompts
#

@mcp.prompt()
def connect_database_prompt(connection_string: str = "") -> str:
    """
    Create a prompt for connecting to a database.
    
    Args:
        connection_string: Optional database connection string
    """
    if connection_string:
        masked_connection = mask_password(connection_string)
        return f"""I'd like to connect to the database at {masked_connection}.

Please use the database connection tool to establish a connection and then show me what tables are available.
"""
    else:
        return """I'd like to connect to a SQL database.

Please provide the connection string in one of these formats:
- MySQL: "mysql+pymysql://user:password@host:port/database"
- PostgreSQL: "postgresql+psycopg2://user:password@host:port/database"

I'll help you explore the database schema and run queries.
"""

@mcp.prompt()
def explore_database_prompt(connection_id: str = "") -> str:
    """
    Create a prompt for exploring a connected database.
    
    Args:
        connection_id: Connection identifier returned from connect_database
    """
    return f"""I'm now connected to the database with connection ID: {connection_id}.

Let's explore this database. I can:
1. List all tables
2. Describe specific tables in detail
3. Run SQL queries
4. Analyze the data

What would you like to do first?
"""

@mcp.prompt()
def news_search_prompt(
    query: str = "", 
    language: str = "en", 
    country: str = "US"
) -> str:
    """
    Create a prompt for searching news with language and country options.
    
    Args:
        query: Optional initial search query
        language: Language code (e.g., 'en', 'id', 'es', 'fr')
        country: Country code (e.g., 'US', 'ID', 'UK', 'CA')
    """
    lang_names = {
        "en": "English",
        "id": "Indonesian",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "nl": "Dutch",
        "cs": "Czech",
        "ru": "Russian",
        "uk": "Ukrainian",
        "ja": "Japanese",
        "zh-cn": "Chinese (Simplified)",
        "zh-tw": "Chinese (Traditional)",
        "ko": "Korean",
        "ar": "Arabic"
    }
    
    country_names = {
        "US": "United States",
        "ID": "Indonesia",
        "UK": "United Kingdom",
        "CA": "Canada",
        "AU": "Australia",
        "IN": "India",
        "DE": "Germany",
        "FR": "France",
        "IT": "Italy",
        "ES": "Spain",
        "BR": "Brazil",
        "MX": "Mexico",
        "JP": "Japan",
        "KR": "South Korea",
        "RU": "Russia"
    }
    
    lang_name = lang_names.get(language, language)
    country_name = country_names.get(country, country)
    
    if query:
        return f"""I'd like to find recent news about: {query}

Please search for news in {lang_name} from {country_name}.

Use the GNews search tool with language="{language}" and country="{country}" to find relevant articles and summarize what you find.
"""
    else:
        return f"""I'd like to find recent news articles in {lang_name} from {country_name}.

What topic or subject would you like to search for? Once you tell me, I'll use the GNews search tool to find relevant articles and summarize them for you.
"""

@mcp.prompt()
def top_news_prompt(language: str = "en", country: str = "US") -> str:
    """
    Create a prompt for getting top news headlines with language and country options.
    
    Args:
        language: Language code (e.g., 'en', 'id', 'es', 'fr')
        country: Country code (e.g., 'US', 'ID', 'UK', 'CA')
    """
    lang_names = {
        "en": "English",
        "id": "Indonesian",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "nl": "Dutch",
        "cs": "Czech",
        "ru": "Russian",
        "uk": "Ukrainian",
        "ja": "Japanese",
        "zh-cn": "Chinese (Simplified)",
        "zh-tw": "Chinese (Traditional)",
        "ko": "Korean",
        "ar": "Arabic"
    }
    
    country_names = {
        "US": "United States",
        "ID": "Indonesia",
        "UK": "United Kingdom",
        "CA": "Canada",
        "AU": "Australia",
        "IN": "India",
        "DE": "Germany",
        "FR": "France",
        "IT": "Italy",
        "ES": "Spain",
        "BR": "Brazil",
        "MX": "Mexico",
        "JP": "Japan",
        "KR": "South Korea",
        "RU": "Russia"
    }
    
    lang_name = lang_names.get(language, language)
    country_name = country_names.get(country, country)
    
    return f"""I'd like to see today's top news headlines in {lang_name} from {country_name}.

Please use the GNews top news tool with language="{language}" and country="{country}" to retrieve the latest headlines and provide a brief summary of each story.
"""

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

@mcp.prompt()
def extract_prompt(url: str = "") -> str:
    """
    Create a prompt for extracting content from a URL.
    
    Args:
        url: Optional URL to extract
    """
    if url:
        return f"""I'd like to extract and analyze the content from this URL: {url}

Please use the URL extraction tool to get the content and then summarize the key points for me.
"""
    else:
        return """I'd like to extract content from a webpage.

Please provide the URL you'd like me to extract, and I'll use the URL extraction tool to get the content and summarize it for you.
"""

@mcp.prompt()
def pdf_reader_prompt(file_path: str = "") -> str:
    """
    Create a prompt for reading and summarizing a PDF file.
    
    Args:
        file_path: Path to the PDF file
    """
    if file_path:
        return f"""I have a PDF file at "{file_path}" that I'd like to read and analyze.

Please use the PDF Reader tool to extract and summarize the content of this document for me.
If the PDF is password-protected, I'll provide the password when asked.
"""
    else:
        return """I'd like to read and analyze a PDF file.

I'll provide the file path, and then I'd like you to use the PDF Reader tool to extract and summarize the document for me.
If the PDF is password-protected, I'll provide the password when asked.
"""

# Helper function to mask password in connection strings for logging
def mask_password(connection_string: str) -> str:
    """Masks the password in a database connection string for security."""
    return re.sub(r'(://.+:).+(@.+)', r'\1*****\2', connection_string)

@mcp.tool()
def read_excel(file_path: str, sheet_name: str = None) -> pd.DataFrame:
    """
    Read an Excel file and return its content as a pandas DataFrame.
    
    Args:
        file_path (str): Path to the Excel file.
        sheet_name (str, optional): Name or index of the sheet to read. 
                                   If None, reads the first sheet by default.
    
    Returns:
        pd.DataFrame: DataFrame containing the Excel sheet data.
        
    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the specified sheet does not exist in the Excel file.
    """
    try:
        # If sheet_name is None, pandas will read the first sheet by default
        if sheet_name is None:
            print(f"No specific sheet requested. Reading the first sheet from {file_path}")
            return pd.read_excel(file_path, engine="openpyxl")
        else:
            print(f"Reading sheet '{sheet_name}' from {file_path}")
            return pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
    except FileNotFoundError:
        raise FileNotFoundError(f"Excel file not found at path: {file_path}")
    except ValueError as e:
        if "No sheet named" in str(e):
            raise ValueError(f"Sheet '{sheet_name}' not found in the Excel file.")
        raise e
    except Exception as e:
        raise Exception(f"Error reading Excel file: {str(e)}")


# Allow direct execution of the server
if __name__ == "__main__":
    mcp.run()