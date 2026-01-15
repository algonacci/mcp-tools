from mcp.server.fastmcp import FastMCP, Context
from typing import List, Dict, Any, Optional, AsyncIterator
import nbformat
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
import wikipedia
import arxiv
from pathlib import Path
import httpx
import asyncio
import json
from playwright.async_api import async_playwright, Page
from playwright_stealth import Stealth
# Load environment variables
load_dotenv()

# =========================
# Notebook Parsing Utils
# =========================

def normalize_text(x):
    if isinstance(x, list):
        return "".join(x)
    return x or ""

def format_outputs(outputs):
    lines = []
    has_error = False
    for out in outputs:
        otype = out.output_type
        if otype == "stream":
            text = normalize_text(out.text).strip()
            if text:
                lines.append(text)
        elif otype in ("execute_result", "display_data"):
            data = out.data or {}
            if "text/plain" in data:
                lines.append(str(data["text/plain"]).strip())
        elif otype == "error":
            has_error = True
            lines.append("ERROR:")
            lines.append(f"{out.ename}: {out.evalue}")
            lines.extend(out.traceback)
    return "\n".join(lines), has_error

def format_cell(cell, index):
    source = normalize_text(cell.source).strip()
    if cell.cell_type == "markdown":
        return f"\n[CELL {index} | MARKDOWN]\n{source}\n"
    if cell.cell_type == "code":
        output_text, has_error = format_outputs(cell.outputs)
        execution_count = cell.execution_count
        return (
            f"\n[CELL {index} | CODE]\n"
            f"[EXECUTION_COUNT] {execution_count}\n"
            f"[HAS_ERROR] {has_error}\n\n"
            f"{source}\n\n"
            f"[OUTPUT]\n"
            f"{output_text if output_text else '<NO OUTPUT>'}\n"
        )
    return ""

def notebook_to_llm_blocks(notebook_path):
    nb = nbformat.read(notebook_path, as_version=4)
    blocks = []
    for i, cell in enumerate(nb.cells):
        block = format_cell(cell, i)
        if block.strip():
            blocks.append(block)
    return blocks

def filter_by_keyword(blocks, keywords):
    if isinstance(keywords, str):
        keywords = [keywords]
    result = []
    for block in blocks:
        text = block.lower()
        if any(k.lower() in text for k in keywords):
            result.append(block)
    return result

def filter_by_cell_index(blocks, start=None, end=None):
    result = []
    for block in blocks:
        header = block.split("\n", 1)[0]
        if not header.startswith("[CELL"):
            continue
        idx = int(header.split("[CELL")[1].split("|")[0].strip())
        if start is not None and idx < start:
            continue
        if end is not None and idx >= end:
            continue
        result.append(block)
    return result

def filter_has_error(blocks, has_error=True):
    result = []
    for block in blocks:
        for line in block.splitlines():
            if line.startswith("[HAS_ERROR]"):
                flag = line.split("]", 1)[1].strip().lower() == "true"
                if flag == has_error:
                    result.append(block)
                break
    return result

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
    dependencies=[
        "gnews", 
        "tavily-python", 
        "PyPDF2>=3.0.0",
        "python-dotenv",
        "sqlalchemy",
        "pandas",
        "pymysql",
        "psycopg2-binary",
        "pyodbc",
        "oracledb",
        "wikipedia",
        "arxiv",
        "httpx",
        "playwright",
        "playwright-stealth",
        "nbformat"
    ],
    lifespan=app_lifespan
)

# ArXiv storage path configuration
STORAGE_PATH = Path(os.getenv("ARXIV_PAPER_STORAGE_PATH", str(Path.cwd() / "downloads")))
STORAGE_PATH.mkdir(parents=True, exist_ok=True)

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
        if not (connection_string.startswith('mysql') or 
                connection_string.startswith('postgresql') or
                connection_string.startswith('postgres') or
                connection_string.startswith('sqlite') or
                connection_string.startswith('mssql') or
                connection_string.startswith('oracle')):
            
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
            # Simple pass-through for others or common alias corrections could go here
            elif "sqlite" in connection_string.lower() and not connection_string.startswith("sqlite"):
                 connection_string = "sqlite:///" + connection_string # fallback helper, maybe risky
            
            # If still not matching known prefixes (strict check removed for flexibility, but let's keep basic validation)
            if not any(connection_string.startswith(p) for p in ['mysql', 'postgres', 'sqlite', 'mssql', 'oracle']):
                 if ctx:
                     ctx.info("Connection string doesn't match common prefixes. Attempting anyway...")
        
        # Create engine and connect
        engine = create_engine(connection_string)
        connection = engine.connect()
        
        # Determine database type
        if "mysql" in connection_string.lower():
            db_type = "MySQL"
        elif "postgre" in connection_string.lower():
            db_type = "PostgreSQL"
        elif "sqlite" in connection_string.lower():
            db_type = "SQLite"
        elif "mssql" in connection_string.lower():
            db_type = "SQL Server"
        elif "oracle" in connection_string.lower():
            db_type = "Oracle"
        else:
            db_type = "Unknown URL"
        
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
- SQLite: "sqlite:///path/to/database.db" (use 4 slashes for absolute paths: sqlite:////absolute/path/db.db)
- SQL Server: "mssql+pyodbc://user:password@dsn_name" or with driver params
- Oracle: "oracle+oracledb://user:password@host:port/service_name"

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

# Wikipedia
@mcp.tool()
def search(query: str):
    return wikipedia.search(query)

@mcp.tool()
def summary(query: str):
    return wikipedia.summary(query)

@mcp.tool()
def page(query: str):
    return wikipedia.page(query)

@mcp.tool()
def random():
    return wikipedia.random()

@mcp.tool()
def set_lang(lang: str):
    wikipedia.set_lang(lang)
    return f"Language set to {lang}"


#
# ArXiv functionality
#

@mcp.tool()
def search_papers(
    query: str, 
    max_results: int = 10,
    sort_by: str = "submitted_date",
    sort_order: str = "descending"
):
    """
    Search for papers on ArXiv.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        sort_by: Criterion to sort by ("relevance", "last_updated_date", "submitted_date")
        sort_order: Order of results ("ascending", "descending")
    """
    client = arxiv.Client()

    # Map string parameters to arxiv enums
    sort_criterion = {
        "relevance": arxiv.SortCriterion.Relevance,
        "last_updated_date": arxiv.SortCriterion.LastUpdatedDate,
        "submitted_date": arxiv.SortCriterion.SubmittedDate
    }.get(sort_by, arxiv.SortCriterion.SubmittedDate)

    sort_order_enum = {
        "ascending": arxiv.SortOrder.Ascending,
        "descending": arxiv.SortOrder.Descending
    }.get(sort_order, arxiv.SortOrder.Descending)

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_criterion,
        sort_order=sort_order_enum,
    )

    results_data = []

    for r in client.results(search):
        affiliation = None
        if hasattr(r, "_raw") and isinstance(r._raw, dict):
            affiliation = r._raw.get("arxiv_affiliation")

        paper = {
            "title": r.title,
            "pdf_url": r.pdf_url,
            "authors": [author.name for author in r.authors],
            "summary": r.summary,
            "published": r.published.strftime("%Y-%m-%d"),
            "categories": r.categories,
            "entry_id": r.entry_id,
            "comment": r.comment,
            "affiliation": affiliation,
        }

        results_data.append(paper)

    return results_data

@mcp.tool()
def download_paper(paper_id: str) -> str:
    """
    Download a paper from ArXiv as a PDF.
    
    Args:
        paper_id: The ArXiv ID of the paper (e.g., "2301.12345" or the full URL)
    """
    # Clean paper_id if it's a URL
    clean_id = paper_id.split('/')[-1]
    if clean_id.endswith('v'): # handle version numbers
        clean_id = clean_id.split('v')[0]
        
    client = arxiv.Client()
    search = arxiv.Search(id_list=[clean_id])
    
    try:
        paper = next(client.results(search))
        
        # Create filename
        safe_title = "".join([c if c.isalnum() else "_" for c in paper.title])
        filename = f"{clean_id}_{safe_title[:50]}.pdf"
        filepath = STORAGE_PATH / filename
        
        # Download
        paper.download_pdf(dirpath=str(STORAGE_PATH), filename=filename)
        
        return f"Paper downloaded successfully to: {filepath}"
    except StopIteration:
        return f"Error: Paper with ID {paper_id} not found."
    except Exception as e:
        return f"Error downloading paper: {str(e)}"


#
# IEEE Xplore functionality
#

@mcp.tool()
async def search_ieee(query: str, limit: int = 10, start_year: int = None, end_year: int = None) -> str:
    """
    Search for papers on IEEE Xplore and retrieve details including abstracts (Parallel Fetching).
    
    Args:
        query: The search term (e.g., "hr cv screening")
        limit: Maximum number of results to process (default: 10)
        start_year: Optional start year filter (e.g., 2020)
        end_year: Optional end year filter (e.g., 2024)
    """
    url = "https://ieeexplore.ieee.org/rest/search"
    
    payload = {
        "newsearch": True,
        "queryText": query,
        "highlight": True,
        "returnFacets": ["ALL"],
        "returnType": "SEARCH",
        "matchPubs": True
    }
    
    # Add year range filter if provided
    if start_year and end_year:
        payload["ranges"] = [f"{start_year}_{end_year}_Year"]
    elif start_year:
        # If only start year, assume until current year + small buffer or max
        import datetime
        current_year = datetime.datetime.now().year + 1
        payload["ranges"] = [f"{start_year}_{current_year}_Year"]
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://ieeexplore.ieee.org",
        "Referer": f"https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText={query}",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # print(f"Fetching data from IEEE REST API for query: {query}...") # Optional logging for server
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            records = data.get("records", [])
            
            if not records:
                return json.dumps({"error": "No records found."}, indent=2)
            
            # Semaphore to control concurrency
            sem = asyncio.Semaphore(5)
            
            async def process_record(index, record):
                async with sem:
                    try:
                        title = record.get("articleTitle", "")
                        article_number = record.get("articleNumber", "")
                        
                        # Basic info
                        item = {
                            "index": index + 1,
                            "title": title,
                            "authors": [a.get("preferredName", "") for a in record.get("authors", [])],
                            "publication": record.get("publicationTitle", ""),
                            "year": record.get("publicationYear", ""),
                            "doi": record.get("doi", "N/A"),
                            "url": f"https://ieeexplore.ieee.org/document/{article_number}" if article_number else "N/A",
                            "pdf_url": f"https://ieeexplore.ieee.org{record.get('pdfLink', '')}" if record.get('pdfLink') else "N/A",
                            "abstract": record.get("abstract", "") # Default abstract
                        }

                        # Fetch full abstract if possible/needed
                        if article_number:
                            # Small random delay
                            await asyncio.sleep(0.1) 
                            
                            doc_response = await client.get(item["url"], headers=headers)
                            if doc_response.status_code == 200:
                                doc_text = doc_response.text
                                match = re.search(r'"abstract":"(.*?)","isbn":', doc_text)
                                if match:
                                    item["abstract"] = match.group(1)
                        
                        return item
                    except Exception as e:
                        return None

            # Create tasks
            tasks = [process_record(i, rec) for i, rec in enumerate(records[:limit])]
            results = await asyncio.gather(*tasks)
            
            # Filter results
            clean_results = [r for r in results if r is not None]
            
            return json.dumps(clean_results, indent=2)
                
        except Exception as e:
            return json.dumps({"error": f"Error occurred: {str(e)}"}, indent=2)



#
# ScienceDirect functionality
#

@mcp.tool()
async def search_sciencedirect(query: str, limit: int = 3) -> str:
    """
    Search ScienceDirect for papers and extract abstracts.
    
    Args:
        query: The search query (e.g., "text-to-sql")
        limit: Max number of results to process (default: 3)
    """
    print(f"Launching Browser (Persistent Context) to search for: {query}...")

    # Create user_data directory if not exists
    user_data_dir = os.path.join(os.getcwd(), "user_data")
    os.makedirs(user_data_dir, exist_ok=True)

    async with async_playwright() as p:
        # Using launch_persistent_context for persistence and stealth
        # Headless configurable via env var, default to False (safer for bot detection)
        # Users can set HEADLESS=true in .env if they extracted valid cookies/state
        headless_mode = os.getenv("HEADLESS", "false").lower() == "true"
        
        context = await p.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=headless_mode,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--no-sandbox',
                '--disable-setuid-sandbox',
            ],
            ignore_default_args=["--enable-automation"],
            locale="id-ID",
            viewport={"width": 1920, "height": 1080}
        )
        
        page = context.pages[0] if context.pages else await context.new_page()

        # Token capture mechanism
        token_container = {"token": None}

        async def handle_request(request):
            if "sciencedirect.com/search/api?" in request.url:
                # url parse
                from urllib.parse import urlparse, parse_qs
                parsed = urlparse(request.url)
                qs = parse_qs(parsed.query)
                t_val = qs.get("t", [None])[0]
                if t_val and not token_container["token"]:
                    token_container["token"] = t_val
                    print("Token captured via network interception.")

        page.on("request", handle_request)

        try:
            print("Navigating to ScienceDirect...")
            # Navigate to generic search page to trigger token generation
            # URL encode the query
            import urllib.parse
            encoded_query = urllib.parse.quote(query)
            
            try:
                await page.goto(f"https://www.sciencedirect.com/search?qs={encoded_query}", wait_until="domcontentloaded", timeout=60000)
            except Exception as e:
                print(f"Navigation warning: {e}")
                print("Continuing as the page might have loaded partially...")

            # Wait a bit for token if not yet caught
            if not token_container["token"]:
                await asyncio.sleep(5)
                
            # Manual intervention block
            if not token_container["token"]:
                print("Token not yet captured. Waiting 15s for manual intervention if needed...")
                await asyncio.sleep(15)

            token = token_container["token"]
            
            if not token:
                return "Error: Could not capture ScienceDirect API token. Blocking may be active."
            
            print("Token intercepted. Fetching metadata API...")

            # Execute fetch inside browser context
            js_script = """
            async (args) => {
                const { token, query } = args;
                const apiUrl = `https://www.sciencedirect.com/search/api?qs=${encodeURIComponent(query)}&t=${token}&hostname=www.sciencedirect.com`;
                try {
                    const resp = await fetch(apiUrl, {
                        headers: { "X-Requested-With": "XMLHttpRequest" }
                    });
                    if (resp.ok) return await resp.json();
                    return { error: `HTTP ${resp.status}` };
                } catch (e) {
                    return { error: e.message };
                }
            }
            """
            
            results = await page.evaluate(js_script, {"token": token, "query": query})

            if not results or results.get("error"):
                return f"API Call Failed: {results.get('error') if results else 'Unknown error'}"

            search_results = results.get("searchResults", [])
            total_found = results.get("resultsFound", 0)
            
            print(f"Found {total_found} results. Processing top {limit}...")

            output = f"# ScienceDirect Search Results for: '{query}'\n"
            output += f"**Total Found:** {total_found} | **Showing Top:** {limit}\n\n"
            
            process_count = min(limit, len(search_results))
            
            for i in range(process_count):
                record = search_results[i]
                title = record.get("title", "No Title")
                link = record.get("link", "")
                if link and not link.startswith("http"):
                    link = "https://www.sciencedirect.com" + link
                    
                doi = record.get("doi", "N/A")
                authors_list = record.get("authors", [])
                authors = "; ".join([a.get("name") for a in authors_list]) if authors_list else "N/A"

                print(f"[{i+1}/{process_count}] Navigating to extract abstract...")
                
                abstract = "Abstract could not be loaded"
                
                try:
                    await page.goto(link, wait_until="domcontentloaded", timeout=45000)
                    await asyncio.sleep(2)
                    
                    abstract = await page.evaluate(r"""() => {
                        const selectors = [
                            '#abstracts', 
                            '.Abstracts', 
                            'div[class*="Abstract"]', 
                            'section[id="abstracts"]',
                            '.abstract'
                        ];
                        
                        for (const sel of selectors) {
                            const el = document.querySelector(sel);
                            if (el && el.innerText.trim().length > 20) {
                                return el.innerText.replace(/^(Abstract|Summary)\s*/i, '').trim();
                            }
                        }
                        return null;
                    }""")
                    
                    if not abstract:
                        abstract = "Abstract section not found in the DOM (Access might be restricted)."
                    
                except Exception as e:
                    abstract = f"(Page Load Error: {str(e)})"
                
                entry = f"""
## {i+1}. {title}
**Authors:** {authors}
**DOI:** {doi}
**Link:** [View Article]({link})

### Abstract
{abstract}

---
"""
                output += entry
                await asyncio.sleep(1)

        finally:
            await context.close()
            
        return output

@mcp.tool()
def read_notebook(
    path: str,
    keywords: Optional[List[str]] = None,
    start_cell: Optional[int] = None,
    end_cell: Optional[int] = None,
    only_errors: Optional[bool] = None
) -> str:
    """
    Reads a Jupyter Notebook (.ipynb) and returns a formatted text representation for LLM analysis.
    Filters are optional and can be combined.
    
    Args:
        path: Path to the .ipynb file.
        keywords: List of keywords to filter cells (e.g., ["fit", "model"]).
        start_cell: Start cell index (inclusive).
        end_cell: End cell index (exclusive).
        only_errors: If True, only returns cells that have execution errors.
    """
    try:
        blocks = notebook_to_llm_blocks(path)
        
        if keywords:
            blocks = filter_by_keyword(blocks, keywords)
        
        if start_cell is not None or end_cell is not None:
            blocks = filter_by_cell_index(blocks, start=start_cell, end=end_cell)
            
        if only_errors is not None:
            blocks = filter_has_error(blocks, has_error=only_errors)
            
        if not blocks:
            return "No matching cells found with the specified filters."
            
        return "\n".join(blocks)
    except Exception as e:
        return f"Error reading notebook: {str(e)}"

# Allow direct execution of the server
if __name__ == "__main__":
    print("Running MCP server...")
    mcp.run()