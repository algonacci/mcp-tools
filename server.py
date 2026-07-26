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
from datetime import datetime, timezone
from urllib.parse import urljoin
from playwright.async_api import async_playwright, Page
from playwright_stealth import Stealth
from bs4 import BeautifulSoup
import email
import imaplib
import logging
import smtplib
from email.header import decode_header, make_header
from email.message import EmailMessage, Message
from uuid import uuid4
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
# Load environment variables
load_dotenv()

logger = logging.getLogger("mcp-tools")

# =========================
# Notebook Parsing Utils
# =========================

def normalize_text(x):
    if isinstance(x, list):
        return "".join(x)
    return x or ""

def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()

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
        close_email_imap()

# Configure FastMCP with dependencies and lifespan
mcp = FastMCP(
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
        "nbformat",
        "beautifulsoup4",
        "lxml",
        "google-api-python-client",
        "google-auth-oauthlib",
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
# GARUDA (Garba Rujukan Digital) functionality
#

GARUDA_BASE_URL = "https://garuda.kemdiktisaintek.go.id"


def extract_garuda_id_from_url(url: str) -> str:
    match = re.search(r"/documents/detail/(\d+)", url)
    return match.group(1) if match else ""


def build_garuda_detail_url(garuda_id_or_url: str) -> str:
    garuda_id_or_url = clean_text(garuda_id_or_url)
    if garuda_id_or_url.startswith("http://") or garuda_id_or_url.startswith("https://"):
        return garuda_id_or_url
    return f"{GARUDA_BASE_URL}/documents/detail/{garuda_id_or_url}"


def format_simple_apa_citation(article: Dict[str, Any]) -> str:
    authors = article.get("authors", [])
    journal = article.get("journal") or "Unknown journal"
    title = article.get("title") or "Untitled"
    doi = article.get("doi")
    detail_url = article.get("detail_url")

    if authors:
        author_text = ", ".join(authors[:5])
        if len(authors) > 5:
            author_text += ", et al."
    else:
        author_text = "Unknown author"

    citation = f"{author_text}. {title}. {journal}."
    if doi:
        citation += f" DOI: {doi}."
    elif detail_url:
        citation += f" {detail_url}"
    return citation.strip()


def parse_garuda_article_item(item, base_url: str = GARUDA_BASE_URL) -> Dict[str, Any]:
    title_tag = item.select_one("a.title-article")
    title = clean_text(title_tag.get_text(" ", strip=True)) if title_tag else ""
    detail_url = urljoin(base_url, title_tag.get("href", "")) if title_tag else ""
    garuda_id = extract_garuda_id_from_url(detail_url)

    authors = [
        clean_text(author.get_text(" ", strip=True))
        for author in item.select("a.author-article")
    ]

    subtitles = item.select("xmp.subtitle-article")
    journal = clean_text(subtitles[0].get_text(" ", strip=True)) if len(subtitles) > 0 else ""
    publisher = clean_text(subtitles[1].get_text(" ", strip=True)) if len(subtitles) > 1 else ""

    abstract_tag = item.select_one(".abstract-article xmp.abstract-article")
    abstract = clean_text(abstract_tag.get_text(" ", strip=True)) if abstract_tag else ""

    links = item.select("p.action-article a")
    download_original = ""
    original_source = ""
    google_scholar = ""
    full_pdf = ""
    doi = ""

    for link in links:
        text = clean_text(link.get_text(" ", strip=True))
        href = link.get("href", "")

        if "Download Original" in text:
            download_original = href
        elif "Original Source" in text:
            original_source = href
        elif "Check in Google Scholar" in text:
            google_scholar = href
        elif "Full PDF" in text:
            full_pdf = href
        elif "DOI:" in text:
            doi = text.replace("DOI:", "").strip()

    article = {
        "garuda_id": garuda_id,
        "title": title,
        "authors": authors,
        "authors_text": "; ".join(authors),
        "journal": journal,
        "publisher": publisher,
        "abstract": abstract,
        "doi": doi,
        "detail_url": detail_url,
        "download_original": download_original,
        "original_source": original_source,
        "full_pdf": full_pdf,
        "google_scholar": google_scholar,
    }
    article["citation"] = format_simple_apa_citation(article)
    return article


def parse_garuda_search_page(html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")

    found_header = soup.select_one("h2.ui.header")
    found_text = clean_text(found_header.get_text(" ", strip=True)) if found_header else ""
    total_documents = None

    match = re.search(r"Found\s+([\d,\.]+)\s+documents", found_text)
    if match:
        total_documents = int(match.group(1).replace(",", "").replace(".", ""))

    articles = [
        parse_garuda_article_item(item)
        for item in soup.select("div.article-item")
    ]

    return {
        "total_documents": total_documents,
        "articles": articles,
        "articles_on_page": len(articles),
    }


def parse_garuda_detail_page(html: str, detail_url: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")

    article_display = soup.select_one("div.article-display")
    if article_display is None:
        return {
            "garuda_id": extract_garuda_id_from_url(detail_url),
            "detail_url": detail_url,
            "title": "",
            "journal_short": "",
            "journal_volume": "",
            "authors": [],
            "publish_date": "",
            "abstract": "",
            "copyright": "",
            "download_original": "",
            "google_scholar": "",
            "citation": "",
        }

    container = article_display.find("div")
    journal_blocks = container.select("xmp") if container else []
    journal_short = clean_text(journal_blocks[0].get_text(" ", strip=True)) if len(journal_blocks) > 0 else ""
    journal_volume = clean_text(journal_blocks[1].get_text(" ", strip=True)) if len(journal_blocks) > 1 else ""

    title_tag = article_display.select_one("h3.ui.header xmp")
    title = clean_text(title_tag.get_text(" ", strip=True)) if title_tag else ""

    authors = [
        clean_text(author.get_text(" ", strip=True))
        for author in article_display.select("a[href*='/author/view/'] xmp")
    ]

    publish_date = ""
    article_info = article_display.select_one("div.four.wide.column")
    if article_info:
        info_text = clean_text(article_info.get_text(" ", strip=True))
        match = re.search(r"Publish Date\s+(.*)", info_text)
        if match:
            publish_date = clean_text(match.group(1))

    abstract_tag = article_display.select_one("xmp.abstract-article")
    abstract = clean_text(abstract_tag.get_text(" ", strip=True)) if abstract_tag else ""

    download_original = ""
    google_scholar = ""
    original_source = ""
    full_pdf = ""
    doi = ""

    for link in soup.select("a[href]"):
        text = clean_text(link.get_text(" ", strip=True))
        href = link.get("href", "")
        if "Download Original" in text:
            download_original = href
        elif "Check in Google Scholar" in text or "Google Scholar" in text:
            google_scholar = href
        elif "Original Source" in text:
            original_source = href
        elif "Full PDF" in text:
            full_pdf = href
        elif "DOI:" in text:
            doi = text.replace("DOI:", "").strip()

    copyright_text = ""
    paragraphs = article_display.select("div.art-content p")
    if paragraphs:
        copyright_text = clean_text(paragraphs[-1].get_text(" ", strip=True))

    article = {
        "garuda_id": extract_garuda_id_from_url(detail_url),
        "detail_url": detail_url,
        "title": title,
        "journal_short": journal_short,
        "journal_volume": journal_volume,
        "journal": clean_text(f"{journal_short} {journal_volume}"),
        "authors": authors,
        "authors_text": "; ".join(authors),
        "publish_date": publish_date,
        "abstract": abstract,
        "copyright": copyright_text,
        "doi": doi,
        "download_original": download_original,
        "original_source": original_source,
        "full_pdf": full_pdf,
        "google_scholar": google_scholar,
    }
    article["citation"] = format_simple_apa_citation(article)
    return article


async def fetch_garuda_search_page(
    client: httpx.AsyncClient,
    query: str,
    page: int = 1,
) -> str:
    response = await client.get(
        f"{GARUDA_BASE_URL}/documents",
        params={
            "q": query,
            "page": page,
        },
    )
    response.raise_for_status()
    return response.text


async def fetch_garuda_detail_page(
    client: httpx.AsyncClient,
    garuda_id_or_url: str,
) -> tuple[str, str]:
    detail_url = build_garuda_detail_url(garuda_id_or_url)
    response = await client.get(detail_url)
    response.raise_for_status()
    return response.text, str(response.url)


@mcp.tool()
async def search_garuda(
    query: str,
    limit: int = 10,
    max_pages: int = 3,
    include_abstract: bool = True,
    delay_seconds: float = 0.5,
) -> Dict[str, Any]:
    """
    Search Indonesian local journals and articles from GARUDA.

    Args:
        query: Search query
        limit: Maximum number of results to return
        max_pages: Maximum number of result pages to scan
        include_abstract: Whether to include abstract text in the output
        delay_seconds: Delay between page requests to stay polite to the site

    Returns:
        Matching GARUDA articles with citation-ready metadata
    """
    limit = max(1, min(limit, 50))
    max_pages = max(1, min(max_pages, 10))
    delay_seconds = max(0.0, min(delay_seconds, 5.0))

    headers = {
        "User-Agent": "Mozilla/5.0 mcp-tools-garuda/0.1",
        "Accept": "text/html,application/xhtml+xml",
    }

    collected_articles = []
    total_documents = None

    async with httpx.AsyncClient(
        headers=headers,
        timeout=30,
        follow_redirects=True,
    ) as client:
        for page in range(1, max_pages + 1):
            html = await fetch_garuda_search_page(client, query=query, page=page)
            parsed = parse_garuda_search_page(html)

            if total_documents is None:
                total_documents = parsed["total_documents"]

            for article in parsed["articles"]:
                if not include_abstract:
                    article["abstract"] = ""
                collected_articles.append(article)
                if len(collected_articles) >= limit:
                    break

            if len(collected_articles) >= limit or parsed["articles_on_page"] == 0:
                break

            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)

    return {
        "query": query,
        "source": "GARUDA",
        "source_url": GARUDA_BASE_URL,
        "total_documents": total_documents,
        "returned_results": len(collected_articles),
        "results": collected_articles,
    }


@mcp.tool()
async def get_garuda_detail(
    garuda_id_or_url: str,
) -> Dict[str, Any]:
    """
    Fetch the detail page for a GARUDA article by article id or detail URL.

    Args:
        garuda_id_or_url: Numeric GARUDA article id or full GARUDA detail URL

    Returns:
        Detailed GARUDA article metadata from the article page
    """
    headers = {
        "User-Agent": "Mozilla/5.0 mcp-garuda/0.1",
        "Accept": "text/html,application/xhtml+xml",
    }

    async with httpx.AsyncClient(
        headers=headers,
        timeout=30,
        follow_redirects=True,
    ) as client:
        html, resolved_url = await fetch_garuda_detail_page(client, garuda_id_or_url)

    return parse_garuda_detail_page(html, resolved_url)


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

# =========================
# Crypto Price Monitor
# =========================

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

COIN_ALIASES = {
    "btc": "bitcoin",
    "eth": "ethereum",
    "bnb": "binancecoin",
    "sol": "solana",
    "xrp": "ripple",
    "ada": "cardano",
    "doge": "dogecoin",
    "dot": "polkadot",
    "matic": "matic-network",
    "link": "chainlink",
    "avax": "avalanche-2",
    "uni": "uniswap",
    "ltc": "litecoin",
    "atom": "cosmos",
    "near": "near",
    "apt": "aptos",
    "arb": "arbitrum",
    "op": "optimism",
    "idr": "rupiah-token",
    "usdt": "tether",
    "usdc": "usd-coin",
    "dai": "dai",
}

def resolve_coin_id(coin: str) -> str:
    """Resolve nama/simbol coin ke CoinGecko ID."""
    coin_lower = coin.lower().strip()
    return COIN_ALIASES.get(coin_lower, coin_lower)


@mcp.tool()
async def get_price(
    coins: str,
    currencies: str = "usd,idr",
) -> dict:
    """
    Dapatkan harga crypto saat ini.

    Args:
        coins: Nama atau simbol koin, pisahkan dengan koma (contoh: "bitcoin,ethereum" atau "btc,eth,sol")
        currencies: Mata uang target, pisahkan dengan koma (contoh: "usd,idr"). Default: usd,idr

    Returns:
        Harga koin dalam mata uang yang diminta
    """
    coin_list = [resolve_coin_id(c.strip()) for c in coins.split(",")]
    coin_ids = ",".join(coin_list)

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"{COINGECKO_BASE}/simple/price",
            params={
                "ids": coin_ids,
                "vs_currencies": currencies,
                "include_24hr_change": "true",
                "include_market_cap": "true",
                "include_last_updated_at": "true",
            },
        )
        resp.raise_for_status()
        data = resp.json()

    result = {}
    for coin_id, prices in data.items():
        result[coin_id] = {}
        for key, value in prices.items():
            if key.endswith("_24h_change"):
                currency = key.replace("_24h_change", "")
                if currency not in result[coin_id]:
                    result[coin_id][currency] = {}
                result[coin_id][currency]["change_24h_pct"] = round(value, 2) if value else None
            elif key.endswith("_market_cap"):
                currency = key.replace("_market_cap", "")
                if currency not in result[coin_id]:
                    result[coin_id][currency] = {}
                result[coin_id][currency]["market_cap"] = value
            elif key == "last_updated_at":
                result[coin_id]["last_updated_at"] = value
            else:
                if key not in result[coin_id]:
                    result[coin_id][key] = {}
                result[coin_id][key]["price"] = value

    return result


@mcp.tool()
async def get_coin_detail(coin: str) -> dict:
    """
    Dapatkan detail lengkap sebuah koin termasuk info pasar, ATH, ATL, dll.

    Args:
        coin: Nama atau simbol koin (contoh: "bitcoin" atau "btc")

    Returns:
        Detail lengkap koin
    """
    coin_id = resolve_coin_id(coin)

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(
            f"{COINGECKO_BASE}/coins/{coin_id}",
            params={
                "localization": "false",
                "tickers": "false",
                "market_data": "true",
                "community_data": "false",
                "developer_data": "false",
            },
        )
        resp.raise_for_status()
        data = resp.json()

    market = data.get("market_data", {})
    return {
        "id": data.get("id"),
        "symbol": data.get("symbol", "").upper(),
        "name": data.get("name"),
        "description": (data.get("description", {}).get("en", "") or "")[:300],
        "market_cap_rank": data.get("market_cap_rank"),
        "price": {
            "usd": market.get("current_price", {}).get("usd"),
            "idr": market.get("current_price", {}).get("idr"),
            "btc": market.get("current_price", {}).get("btc"),
        },
        "price_change_24h": {
            "usd_pct": round(market.get("price_change_percentage_24h") or 0, 2),
            "7d_pct": round(market.get("price_change_percentage_7d") or 0, 2),
            "30d_pct": round(market.get("price_change_percentage_30d") or 0, 2),
        },
        "market_cap_usd": market.get("market_cap", {}).get("usd"),
        "volume_24h_usd": market.get("total_volume", {}).get("usd"),
        "circulating_supply": market.get("circulating_supply"),
        "total_supply": market.get("total_supply"),
        "ath": {
            "usd": market.get("ath", {}).get("usd"),
            "usd_date": market.get("ath_date", {}).get("usd"),
            "change_from_ath_pct": round(market.get("ath_change_percentage", {}).get("usd") or 0, 2),
        },
        "atl": {
            "usd": market.get("atl", {}).get("usd"),
            "usd_date": market.get("atl_date", {}).get("usd"),
        },
        "last_updated": data.get("last_updated"),
    }


@mcp.tool()
async def get_top_coins(limit: int = 10, currency: str = "usd") -> list[dict]:
    """
    Dapatkan daftar koin teratas berdasarkan market cap.

    Args:
        limit: Jumlah koin yang ditampilkan (1-250, default: 10)
        currency: Mata uang untuk harga (default: "usd")

    Returns:
        List koin teratas dengan harga dan data pasar
    """
    limit = max(1, min(250, limit))

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(
            f"{COINGECKO_BASE}/coins/markets",
            params={
                "vs_currency": currency,
                "order": "market_cap_desc",
                "per_page": limit,
                "page": 1,
                "sparkline": "false",
                "price_change_percentage": "24h,7d",
            },
        )
        resp.raise_for_status()
        data = resp.json()

    return [
        {
            "rank": coin.get("market_cap_rank"),
            "id": coin.get("id"),
            "symbol": (coin.get("symbol") or "").upper(),
            "name": coin.get("name"),
            "price": coin.get("current_price"),
            "market_cap": coin.get("market_cap"),
            "volume_24h": coin.get("total_volume"),
            "change_24h_pct": round(coin.get("price_change_percentage_24h") or 0, 2),
            "change_7d_pct": round(coin.get("price_change_percentage_7d_in_currency") or 0, 2),
        }
        for coin in data
    ]


@mcp.tool()
async def search_coin(query: str) -> list[dict]:
    """
    Cari koin berdasarkan nama atau simbol.

    Args:
        query: Kata kunci pencarian (contoh: "bitcoin", "pepe", "layer2")

    Returns:
        List koin yang cocok dengan hasil pencarian
    """
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"{COINGECKO_BASE}/search",
            params={"query": query},
        )
        resp.raise_for_status()
        data = resp.json()

    coins = data.get("coins", [])[:10]
    return [
        {
            "id": c.get("id"),
            "symbol": (c.get("symbol") or "").upper(),
            "name": c.get("name"),
            "market_cap_rank": c.get("market_cap_rank"),
        }
        for c in coins
    ]


@mcp.tool()
async def get_global_market() -> dict:
    """
    Dapatkan data pasar crypto global (total market cap, dominasi BTC, dll).

    Returns:
        Data pasar crypto global
    """
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{COINGECKO_BASE}/global")
        resp.raise_for_status()
        data = resp.json().get("data", {})

    return {
        "total_market_cap_usd": data.get("total_market_cap", {}).get("usd"),
        "total_volume_24h_usd": data.get("total_volume", {}).get("usd"),
        "market_cap_change_24h_pct": round(data.get("market_cap_change_percentage_24h_usd") or 0, 2),
        "btc_dominance_pct": round(data.get("market_cap_percentage", {}).get("btc") or 0, 2),
        "eth_dominance_pct": round(data.get("market_cap_percentage", {}).get("eth") or 0, 2),
        "active_cryptocurrencies": data.get("active_cryptocurrencies"),
        "markets": data.get("markets"),
        "updated_at": data.get("updated_at"),
    }


@mcp.tool()
async def get_price_history(
    coin: str,
    days: int = 7,
    currency: str = "usd",
) -> dict:
    """
    Dapatkan riwayat harga koin dalam beberapa hari terakhir.

    Args:
        coin: Nama atau simbol koin (contoh: "bitcoin" atau "btc")
        days: Jumlah hari ke belakang (1-365, default: 7)
        currency: Mata uang target (default: "usd")

    Returns:
        Riwayat harga OHLC (Open, High, Low, Close)
    """
    coin_id = resolve_coin_id(coin)
    days = max(1, min(365, days))

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(
            f"{COINGECKO_BASE}/coins/{coin_id}/ohlc",
            params={
                "vs_currency": currency,
                "days": days,
            },
        )
        resp.raise_for_status()
        ohlc_data = resp.json()

    if not ohlc_data:
        return {"coin": coin_id, "currency": currency, "days": days, "data": []}

    formatted = []
    for entry in ohlc_data:
        ts, open_p, high_p, low_p, close_p = entry
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        formatted.append({
            "datetime": dt,
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
        })

    first_close = formatted[0]["close"] if formatted else None
    last_close = formatted[-1]["close"] if formatted else None
    change_pct = None
    if first_close and last_close:
        change_pct = round(((last_close - first_close) / first_close) * 100, 2)

    return {
        "coin": coin_id,
        "currency": currency,
        "days": days,
        "summary": {
            "start_price": first_close,
            "end_price": last_close,
            "change_pct": change_pct,
            "high": max(e["high"] for e in formatted),
            "low": min(e["low"] for e in formatted),
        },
        "ohlc": formatted,
    }


@mcp.tool()
async def compare_coins(
    coins: str,
    currency: str = "usd",
) -> list[dict]:
    """
    Bandingkan beberapa koin secara side-by-side.

    Args:
        coins: Daftar koin dipisah koma (contoh: "btc,eth,sol,bnb")
        currency: Mata uang untuk perbandingan (default: "usd")

    Returns:
        Perbandingan koin-koin yang diminta
    """
    coin_list = [resolve_coin_id(c.strip()) for c in coins.split(",")]

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(
            f"{COINGECKO_BASE}/coins/markets",
            params={
                "vs_currency": currency,
                "ids": ",".join(coin_list),
                "order": "market_cap_desc",
                "per_page": 50,
                "page": 1,
                "sparkline": "false",
                "price_change_percentage": "24h,7d,30d",
            },
        )
        resp.raise_for_status()
        data = resp.json()

    return [
        {
            "rank": coin.get("market_cap_rank"),
            "id": coin.get("id"),
            "symbol": (coin.get("symbol") or "").upper(),
            "name": coin.get("name"),
            "price": coin.get("current_price"),
            "market_cap": coin.get("market_cap"),
            "volume_24h": coin.get("total_volume"),
            "change_24h_pct": round(coin.get("price_change_percentage_24h") or 0, 2),
            "change_7d_pct": round(coin.get("price_change_percentage_7d_in_currency") or 0, 2),
            "change_30d_pct": round(coin.get("price_change_percentage_30d_in_currency") or 0, 2),
            "ath": coin.get("ath"),
            "ath_change_pct": round(coin.get("ath_change_percentage") or 0, 2),
            "circulating_supply": coin.get("circulating_supply"),
        }
        for coin in data
    ]


#
# Google Calendar functionality
#

GOOGLE_CALENDAR_SCOPES = ["https://www.googleapis.com/auth/calendar"]
GOOGLE_CREDENTIALS = os.getenv("GOOGLE_CREDENTIALS", "credentials.json")
GOOGLE_TOKEN = os.getenv("GOOGLE_TOKEN", "token.json")
GOOGLE_CALENDAR_TIMEZONE = os.getenv("GOOGLE_CALENDAR_TIMEZONE", "Asia/Jakarta")

google_calendar_service = None


def google_calendar_error(exc: Exception) -> str:
    if isinstance(exc, RefreshError):
        return "Google authorization expired or was revoked. Delete token.json and reconnect."
    if isinstance(exc, HttpError):
        status = getattr(exc.resp, "status", None)
        if status == 400:
            return f"Google Calendar rejected the request: {exc.reason}"
        if status == 401:
            return "Google Calendar authentication failed. Reauthorize the account."
        if status == 403:
            return "Google Calendar denied this operation. Check account permissions and API access."
        if status == 404:
            return "Calendar or event not found."
        if status == 410:
            return "The sync token expired. Run list_events again to obtain a new sync token."
        return f"Google Calendar API error: {exc.reason}"
    return str(exc).strip() or exc.__class__.__name__


def get_google_calendar_service():
    global google_calendar_service

    if google_calendar_service is not None:
        return google_calendar_service

    credentials = None
    if os.path.exists(GOOGLE_TOKEN):
        credentials = Credentials.from_authorized_user_file(
            GOOGLE_TOKEN,
            GOOGLE_CALENDAR_SCOPES,
        )
    if credentials and credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
    elif not credentials or not credentials.valid:
        if not os.path.exists(GOOGLE_CREDENTIALS):
            raise FileNotFoundError(
                f"Google OAuth credentials not found: {GOOGLE_CREDENTIALS}"
            )
        flow = InstalledAppFlow.from_client_secrets_file(
            GOOGLE_CREDENTIALS,
            GOOGLE_CALENDAR_SCOPES,
        )
        credentials = flow.run_local_server(port=0)

    with open(GOOGLE_TOKEN, "w", encoding="utf-8") as token_file:
        token_file.write(credentials.to_json())
    google_calendar_service = build("calendar", "v3", credentials=credentials)
    return google_calendar_service


def google_calendar_event(event: Dict[str, Any]) -> Dict[str, Any]:
    start = event.get("start", {})
    end = event.get("end", {})
    entry_points = event.get("conferenceData", {}).get("entryPoints", [])
    meet_url = next(
        (item.get("uri") for item in entry_points if item.get("entryPointType") == "video"),
        event.get("hangoutLink", ""),
    )
    return {
        "id": event.get("id", ""),
        "summary": event.get("summary", "(no title)"),
        "description": event.get("description", ""),
        "location": event.get("location", ""),
        "start": start.get("dateTime") or start.get("date"),
        "end": end.get("dateTime") or end.get("date"),
        "status": event.get("status", ""),
        "html_link": event.get("htmlLink", ""),
        "meet_url": meet_url or "",
        "attendees": [item.get("email", "") for item in event.get("attendees", [])],
        "attachments": [
            {"title": item.get("title", ""), "file_url": item.get("fileUrl", "")}
            for item in event.get("attachments", [])
        ],
    }


def google_event_time(value: str, time_zone: str) -> Dict[str, str]:
    if not value.strip():
        raise ValueError("Event date-time cannot be empty")
    return {"dateTime": value, "timeZone": time_zone}


def run_google_calendar(operation):
    try:
        return operation()
    except Exception as exc:
        logger.warning("Google Calendar operation failed: %s", exc)
        return {"success": False, "error": google_calendar_error(exc)}


@mcp.tool()
def connect_calendar() -> Dict[str, Any]:
    """Authenticate with Google Calendar and cache the API client."""
    def operation():
        get_google_calendar_service().calendarList().list(maxResults=1).execute()
        return {"success": True, "message": "Connected to Google Calendar."}

    return run_google_calendar(operation)


@mcp.tool()
def calendar_health() -> Dict[str, Any]:
    """Check whether Google Calendar authentication and API access work."""
    def operation():
        calendar = get_google_calendar_service().calendars().get(
            calendarId="primary"
        ).execute()
        return {
            "healthy": True,
            "calendar": calendar.get("summary", "primary"),
            "time_zone": calendar.get("timeZone", GOOGLE_CALENDAR_TIMEZONE),
        }

    return run_google_calendar(operation)


@mcp.tool()
def list_calendars() -> Dict[str, Any]:
    """List calendars visible to the authenticated Google account."""
    def operation():
        result = get_google_calendar_service().calendarList().list().execute()
        calendars = [
            {
                "id": item.get("id", ""),
                "summary": item.get("summary", ""),
                "primary": item.get("primary", False),
                "access_role": item.get("accessRole", ""),
                "time_zone": item.get("timeZone", ""),
            }
            for item in result.get("items", [])
        ]
        return {"success": True, "count": len(calendars), "calendars": calendars}

    return run_google_calendar(operation)


@mcp.tool()
def list_events(
    calendar_id: str = "primary",
    limit: int = 10,
    time_min: str | None = None,
    time_max: str | None = None,
    query: str | None = None,
) -> Dict[str, Any]:
    """List upcoming events with optional time range and free-text search."""
    def operation():
        if not 1 <= limit <= 250:
            raise ValueError("limit must be between 1 and 250")
        params = {
            "calendarId": calendar_id,
            "timeMin": time_min or datetime.now(timezone.utc).isoformat(),
            "maxResults": limit,
            "singleEvents": True,
            "orderBy": "startTime",
        }
        if time_max:
            params["timeMax"] = time_max
        if query:
            params["q"] = query
        result = get_google_calendar_service().events().list(**params).execute()
        events = [google_calendar_event(item) for item in result.get("items", [])]
        return {
            "success": True,
            "count": len(events),
            "events": events,
            "next_sync_token": result.get("nextSyncToken"),
        }

    return run_google_calendar(operation)


@mcp.tool()
def get_event(event_id: str, calendar_id: str = "primary") -> Dict[str, Any]:
    """Get details for one Google Calendar event."""
    def operation():
        if not event_id.strip():
            raise ValueError("event_id is required")
        event = get_google_calendar_service().events().get(
            calendarId=calendar_id,
            eventId=event_id,
        ).execute()
        return {"success": True, "event": google_calendar_event(event)}

    return run_google_calendar(operation)


@mcp.tool()
def create_event(
    summary: str,
    start: str,
    end: str,
    calendar_id: str = "primary",
    description: str | None = None,
    location: str | None = None,
    attendees: List[str] | None = None,
    time_zone: str = GOOGLE_CALENDAR_TIMEZONE,
    recurrence: List[str] | None = None,
    send_updates: str = "none",
    add_google_meet: bool = False,
) -> Dict[str, Any]:
    """Create a timed event, optionally with attendees, recurrence, and Google Meet."""
    def operation():
        if not summary.strip():
            raise ValueError("summary is required")
        if send_updates not in {"all", "externalOnly", "none"}:
            raise ValueError("send_updates must be all, externalOnly, or none")
        body = {
            "summary": summary,
            "start": google_event_time(start, time_zone),
            "end": google_event_time(end, time_zone),
        }
        if description:
            body["description"] = description
        if location:
            body["location"] = location
        if attendees:
            body["attendees"] = [{"email": address} for address in attendees]
        if recurrence:
            body["recurrence"] = recurrence
        if add_google_meet:
            body["conferenceData"] = {"createRequest": {"requestId": str(uuid4())}}
        event = get_google_calendar_service().events().insert(
            calendarId=calendar_id,
            body=body,
            sendUpdates=send_updates,
            conferenceDataVersion=1 if add_google_meet else 0,
        ).execute()
        return {"success": True, "event": google_calendar_event(event)}

    return run_google_calendar(operation)


@mcp.tool()
def update_event(
    event_id: str,
    calendar_id: str = "primary",
    summary: str | None = None,
    start: str | None = None,
    end: str | None = None,
    description: str | None = None,
    location: str | None = None,
    time_zone: str = GOOGLE_CALENDAR_TIMEZONE,
    send_updates: str = "none",
) -> Dict[str, Any]:
    """Update selected fields of an existing Google Calendar event."""
    def operation():
        if not event_id.strip():
            raise ValueError("event_id is required")
        service = get_google_calendar_service()
        event = service.events().get(calendarId=calendar_id, eventId=event_id).execute()
        for key, value in {
            "summary": summary,
            "description": description,
            "location": location,
        }.items():
            if value is not None:
                event[key] = value
        if start is not None:
            event["start"] = google_event_time(start, time_zone)
        if end is not None:
            event["end"] = google_event_time(end, time_zone)
        updated = service.events().update(
            calendarId=calendar_id,
            eventId=event_id,
            body=event,
            sendUpdates=send_updates,
        ).execute()
        return {"success": True, "event": google_calendar_event(updated)}

    return run_google_calendar(operation)


@mcp.tool()
def delete_event(
    event_id: str,
    calendar_id: str = "primary",
    send_updates: str = "none",
) -> Dict[str, Any]:
    """Permanently delete a Google Calendar event."""
    def operation():
        if not event_id.strip():
            raise ValueError("event_id is required")
        get_google_calendar_service().events().delete(
            calendarId=calendar_id,
            eventId=event_id,
            sendUpdates=send_updates,
        ).execute()
        return {"success": True, "event_id": event_id, "message": "Event deleted."}

    return run_google_calendar(operation)


@mcp.tool()
def list_calendar_acl(calendar_id: str = "primary") -> Dict[str, Any]:
    """List access-control rules for a Google Calendar."""
    def operation():
        result = get_google_calendar_service().acl().list(calendarId=calendar_id).execute()
        rules = [
            {"id": item.get("id", ""), "scope": item.get("scope", {}), "role": item.get("role", "")}
            for item in result.get("items", [])
        ]
        return {"success": True, "count": len(rules), "rules": rules}

    return run_google_calendar(operation)


@mcp.tool()
def share_calendar(
    email_address: str,
    role: str = "reader",
    calendar_id: str = "primary",
) -> Dict[str, Any]:
    """Share a Google Calendar with another account."""
    def operation():
        if role not in {"none", "freeBusyReader", "reader", "writer", "owner"}:
            raise ValueError("Invalid ACL role")
        result = get_google_calendar_service().acl().insert(
            calendarId=calendar_id,
            body={"scope": {"type": "user", "value": email_address}, "role": role},
        ).execute()
        return {"success": True, "rule_id": result.get("id"), "role": result.get("role")}

    return run_google_calendar(operation)


@mcp.tool()
def watch_calendar_events(
    webhook_url: str,
    calendar_id: str = "primary",
    channel_id: str | None = None,
) -> Dict[str, Any]:
    """Create a Google push-notification channel for event changes."""
    def operation():
        if not webhook_url.startswith("https://"):
            raise ValueError("webhook_url must use HTTPS")
        channel = get_google_calendar_service().events().watch(
            calendarId=calendar_id,
            body={
                "id": channel_id or str(uuid4()),
                "type": "web_hook",
                "address": webhook_url,
            },
        ).execute()
        return {"success": True, "channel": channel}

    return run_google_calendar(operation)


@mcp.tool()
def list_event_changes(
    sync_token: str,
    calendar_id: str = "primary",
) -> Dict[str, Any]:
    """List changes since a previously returned Google Calendar sync token."""
    def operation():
        if not sync_token.strip():
            raise ValueError("sync_token is required")
        result = get_google_calendar_service().events().list(
            calendarId=calendar_id,
            syncToken=sync_token,
        ).execute()
        events = [google_calendar_event(item) for item in result.get("items", [])]
        return {
            "success": True,
            "count": len(events),
            "events": events,
            "next_sync_token": result.get("nextSyncToken"),
        }

    return run_google_calendar(operation)


#
# Email functionality
#

EMAIL_PROVIDER = os.getenv("EMAIL_PROVIDER", "gmail")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS", "")
EMAIL_SECRET = os.getenv("EMAIL_SECRET", "")
IMAP_HOST = os.getenv("IMAP_HOST", "imap.gmail.com")
IMAP_PORT = int(os.getenv("IMAP_PORT", "993"))
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))

email_imap_connection: imaplib.IMAP4_SSL | None = None


def close_email_imap() -> None:
    global email_imap_connection

    if email_imap_connection is None:
        return

    try:
        email_imap_connection.logout()
    except (imaplib.IMAP4.error, OSError):
        logger.debug("IMAP connection was already closed", exc_info=True)
    finally:
        email_imap_connection = None


def email_configuration_error() -> str | None:
    missing = [
        name
        for name, value in {
            "EMAIL_ADDRESS": EMAIL_ADDRESS,
            "EMAIL_SECRET": EMAIL_SECRET,
            "IMAP_HOST": IMAP_HOST,
            "SMTP_HOST": SMTP_HOST,
        }.items()
        if not value
    ]
    if missing:
        return f"Missing environment variables: {', '.join(missing)}"
    return None


def open_email_imap() -> imaplib.IMAP4_SSL:
    global email_imap_connection

    config_error = email_configuration_error()
    if config_error:
        raise ValueError(config_error)

    if email_imap_connection is not None:
        try:
            status, _ = email_imap_connection.noop()
            if status == "OK":
                return email_imap_connection
        except (imaplib.IMAP4.error, OSError):
            email_imap_connection = None

    logger.info("Connecting to IMAP server %s:%s", IMAP_HOST, IMAP_PORT)
    connection = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT, timeout=30)
    connection.login(EMAIL_ADDRESS, EMAIL_SECRET)
    email_imap_connection = connection
    return connection


def email_friendly_error(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    lowered = message.lower()
    if isinstance(exc, imaplib.IMAP4.error) and any(
        text in lowered for text in ("auth", "credential", "login")
    ):
        return "Authentication failed. Check EMAIL_ADDRESS and EMAIL_SECRET."
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return f"Connection failed: {message}"
    return message


def select_email_folder(
    connection: imaplib.IMAP4_SSL,
    folder: str = "INBOX",
) -> None:
    status, _ = connection.select(folder)
    if status != "OK":
        raise ValueError(f"Folder not found or cannot be opened: {folder}")


def decode_email_header(value: str | None) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except (LookupError, UnicodeError):
        return value


def decode_email_part(part: Message) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        raw_payload = part.get_payload()
        return raw_payload if isinstance(raw_payload, str) else ""

    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def parse_email_message(message: Message) -> Dict[str, Any]:
    plain_parts: List[str] = []
    html_parts: List[str] = []
    attachments: List[str] = []

    for part in message.walk():
        if part.is_multipart():
            continue

        filename = part.get_filename()
        if filename:
            attachments.append(decode_email_header(filename))
            continue
        if part.get_content_disposition() == "attachment":
            continue

        if part.get_content_type() == "text/plain":
            plain_parts.append(decode_email_part(part))
        elif part.get_content_type() == "text/html":
            html_parts.append(decode_email_part(part))

    plain_body = "\n".join(part.strip() for part in plain_parts if part.strip())
    html_body = "\n".join(part.strip() for part in html_parts if part.strip())
    html_text = ""
    if html_body:
        html_text = BeautifulSoup(html_body, "html.parser").get_text("\n", strip=True)

    return {
        "headers": {
            "subject": decode_email_header(message.get("Subject")),
            "from": decode_email_header(message.get("From")),
            "to": decode_email_header(message.get("To")),
            "cc": decode_email_header(message.get("Cc")),
            "date": decode_email_header(message.get("Date")),
            "message_id": decode_email_header(message.get("Message-ID")),
        },
        "plain_body": plain_body,
        "html_body": html_body,
        "html_text": html_text,
        "attachments": attachments,
    }


def fetch_email_message(message_id: str) -> Message:
    if not str(message_id).strip():
        raise ValueError("message_id is required")

    connection = open_email_imap()
    select_email_folder(connection)
    status, data = connection.fetch(str(message_id), "(BODY.PEEK[])")
    if status != "OK" or not data or not isinstance(data[0], tuple):
        raise LookupError(f"Message not found: {message_id}")

    raw_message = data[0][1]
    if not isinstance(raw_message, bytes):
        raise LookupError(f"Message not found: {message_id}")
    return email.message_from_bytes(raw_message)


def format_email_search_date(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("Date cannot be empty")
    for date_format in ("%Y-%m-%d", "%d-%b-%Y"):
        try:
            return datetime.strptime(value, date_format).strftime("%d-%b-%Y")
        except ValueError:
            continue
    raise ValueError("Dates must use YYYY-MM-DD or DD-Mon-YYYY format")


def quote_email_search_value(value: str) -> str:
    cleaned = value.replace("\\", "\\\\").replace('"', '\\"').strip()
    if not cleaned:
        raise ValueError("Search values cannot be empty")
    return f'"{cleaned}"'


def set_email_seen_flag(message_id: str, seen: bool) -> Dict[str, Any]:
    try:
        connection = open_email_imap()
        select_email_folder(connection)
        operation = "+FLAGS" if seen else "-FLAGS"
        status, _ = connection.store(str(message_id), operation, "\\Seen")
        if status != "OK":
            raise LookupError(f"Message not found: {message_id}")
        return {"success": True, "message_id": str(message_id), "read": seen}
    except Exception as exc:
        logger.warning("Could not update email flag: %s", exc)
        return {"success": False, "error": email_friendly_error(exc)}


def email_header_metadata(
    connection: imaplib.IMAP4_SSL,
    message_id: str,
) -> Dict[str, str] | None:
    status, data = connection.fetch(
        message_id,
        "(BODY.PEEK[HEADER.FIELDS (SUBJECT FROM DATE)])",
    )
    if status != "OK" or not data or not isinstance(data[0], tuple):
        return None
    message = email.message_from_bytes(data[0][1])
    return {
        "id": message_id,
        "subject": decode_email_header(message.get("Subject")),
        "sender": decode_email_header(message.get("From")),
        "date": decode_email_header(message.get("Date")),
    }


@mcp.tool()
def connect() -> Dict[str, Any]:
    """Establish and retain an authenticated IMAP connection."""
    try:
        open_email_imap()
        return {
            "success": True,
            "provider": EMAIL_PROVIDER,
            "message": "IMAP connection established.",
        }
    except Exception as exc:
        logger.error("IMAP connection failed: %s", exc)
        return {"success": False, "error": email_friendly_error(exc)}


@mcp.tool()
def health() -> Dict[str, Any]:
    """Check email configuration, IMAP connectivity, and authentication."""
    try:
        connection = open_email_imap()
        status, _ = connection.noop()
        if status != "OK":
            raise ConnectionError("IMAP server did not respond successfully")
        return {
            "healthy": True,
            "provider": EMAIL_PROVIDER,
            "imap_host": IMAP_HOST,
        }
    except Exception as exc:
        logger.warning("Email health check failed: %s", exc)
        return {"healthy": False, "error": email_friendly_error(exc)}


@mcp.tool()
def list_folders() -> Dict[str, Any]:
    """List folders available in the configured email account."""
    try:
        status, folders = open_email_imap().list()
        if status != "OK":
            raise ConnectionError("Could not list folders")
        values = [
            folder.decode("utf-8", errors="replace")
            for folder in folders or []
            if isinstance(folder, bytes)
        ]
        return {"success": True, "folders": values}
    except Exception as exc:
        logger.warning("Could not list email folders: %s", exc)
        return {"success": False, "error": email_friendly_error(exc)}


@mcp.tool()
def latest_emails(limit: int = 5) -> Dict[str, Any]:
    """Return metadata for the latest messages in the inbox."""
    try:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        connection = open_email_imap()
        select_email_folder(connection)
        status, data = connection.search(None, "ALL")
        if status != "OK":
            raise ConnectionError("Could not search the inbox")

        ids = data[0].split()[-limit:] if data and data[0] else []
        messages = []
        for raw_id in reversed(ids):
            metadata = email_header_metadata(connection, raw_id.decode())
            if metadata:
                messages.append(metadata)
        return {"success": True, "count": len(messages), "emails": messages}
    except Exception as exc:
        logger.warning("Could not fetch latest emails: %s", exc)
        return {"success": False, "error": email_friendly_error(exc)}


@mcp.tool()
def read_email(message_id: str) -> Dict[str, Any]:
    """Read one email without changing its read/unread state."""
    try:
        parsed = parse_email_message(fetch_email_message(message_id))
        return {"success": True, "id": str(message_id), **parsed}
    except Exception as exc:
        logger.warning("Could not read email %s: %s", message_id, exc)
        return {"success": False, "error": email_friendly_error(exc)}


@mcp.tool()
def search_emails(
    unread: bool = False,
    subject: str | None = None,
    sender: str | None = None,
    since: str | None = None,
    before: str | None = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """Search inbox email by unread status, subject, sender, and date range."""
    try:
        if not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200")
        criteria = []
        if unread:
            criteria.append("UNSEEN")
        if subject:
            criteria.extend(["SUBJECT", quote_email_search_value(subject)])
        if sender:
            criteria.extend(["FROM", quote_email_search_value(sender)])
        if since:
            criteria.extend(["SINCE", format_email_search_date(since)])
        if before:
            criteria.extend(["BEFORE", format_email_search_date(before)])
        if not criteria:
            criteria.append("ALL")

        connection = open_email_imap()
        select_email_folder(connection)
        status, data = connection.search(None, *criteria)
        if status != "OK":
            raise ValueError("The IMAP server rejected the search query")

        ids = data[0].split()[-limit:] if data and data[0] else []
        results = []
        for raw_id in reversed(ids):
            metadata = email_header_metadata(connection, raw_id.decode())
            if metadata:
                results.append(metadata)
        return {
            "success": True,
            "count": len(results),
            "criteria": criteria,
            "emails": results,
        }
    except Exception as exc:
        logger.warning("Email search failed: %s", exc)
        return {"success": False, "error": email_friendly_error(exc)}


@mcp.tool()
def send_email(
    to: str,
    subject: str,
    body: str,
    cc: str | None = None,
    bcc: str | None = None,
    html: bool = False,
) -> Dict[str, Any]:
    """Send a plain-text or HTML email through the configured SMTP server."""
    try:
        config_error = email_configuration_error()
        if config_error:
            raise ValueError(config_error)
        if not to.strip():
            raise ValueError("to is required")
        if not subject.strip():
            raise ValueError("subject is required")
        if not body:
            raise ValueError("body is required")

        message = EmailMessage()
        message["From"] = EMAIL_ADDRESS
        message["To"] = to
        message["Subject"] = subject
        if cc:
            message["Cc"] = cc
        if bcc:
            message["Bcc"] = bcc
        if html:
            fallback = BeautifulSoup(body, "html.parser").get_text("\n", strip=True)
            message.set_content(fallback)
            message.add_alternative(body, subtype="html")
        else:
            message.set_content(body)

        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=30) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_SECRET)
            smtp.send_message(message)
        return {"success": True, "message": "Email sent successfully."}
    except smtplib.SMTPAuthenticationError:
        return {
            "success": False,
            "error": "Authentication failed. Check EMAIL_ADDRESS and EMAIL_SECRET.",
        }
    except Exception as exc:
        logger.error("Could not send email: %s", exc)
        return {"success": False, "error": email_friendly_error(exc)}


@mcp.tool()
def mark_read(message_id: str) -> Dict[str, Any]:
    """Mark an inbox message as read."""
    return set_email_seen_flag(message_id, True)


@mcp.tool()
def mark_unread(message_id: str) -> Dict[str, Any]:
    """Mark an inbox message as unread."""
    return set_email_seen_flag(message_id, False)


@mcp.tool()
def delete_email(message_id: str) -> Dict[str, Any]:
    """Permanently delete an inbox message using the IMAP Deleted flag."""
    try:
        connection = open_email_imap()
        select_email_folder(connection)
        status, _ = connection.store(str(message_id), "+FLAGS", "\\Deleted")
        if status != "OK":
            raise LookupError(f"Message not found: {message_id}")
        connection.expunge()
        return {
            "success": True,
            "message_id": str(message_id),
            "message": "Email deleted permanently.",
        }
    except Exception as exc:
        logger.warning("Could not delete email %s: %s", message_id, exc)
        return {"success": False, "error": email_friendly_error(exc)}


@mcp.tool()
def list_attachments(message_id: str) -> Dict[str, Any]:
    """List attachment filenames for an email without downloading them."""
    try:
        parsed = parse_email_message(fetch_email_message(message_id))
        filenames = parsed["attachments"]
        return {
            "success": True,
            "message_id": str(message_id),
            "count": len(filenames),
            "attachments": filenames,
        }
    except Exception as exc:
        logger.warning("Could not list email attachments for %s: %s", message_id, exc)
        return {"success": False, "error": email_friendly_error(exc)}


@mcp.tool()
def summarize_email(message_id: str) -> Dict[str, Any]:
    """Return the cleaned body of an email without using an LLM."""
    try:
        parsed = parse_email_message(fetch_email_message(message_id))
        body = parsed["plain_body"] or parsed["html_text"]
        return {
            "success": True,
            "message_id": str(message_id),
            "summary": body.strip(),
        }
    except Exception as exc:
        logger.warning("Could not summarize email %s: %s", message_id, exc)
        return {"success": False, "error": email_friendly_error(exc)}


def main() -> None:
    try:
        mcp.run()
    except KeyboardInterrupt:
        logger.info("MCP server stopped")
    finally:
        close_email_imap()


# Allow direct execution of the server
if __name__ == "__main__":
    main()
