from mcp.server.fastmcp import FastMCP, Context, Image
from typing import List, Dict, Any, Optional, AsyncIterator
import nbformat
from dataclasses import dataclass
from contextlib import asynccontextmanager
import os
import re
import PyPDF2
import gnews
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text, inspect
from tavily import TavilyClient
from dotenv import load_dotenv
import wikipedia
import arxiv
from pathlib import Path
import httpx
import asyncio
import time
from io import BytesIO
import json
from datetime import date, datetime, timezone
from collections import Counter
from urllib.parse import urljoin, urlparse, parse_qs, quote_plus
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import webbrowser
from playwright.async_api import async_playwright, Page
from playwright_stealth import Stealth
from bs4 import BeautifulSoup
import email
import imaplib
import logging
import signal
import smtplib
import sys
import html
import zlib
from email.header import decode_header, make_header
from email.message import EmailMessage, Message
from uuid import uuid4
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
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
    tavily_client: Optional[TavilyClient]

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[ServerContext]:
    """Initialize clients on startup"""
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    default_gnews = gnews.GNews()
    tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None
    
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

# Generated files stay together under the repository's ignored downloads directory.
DOWNLOADS_PATH = Path(__file__).resolve().parent / "downloads"
STORAGE_PATH = DOWNLOADS_PATH / "arxiv"
CHART_STORAGE_PATH = DOWNLOADS_PATH / "charts"
PLANTUML_STORAGE_PATH = DOWNLOADS_PATH / "plantuml"

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
    if ctx and getattr(ctx.request_context.lifespan_context, "tavily_client", None):
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
    if ctx and getattr(ctx.request_context.lifespan_context, "tavily_client", None):
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
            logger.info(f"No specific sheet requested. Reading the first sheet from {file_path}")
            return pd.read_excel(file_path, engine="openpyxl")
        else:
            logger.info(f"Reading sheet '{sheet_name}' from {file_path}")
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

ARXIV_ID_PATTERN = re.compile(r"(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?", re.IGNORECASE)


def normalize_arxiv_id(paper_id: str) -> str:
    """Accept 2301.12345, 2301.12345v2, arXiv:2301.12345, abs/pdf URLs or old-style hep-th/9901001; drop the version."""
    match = ARXIV_ID_PATTERN.search(paper_id.strip())
    if not match:
        raise ValueError(f"Not a valid arXiv ID: {paper_id!r}")
    return match.group(1)


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

        arxiv_id = normalize_arxiv_id(r.entry_id)
        paper = {
            "source": "arxiv",
            "title": r.title,
            "authors": [author.name for author in r.authors],
            "year": r.published.year,
            "doi": r.doi,
            "abstract": r.summary,
            "url": r.entry_id,
            "arxiv_id": arxiv_id,
            "arxiv_doi": f"10.48550/arXiv.{arxiv_id}",
            "journal_ref": r.journal_ref,
            "primary_category": r.primary_category,
            "pdf_url": r.pdf_url,
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
    try:
        clean_id = normalize_arxiv_id(paper_id)
    except ValueError as e:
        return f"Error: {e}"

    client = arxiv.Client()
    search = arxiv.Search(id_list=[clean_id])
    
    try:
        paper = next(client.results(search))
        STORAGE_PATH.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        safe_title = "".join([c if c.isalnum() else "_" for c in paper.title])
        filename = f"{clean_id.replace('/', '_')}_{safe_title[:50]}.pdf"
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


GARUDA_DOI_PATTERN = re.compile(r"10\.\d{4,9}/[^\s\"'<>]+")
GARUDA_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


def extract_doi(value: str) -> str:
    match = GARUDA_DOI_PATTERN.search(value or "")
    return match.group(0).rstrip(".,;)") if match else ""


def extract_year(value: str) -> int | None:
    # Journal issue strings look like "Vol. 27 No. 1 (2026): January"; prefer the parenthesised year
    match = re.search(r"\((\d{4})\)", value or "")
    if match:
        return int(match.group(1))
    match = GARUDA_YEAR_PATTERN.search(value or "")
    return int(match.group(0)) if match else None


def classify_garuda_link(text: str, href: str) -> str:
    """Identify an action link by its target first (stable) and its label second (layout-dependent)."""
    if "doi.org/" in href or text.startswith("DOI:"):
        return "doi"
    if "scholar.google." in href or "Google Scholar" in text:
        return "google_scholar"
    if "Download Original" in text:
        return "download_original"
    if "Original Source" in text:
        return "original_source"
    if "Full PDF" in text:
        return "full_pdf"
    return ""


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
        kind = classify_garuda_link(text, href)

        if kind == "doi":
            doi = doi or extract_doi(href) or extract_doi(text)
        elif kind == "google_scholar":
            google_scholar = href
        elif kind == "download_original":
            download_original = href
        elif kind == "original_source":
            original_source = href
        elif kind == "full_pdf":
            full_pdf = href

    article = {
        "source": "garuda",
        "garuda_id": garuda_id,
        "title": title,
        "authors": authors,
        "authors_text": "; ".join(authors),
        "year": extract_year(journal),
        "journal": journal,
        "publisher": publisher,
        "abstract": abstract,
        "doi": doi,
        "url": detail_url,
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
        kind = classify_garuda_link(text, href)
        if kind == "doi":
            doi = doi or extract_doi(href) or extract_doi(text)
        elif kind == "google_scholar":
            google_scholar = href
        elif kind == "download_original":
            download_original = href
        elif kind == "original_source":
            original_source = href
        elif kind == "full_pdf":
            full_pdf = href

    copyright_text = ""
    paragraphs = article_display.select("div.art-content p")
    if paragraphs:
        copyright_text = clean_text(paragraphs[-1].get_text(" ", strip=True))

    article = {
        "source": "garuda",
        "garuda_id": extract_garuda_id_from_url(detail_url),
        "url": detail_url,
        "detail_url": detail_url,
        "title": title,
        "journal_short": journal_short,
        "journal_volume": journal_volume,
        "journal": clean_text(f"{journal_short} {journal_volume}"),
        "authors": authors,
        "authors_text": "; ".join(authors),
        "publish_date": publish_date,
        "year": extract_year(publish_date) or extract_year(journal_volume),
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


GARUDA_SEARCH_FIELDS = {"title", "abstract", "author", "doi"}


async def fetch_garuda_search_page(
    client: httpx.AsyncClient,
    query: str,
    page: int = 1,
    select: str = "",
    publisher: str = "",
    pdf_only: bool = False,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
) -> str:
    params: Dict[str, Any] = {
        "q": query,
        "page": page,
        "select": select,
    }
    if publisher:
        params["pub"] = publisher
    if pdf_only:
        params["pdf"] = 1
    if year_from is not None:
        params["from"] = year_from
    if year_to is not None:
        params["to"] = year_to

    response = await client.get(
        f"{GARUDA_BASE_URL}/documents",
        params=params,
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
    search_field: str = "",
    publisher: str = "",
    pdf_only: bool = False,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    limit: int = 10,
    max_pages: int = 3,
    include_abstract: bool = True,
    delay_seconds: float = 0.5,
) -> Dict[str, Any]:
    """
    Search Indonesian local journals and articles from GARUDA.

    Args:
        query: Search query (min. 3 characters, required by GARUDA)
        search_field: Which field to match query against: "title", "abstract",
            "author", or "doi". Leave empty for GARUDA's default (title/abstract).
            Use "author" to search by author name (e.g. exact author lookups
            that plain keyword search won't reliably surface).
        publisher: Optional publisher name filter (min. 3 characters)
        pdf_only: If true, only return results with a downloadable PDF
        year_from: Optional lower bound (inclusive) of publication year
        year_to: Optional upper bound (inclusive) of publication year
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

    search_field = search_field.strip().lower()
    if search_field and search_field not in GARUDA_SEARCH_FIELDS:
        raise ValueError(
            f"search_field must be one of {sorted(GARUDA_SEARCH_FIELDS)} or empty, "
            f"got {search_field!r}"
        )

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
            html = await fetch_garuda_search_page(
                client,
                query=query,
                page=page,
                select=search_field,
                publisher=publisher,
                pdf_only=pdf_only,
                year_from=year_from,
                year_to=year_to,
            )
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
        "search_field": search_field or "default",
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

IEEE_METADATA_PATTERN = re.compile(r"xplGlobal\.document\.metadata\s*=\s*(\{.*?\});\s*\n", re.DOTALL)


def clean_ieee_text(text: str | None) -> str:
    """Strip IEEE search highlight markers ([::term::]), HTML tags and entities."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text.replace("[::", "").replace("::]", ""))
    return html.unescape(re.sub(r"\s+", " ", text)).strip()


def parse_ieee_document_metadata(page_html: str) -> dict:
    """Read the JSON metadata IEEE embeds in every document page (abstract, doi, keywords, ...)."""
    match = IEEE_METADATA_PATTERN.search(page_html)
    if not match:
        return {}
    try:
        return json.loads(match.group(1))
    except ValueError:
        return {}


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
    
    # Add year range filter if provided; an open end defaults to IEEE's earliest year / next year
    if start_year or end_year:
        start = start_year or 1800
        end = end_year or datetime.now().year + 1
        payload["ranges"] = [f"{start}_{end}_Year"]
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://ieeexplore.ieee.org",
        "Referer": f"https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText={quote_plus(query)}",
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
                            "source": "ieee",
                            "title": title,
                            "authors": [a.get("preferredName", "") for a in record.get("authors", [])],
                            "publication": record.get("publicationTitle", ""),
                            "year": int(record["publicationYear"]) if str(record.get("publicationYear", "")).isdigit() else None,
                            "doi": record.get("doi") or None,
                            "url": f"https://ieeexplore.ieee.org/document/{article_number}" if article_number else "N/A",
                            "pdf_url": f"https://ieeexplore.ieee.org{record.get('pdfLink', '')}" if record.get('pdfLink') else "N/A",
                            "abstract": clean_ieee_text(record.get("abstract")), # search snippet; replaced by the full abstract below
                        }

                        # Fetch full abstract if possible/needed
                        if article_number:
                            # Small random delay
                            await asyncio.sleep(0.1) 
                            
                            doc_response = await client.get(item["url"], headers=headers)
                            if doc_response.status_code == 200:
                                doc_text = doc_response.text
                                metadata = parse_ieee_document_metadata(doc_text)
                                if metadata.get("abstract"):
                                    item["abstract"] = clean_ieee_text(metadata["abstract"])
                        
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

SCIENCEDIRECT_CHALLENGE_TITLES = ("just a moment", "attention required", "are you a robot", "access denied")


def detect_sciencedirect_block(status: int | None, title: str) -> str | None:
    """Recognise rate limiting or a bot challenge so the batch stops instead of hammering the site."""
    if status in (403, 429):
        return f"HTTP {status}"
    if any(marker in (title or "").lower() for marker in SCIENCEDIRECT_CHALLENGE_TITLES):
        return f"challenge page ({title})"
    return None


SCIENCEDIRECT_ABSTRACT_JS = r"""() => {
    const clean = (text) => text.replace(/^(Abstract|Summary)\s*/i, '').trim();

    // An article can carry several .abstract blocks (Highlights, Abstract, Graphical abstract); prefer the author abstract
    const blocks = [...document.querySelectorAll('#abstracts .abstract.author, .Abstracts .abstract.author')];
    const authorAbstract = blocks.find(el => {
        const heading = el.querySelector('h2, h3');
        return heading && /^(abstract|summary)/i.test(heading.innerText.trim());
    });
    if (authorAbstract && clean(authorAbstract.innerText).length > 20) {
        return clean(authorAbstract.innerText);
    }

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
}"""


@mcp.tool()
async def search_sciencedirect(query: str, limit: int = 3) -> str:
    """
    Search ScienceDirect for papers and extract abstracts.
    
    Args:
        query: The search query (e.g., "text-to-sql")
        limit: Max number of results to process (default: 3)

    Set SCIENCEDIRECT_CDP_URL (e.g. http://127.0.0.1:9222) to reuse a browser you started and logged into
    yourself instead of launching a new one. Abstract extraction stops at the first HTTP 403/429 or challenge.
    """
    cdp_url = os.getenv("SCIENCEDIRECT_CDP_URL")

    async with async_playwright() as p:
        if cdp_url:
            # Attach to a browser the user runs and logs into themselves, e.g. Brave started with
            # --remote-debugging-port=9222 and a dedicated research profile. That browser stays open afterwards.
            logger.info(f"Connecting to existing browser at {cdp_url} to search for: {query}...")
            browser = await p.chromium.connect_over_cdp(cdp_url)
            context = browser.contexts[0] if browser.contexts else await browser.new_context()
            page = await context.new_page()
        else:
            logger.info(f"Launching Browser (Persistent Context) to search for: {query}...")
            # Anchored to this file rather than the MCP client's working directory, so the browser session persists
            user_data_dir = os.getenv("SCIENCEDIRECT_USER_DATA_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_data")
            os.makedirs(user_data_dir, exist_ok=True)
            # Headless configurable via env var, default to False (safer for bot detection)
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
                    logger.info("Token captured via network interception.")

        page.on("request", handle_request)

        try:
            logger.info("Navigating to ScienceDirect...")
            # Navigate to generic search page to trigger token generation
            # URL encode the query
            import urllib.parse
            encoded_query = urllib.parse.quote(query)
            
            try:
                await page.goto(f"https://www.sciencedirect.com/search?qs={encoded_query}", wait_until="domcontentloaded", timeout=60000)
            except Exception as e:
                logger.info(f"Navigation warning: {e}")
                logger.info("Continuing as the page might have loaded partially...")

            # Wait a bit for token if not yet caught
            if not token_container["token"]:
                await asyncio.sleep(5)
                
            # Manual intervention block
            if not token_container["token"]:
                manual_wait = float(os.getenv("SCIENCEDIRECT_MANUAL_WAIT_SECONDS", "15"))
                logger.info(f"Token not yet captured. Waiting {manual_wait:.0f}s for manual intervention if needed...")
                await asyncio.sleep(manual_wait)

            token = token_container["token"]
            
            if not token:
                return json.dumps({"error": "Could not capture ScienceDirect API token. Blocking may be active.", "page_title": await page.title()})
            
            logger.info("Token intercepted. Fetching metadata API...")

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
                return json.dumps({"error": f"API call failed: {results.get('error') if results else 'Unknown error'}"})

            search_results = results.get("searchResults", [])
            total_found = results.get("resultsFound", 0)
            process_count = min(len(search_results), limit)

            logger.info(f"Found {total_found} results. Processing top {process_count}...")

            papers = []
            blocked = None
            for i in range(process_count):
                record = search_results[i]
                link = record.get("link", "")
                if link and not link.startswith("http"):
                    link = "https://www.sciencedirect.com" + link
                publication_date = record.get("publicationDateDisplay") or record.get("sortDate") or record.get("availableOnlineDate")
                year_match = re.search(r"\b(19|20)\d{2}\b", str(publication_date or ""))

                paper = {
                    "index": i + 1,
                    "source": "sciencedirect",
                    "title": re.sub(r"<[^>]*>", "", record.get("title") or ""),
                    "authors": [a.get("name") for a in record.get("authors") or [] if a.get("name")],
                    "year": int(year_match.group(0)) if year_match else None,
                    "doi": record.get("doi") or None,
                    "abstract": None,
                    "abstract_error": None,
                    "url": link or None,
                    "publication": record.get("sourceTitle") or None,
                    "volume_issue": record.get("volumeIssue") or None,
                    "open_access": bool(record.get("openAccess") or record.get("openArchive")),
                }

                if blocked:
                    paper["abstract_error"] = f"Skipped: stopped after {blocked}"
                    papers.append(paper)
                    continue

                logger.info(f"[{i+1}/{process_count}] Navigating to extract abstract...")
                try:
                    response = await page.goto(link, wait_until="domcontentloaded", timeout=45000)
                    await asyncio.sleep(2)
                    blocked = detect_sciencedirect_block(response.status if response else None, await page.title())
                    if blocked:
                        # Never retry or work around a block: stop the batch and hand it back to the user
                        paper["abstract_error"] = f"Blocked: {blocked}"
                        logger.info(f"Stopping abstract extraction: {blocked}")
                    else:
                        paper["abstract"] = await page.evaluate(SCIENCEDIRECT_ABSTRACT_JS)
                        if not paper["abstract"]:
                            paper["abstract_error"] = "Abstract section not found in the DOM (access might be restricted)."
                except Exception as e:
                    paper["abstract_error"] = f"Page load error: {e}"

                papers.append(paper)
                await asyncio.sleep(1)

            return json.dumps({
                "query": query,
                "source": "sciencedirect",
                "total_found": total_found,
                "returned_results": len(papers),
                "blocked": blocked,
                "results": papers,
            }, indent=2, ensure_ascii=False)

        finally:
            if cdp_url:
                await page.close()  # leave the user's browser and session running
            else:
                await context.close()

#
# Crossref (scholarly metadata) functionality
#

CROSSREF_API_URL = "https://api.crossref.org"
DOI_RESOLVER_URL = "https://doi.org"
OPENALEX_API_URL = "https://api.openalex.org"

# Contact email for Crossref's "polite" pool: no account needed, higher rate limit
CROSSREF_MAILTO = os.getenv("CROSSREF_MAILTO", "")
# Optional free key from openalex.org settings (10x the keyless daily budget); keyless works too
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "")

CROSSREF_SELECT_FIELDS = [
    "DOI", "title", "subtitle", "author", "container-title", "publisher", "type", "issued",
    "published", "published-print", "published-online", "volume", "issue", "page", "ISSN",
    "ISBN", "URL", "abstract", "subject", "license", "funder", "is-referenced-by-count",
    "references-count", "score",
]
CROSSREF_CITATION_STYLES = {
    "vancouver": "elsevier-vancouver",
    "chicago": "chicago-author-date",
    "mla": "modern-language-association",
    "harvard": "harvard-cite-them-right",
}
CROSSREF_CITATION_FORMATS = {
    "bibtex": "application/x-bibtex",
    "ris": "application/x-research-info-systems",
    "csl-json": "application/vnd.citationstyles.csl+json",
}
DOI_PATTERN = re.compile(r"10\.\d{4,9}/[^\s\"'<>]+", re.IGNORECASE)
DOI_PREFIX_PATTERN = re.compile(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", re.IGNORECASE)


class CrossrefError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


_crossref_cache: dict[str, tuple[float, str]] = {}
_crossref_throttle = {"next_request": 0.0, "interval": 0.35}
_crossref_lock = asyncio.Lock()


async def crossref_http_get(
    url: str,
    params: dict[str, Any] | None = None,
    accept: str = "application/json",
    cache_seconds: int = 3600,
) -> str:
    """GET with polite throttling, retry/backoff on 429/5xx and a small in-memory cache."""
    params = {k: v for k, v in (params or {}).items() if v not in (None, "")}
    if url.startswith(CROSSREF_API_URL) and CROSSREF_MAILTO:
        params.setdefault("mailto", CROSSREF_MAILTO)
    cache_key = f"{accept} {url} {sorted(params.items())}"
    cached = _crossref_cache.get(cache_key)
    if cached and cached[0] > time.monotonic():
        return cached[1]

    user_agent = "mcp-crossref/0.1" + (f" (mailto:{CROSSREF_MAILTO})" if CROSSREF_MAILTO else "")
    host = httpx.URL(url).host
    async with httpx.AsyncClient(timeout=30, follow_redirects=True, headers={"User-Agent": user_agent}) as client:
        for attempt in range(4):
            async with _crossref_lock:
                wait = _crossref_throttle["next_request"] - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
                _crossref_throttle["next_request"] = time.monotonic() + _crossref_throttle["interval"]
            try:
                response = await client.get(url, params=params, headers={"Accept": accept})
            except httpx.TransportError as e:
                if attempt == 3:
                    raise CrossrefError(f"Network error talking to {host}: {e}")
                await asyncio.sleep(2**attempt)
                continue

            if host == "api.crossref.org":
                # Crossref announces its per-pool rate limit on every response
                limit = response.headers.get("x-rate-limit-limit", "")
                interval = response.headers.get("x-rate-limit-interval", "1s").rstrip("s")
                if limit.isdigit() and interval.replace(".", "", 1).isdigit():
                    _crossref_throttle["interval"] = float(interval) / max(int(limit), 1)

            if response.status_code == 200:
                _crossref_cache[cache_key] = (time.monotonic() + cache_seconds, response.text)
                return response.text
            if response.status_code == 404:
                raise CrossrefError(f"Not found: {url}", 404)
            retry_after = response.headers.get("retry-after", "")
            if response.status_code == 429 and retry_after.isdigit() and int(retry_after) > 60:
                raise CrossrefError(f"{host} daily budget exhausted; resets in {int(retry_after) // 3600}h", 429)
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                await asyncio.sleep(float(retry_after) if retry_after.isdigit() else 2**attempt)
                continue
            raise CrossrefError(f"{host} returned HTTP {response.status_code}: {response.text[:300]}", response.status_code)
    raise CrossrefError(f"Exhausted retries for {url}")


async def crossref_api(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    return json.loads(await crossref_http_get(f"{CROSSREF_API_URL}{path}", params))["message"]


async def openalex_api(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """OpenAlex call that uses OPENALEX_API_KEY when set and falls back to keyless if the key is rejected."""
    global OPENALEX_API_KEY
    auth = {"mailto": CROSSREF_MAILTO} if CROSSREF_MAILTO else {}
    if OPENALEX_API_KEY:
        auth["api_key"] = OPENALEX_API_KEY
    try:
        return json.loads(await crossref_http_get(f"{OPENALEX_API_URL}{path}", {**(params or {}), **auth}))
    except CrossrefError as e:
        if OPENALEX_API_KEY and e.status in (401, 403):
            OPENALEX_API_KEY = ""
            auth.pop("api_key", None)
            return json.loads(await crossref_http_get(f"{OPENALEX_API_URL}{path}", {**(params or {}), **auth}))
        raise


def normalize_doi(value: str) -> str:
    doi = DOI_PREFIX_PATTERN.sub("", (value or "").strip()).rstrip(".,;)]}")
    if not DOI_PATTERN.fullmatch(doi):
        raise ValueError(f"Not a valid DOI: {value!r}")
    return doi.lower()


def extract_dois(text: str) -> list[str]:
    found: dict[str, None] = {}
    for match in DOI_PATTERN.findall(text or ""):
        try:
            found.setdefault(normalize_doi(match), None)
        except ValueError:
            continue
    return list(found)


def build_crossref_filter(filters: dict[str, Any]) -> str | None:
    parts = []
    for name, value in filters.items():
        if value is True:
            parts.append(f"{name}:true")
        elif value not in (None, "", False):
            parts.append(f"{name}:{value}")
    return ",".join(parts) or None


def crossref_date(node: dict | None) -> str | None:
    parts = ((node or {}).get("date-parts") or [[None]])[0]
    if not parts or parts[0] is None:
        return None
    return "-".join([f"{parts[0]:04d}", *[f"{p:02d}" for p in parts[1:3]]])


def strip_markup(text: str | None) -> str | None:
    if not text:
        return None
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"^(abstract|summary)\s*[:.]?\s*", "", text, flags=re.IGNORECASE) or None


def normalize_crossref_work(msg: dict[str, Any], include_abstract: bool = True, include_references: bool = False) -> dict[str, Any]:
    published = (
        crossref_date(msg.get("published"))
        or crossref_date(msg.get("issued"))
        or crossref_date(msg.get("published-print"))
        or crossref_date(msg.get("published-online"))
    )
    work = {
        "source": "crossref",
        "doi": (msg.get("DOI") or "").lower() or None,
        "title": strip_markup((msg.get("title") or [None])[0]),
        "subtitle": strip_markup((msg.get("subtitle") or [None])[0]),
        "authors": [
            {
                "name": a.get("name") or " ".join(x for x in (a.get("given"), a.get("family")) if x),
                "orcid": (a.get("ORCID") or "").rsplit("/", 1)[-1] or None,
                "affiliations": [af["name"] for af in a.get("affiliation", []) if af.get("name")],
            }
            for a in msg.get("author", [])
        ],
        "year": int(published[:4]) if published else None,
        "publication_date": published,
        "type": msg.get("type"),
        "journal": (msg.get("container-title") or [None])[0],
        "publisher": msg.get("publisher"),
        "volume": msg.get("volume"),
        "issue": msg.get("issue"),
        "pages": msg.get("page"),
        "issn": msg.get("ISSN", []),
        "isbn": msg.get("ISBN", []),
        "url": msg.get("URL"),
        "subjects": msg.get("subject", []),
        "cited_by_count": msg.get("is-referenced-by-count", 0),
        "reference_count": msg.get("references-count", msg.get("reference-count", 0)),
        "license": sorted({lic.get("URL") for lic in msg.get("license", []) if lic.get("URL")}),
        "funders": [{"name": f.get("name"), "awards": f.get("award", [])} for f in msg.get("funder", [])],
    }
    if include_abstract:
        work["abstract"] = strip_markup(msg.get("abstract"))
    if include_references:
        work["references"] = [
            {
                "doi": (r.get("DOI") or "").lower() or None,
                "title": r.get("article-title") or r.get("volume-title"),
                "author": r.get("author"),
                "year": r.get("year"),
                "journal": r.get("journal-title"),
                "unstructured": r.get("unstructured"),
            }
            for r in msg.get("reference", [])
        ]
    if msg.get("score") is not None:
        work["relevance_score"] = round(msg["score"], 2)
    return work


def crossref_metadata_quality(msg: dict[str, Any]) -> dict[str, Any]:
    """Completeness of the deposited metadata. NOT a measure of scientific quality."""
    authors = msg.get("author", [])
    checks = {
        "doi": bool(msg.get("DOI")),
        "issn_or_isbn": bool(msg.get("ISSN") or msg.get("ISBN")),
        "orcid": any(a.get("ORCID") for a in authors),
        "authors": bool(authors),
        "affiliations": any(a.get("affiliation") for a in authors),
        "journal": bool(msg.get("container-title")),
        "publication_date": bool(msg.get("published") or msg.get("issued")),
        "volume": bool(msg.get("volume")),
        "pages": bool(msg.get("page")),
        "abstract": bool(msg.get("abstract")),
        "references": bool(msg.get("reference")),
        "license": bool(msg.get("license")),
        "funding": bool(msg.get("funder")),
    }
    return {
        "metadata_completeness": round(sum(checks.values()) / len(checks), 2),
        "missing": [name for name, ok in checks.items() if not ok],
    }


def normalize_openalex_work(item: dict[str, Any]) -> dict[str, Any]:
    source = (item.get("primary_location") or {}).get("source") or {}
    index = item.get("abstract_inverted_index") or {}
    abstract = " ".join(word for _, word in sorted((pos, w) for w, poss in index.items() for pos in poss)) or None
    oa = item.get("open_access") or {}
    return {
        "source": "openalex",
        "doi": (item.get("doi") or "").replace("https://doi.org/", "").lower() or None,
        "openalex_id": item.get("id"),
        "title": strip_markup(item.get("title")),
        "authors": [{"name": a.get("author", {}).get("display_name")} for a in item.get("authorships", [])],
        "year": item.get("publication_year"),
        "type": item.get("type"),
        "journal": source.get("display_name"),
        "publisher": source.get("host_organization_name"),
        "cited_by_count": item.get("cited_by_count", 0),
        "open_access": {"is_oa": oa.get("is_oa"), "status": oa.get("oa_status"), "url": oa.get("oa_url")},
        "abstract": abstract,
    }


def tokenize(text: str | None) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def contains_phrase(text: str | None, phrase: str) -> bool:
    haystack, needle = tokenize(text), tokenize(phrase)
    return bool(needle) and any(haystack[i : i + len(needle)] == needle for i in range(len(haystack) - len(needle) + 1))


def compact_work(work: dict[str, Any]) -> dict[str, Any]:
    """Short form for lists: enough to judge relevance and follow up with get_crossref_work."""
    return {
        "doi": work.get("doi"),
        "title": work.get("title"),
        "authors": [a["name"] for a in work.get("authors", [])][:6],
        "year": work.get("year"),
        "journal": work.get("journal"),
        "cited_by_count": work.get("cited_by_count", 0),
    }


@mcp.tool()
async def search_crossref(
    query: str = "",
    author: str = "",
    title: str = "",
    bibliographic: str = "",
    journal: str = "",
    publisher: str = "",
    affiliation: str = "",
    funder: str = "",
    from_date: str = "",
    until_date: str = "",
    work_type: str = "",
    issn: str = "",
    orcid: str = "",
    has_abstract: bool = False,
    has_full_text: bool = False,
    sort: str = "relevance",
    order: str = "desc",
    rows: int = 10,
    offset: int = 0,
    include_abstract: bool = False,
) -> dict[str, Any]:
    """
    Search ~170 million scholarly works registered in Crossref (all publishers: Elsevier, IEEE,
    Springer, ACM, MDPI, Indonesian journals, ...). Every result carries a DOI you can pass to the
    other crossref tools.

    When to use:
        - Broad literature discovery across publishers ("papers on X since 2022").
        - Finding a specific paper from a messy citation string (use `bibliographic`).
        - Listing an author's or a journal's works (use `author` / `orcid` / `issn`).

    How matching works (important):
        Crossref has NO exact-phrase search. `query="retrieval augmented generation"` matches any work
        containing ANY of those words, so total_results is inflated and the tail is noise. Rely on the
        top relevance-sorted results, add filters, or use analyze_crossref_topic for phrase-accurate trends.

    Args:
        query: Free-text keywords over all metadata, e.g. "graph neural network traffic forecasting".
        author: Author name, e.g. "Geoffrey Hinton". Fuzzy; combine with `orcid` for precision.
        title: Words that should appear in the title.
        bibliographic: A full or partial citation string, e.g.
            "LeCun Bengio Hinton 2015 Deep learning Nature". Best tool for "find this exact paper".
        journal: Journal / proceedings name, e.g. "Expert Systems with Applications".
        publisher: Publisher name, e.g. "IEEE".
        affiliation: Author affiliation text, e.g. "Universitas Indonesia" (only works where deposited).
        funder: Funder name, e.g. "LPDP" or "National Science Foundation".
        from_date: Earliest publication date, "YYYY", "YYYY-MM" or "YYYY-MM-DD".
        until_date: Latest publication date, same formats.
        work_type: Crossref type id, e.g. "journal-article", "proceedings-article", "book-chapter",
            "posted-content" (preprints), "dissertation", "dataset".
        issn: Restrict to one journal by ISSN, e.g. "0957-4174".
        orcid: Restrict to works carrying this ORCID iD, e.g. "0000-0002-1825-0097".
        has_abstract: Only works with a deposited abstract (many publishers do not deposit them).
        has_full_text: Only works with full-text links deposited (does NOT mean open access).
        sort: "relevance" (default), "published", "is-referenced-by-count" (most cited),
            "references-count", "updated", "created".
        order: "desc" (default) or "asc".
        rows: Results to return, 1-100 (default 10).
        offset: Skip this many results for paging (Crossref caps offset at 10,000).
        include_abstract: Add abstracts to results (longer output). Default False to save tokens;
            fetch a single paper's abstract with get_crossref_work instead.

    Returns:
        {"total_results": int, "returned_results": int, "items": [work, ...]} where each work has
        doi, title, authors, year, journal, publisher, type, cited_by_count, reference_count, url, ...

    Examples:
        search_crossref(query="large language model education", from_date="2023", work_type="journal-article")
        search_crossref(author="Yoshua Bengio", sort="is-referenced-by-count", rows=5)
        search_crossref(bibliographic="Vaswani 2017 Attention is all you need")
        search_crossref(query="deep learning", issn="2169-3536", sort="published")

    Note: cited_by_count counts only citations registered in Crossref; it is not a quality measure.
    """
    rows = max(1, min(rows, 100))
    params = {
        "query": query,
        "query.author": author,
        "query.title": title,
        "query.bibliographic": bibliographic,
        "query.container-title": journal,
        "query.publisher-name": publisher,
        "query.affiliation": affiliation,
        "query.funder-name": funder,
        "filter": build_crossref_filter({
            "from-pub-date": from_date,
            "until-pub-date": until_date,
            "type": work_type,
            "issn": issn,
            "orcid": orcid,
            "has-abstract": has_abstract,
            "has-full-text": has_full_text,
        }),
        "sort": sort,
        "order": order,
        "rows": rows,
        "offset": offset or None,
        "select": ",".join(CROSSREF_SELECT_FIELDS),
    }
    message = await crossref_api("/works", params)
    items = [normalize_crossref_work(item, include_abstract=include_abstract) for item in message.get("items", [])]
    return {
        "total_results": message.get("total-results", 0),
        "returned_results": len(items),
        "items": items,
    }


@mcp.tool()
async def get_crossref_work(doi: str) -> dict[str, Any]:
    """
    Get the complete, normalized metadata record for one DOI.

    When to use:
        - You have a DOI (from any search tool, a PDF, a reference list) and need authors with ORCID
          and affiliations, journal, volume/issue/pages, abstract, license, funders, citation counts.
        - Checking whether a DOI is real before citing it (a 404 means Crossref does not know it).

    Args:
        doi: Any DOI spelling: "10.1145/3065386", "https://doi.org/10.1145/3065386",
            "doi:10.1145/3065386". Case does not matter.

    Returns:
        The work record plus "metadata_quality": {"metadata_completeness": 0-1, "missing": [...]},
        which tells you which fields the publisher did not deposit (e.g. abstract). It describes the
        metadata only, never the scientific quality of the paper.

    Tips:
        - abstract is often null because many publishers do not deposit abstracts to Crossref;
          snowball_doi returns OpenAlex's abstract when it has one.
        - arXiv DOIs (10.48550/arXiv.*) are registered with DataCite, not Crossref, so this returns
          "not found"; cite_dois still works for them.
        - Use get_crossref_references for the reference list.
    """
    message = await crossref_api(f"/works/{normalize_doi(doi)}")
    work = normalize_crossref_work(message)
    work["metadata_quality"] = crossref_metadata_quality(message)
    return work


@mcp.tool()
async def cite_dois(dois: list[str], style: str = "apa") -> dict[str, Any]:
    """
    Format citations for one or more DOIs in any citation style or export format.

    Uses DOI content negotiation, so it works for Crossref AND DataCite DOIs (arXiv, Zenodo, ...).

    When to use:
        - Building a bibliography / reference list for a thesis or paper.
        - Exporting references to Zotero, Mendeley or EndNote (bibtex / ris).

    Args:
        dois: List of DOIs (max 50), e.g. ["10.1038/nature14539", "10.1145/3065386"].
        style: Citation style or export format:
            - "apa" (default), "ieee", "vancouver", "chicago", "mla", "harvard", "nature"
            - any other CSL style id from https://github.com/citation-style-language/styles,
              e.g. "american-medical-association"
            - "bibtex", "ris" or "csl-json" for reference-manager exports

    Returns:
        {"style": str, "citations": [{"doi": str, "citation": str}], "failed": [{"doi", "error"}]}

    Example:
        cite_dois(["10.1038/nature14539"], style="ieee")
        -> "Y. LeCun, Y. Bengio, and G. Hinton, “Deep learning,” Nature, vol. 521, no. 7553, ..."
    """
    style_key = style.strip().lower()
    accept = CROSSREF_CITATION_FORMATS.get(style_key) or (
        f"text/x-bibliography; style={CROSSREF_CITATION_STYLES.get(style_key, style_key)}; locale=en-US"
    )
    citations, failed = [], []
    for raw in dois[:50]:
        try:
            doi = normalize_doi(raw)
            text = (await crossref_http_get(f"{DOI_RESOLVER_URL}/{doi}", accept=accept, cache_seconds=86400)).strip()
            if style_key not in CROSSREF_CITATION_FORMATS:
                # DataCite returns HTML-formatted text styles; numbered styles prefix "[1]" or "1."
                text = html.unescape(re.sub(r"<[^>]+>", "", text))
                text = re.sub(r"^(\[\d+\]|\d+\.)\s*", "", text)
            citations.append({"doi": doi, "citation": text})
        except (ValueError, CrossrefError) as e:
            failed.append({"doi": raw, "error": str(e)})
    return {"style": style, "citations": citations, "failed": failed}


@mcp.tool()
async def resolve_dois(text: str, max_dois: int = 25) -> dict[str, Any]:
    """
    Find every DOI mentioned in free text and resolve each one to clean metadata.

    When to use:
        - The user pastes a messy reference list, a PDF's text, notes or a URL list and wants to know
          what the papers are, check that the DOIs exist, or turn them into a clean table.
        - Verifying DOIs produced by another tool or model before citing them.

    Args:
        text: Any text containing DOIs in any form ("doi:10.x/y", "https://doi.org/10.x/y", bare).
        max_dois: Resolve at most this many unique DOIs (default 25, max 100).

    Returns:
        {"found": int, "works": [compact work], "failed": [{"doi", "error"}]}. A DOI in "failed"
        with "Not found" is unknown to Crossref (typo, fabricated, or registered with DataCite).

    Follow-up: pass the resolved DOIs to cite_dois to format a bibliography.
    """
    dois = extract_dois(text)
    works, failed = [], []
    for doi in dois[: max(1, min(max_dois, 100))]:
        try:
            works.append(compact_work(normalize_crossref_work(await crossref_api(f"/works/{doi}"), include_abstract=False)))
        except CrossrefError as e:
            failed.append({"doi": doi, "error": str(e)})
    return {"found": len(dois), "works": works, "failed": failed}


@mcp.tool()
async def get_crossref_references(doi: str, resolve: int = 0) -> dict[str, Any]:
    """
    List the references (bibliography) of a paper, i.e. the older works it cites.

    When to use:
        - Backward snowballing in a literature review: find the foundational papers a key paper builds on.
        - Checking which sources a paper relies on.

    Args:
        doi: DOI of the citing paper.
        resolve: Also fetch full metadata for the first N references that have a DOI (max 20),
            sorted by citation count, to spot the most influential ones. Default 0 (no extra calls).

    Returns:
        {"doi", "title", "reference_count", "references": [{doi, title, author, year, journal,
        unstructured}], "resolved": [compact work]}

    Notes:
        - Only references the publisher deposited are available; some publishers deposit none.
        - Crossref does not expose the reverse direction (papers that cite this one); use snowball_doi.
    """
    message = await crossref_api(f"/works/{normalize_doi(doi)}")
    work = normalize_crossref_work(message, include_abstract=False, include_references=True)
    resolved = []
    for ref in [r for r in work["references"] if r["doi"]][: max(0, min(resolve, 20))]:
        try:
            resolved.append(compact_work(normalize_crossref_work(await crossref_api(f"/works/{ref['doi']}"), include_abstract=False)))
        except CrossrefError:
            continue
    return {
        "doi": work["doi"],
        "title": work["title"],
        "reference_count": work["reference_count"],
        "references": work["references"],
        "resolved": sorted(resolved, key=lambda w: -w["cited_by_count"]),
    }


@mcp.tool()
async def find_related_works(doi: str, rows: int = 10) -> dict[str, Any]:
    """
    Find works bibliographically similar to a given paper (same title vocabulary and subjects).

    When to use:
        - "More like this" from one good paper, without needing its references or citations.

    Args:
        doi: DOI of the seed paper.
        rows: Number of similar works to return (1-50, default 10).

    Returns:
        {"seed": compact work, "related": [compact work + relevance_score]}

    Tip: snowball_doi gives citation-based neighbours (references, citing works, OpenAlex related),
    which are usually more meaningful than text similarity.
    """
    seed = normalize_crossref_work(await crossref_api(f"/works/{normalize_doi(doi)}"), include_abstract=False)
    message = await crossref_api("/works", {
        "query.bibliographic": " ".join(filter(None, [seed["title"], seed["subtitle"], *seed["subjects"][:3]])),
        "rows": max(1, min(rows, 50)) + 5,
        "select": ",".join(CROSSREF_SELECT_FIELDS),
    })
    related = []
    for item in message.get("items", []):
        work = normalize_crossref_work(item, include_abstract=False)
        if work["doi"] != seed["doi"] and work["title"] != seed["title"]:
            related.append({**compact_work(work), "relevance_score": work.get("relevance_score")})
    return {"seed": compact_work(seed), "related": related[:rows]}


@mcp.tool()
async def analyze_crossref_topic(
    phrase: str,
    scan: int = 500,
    from_year: int | None = None,
    until_year: int | None = None,
    work_type: str = "",
) -> dict[str, Any]:
    """
    Describe a research topic: publications per year, top venues, publishers and funders, and the
    most cited works, counting only titles that contain the exact phrase.

    When to use:
        - Trend questions: "is X growing?", "when did X take off?", thesis/proposal background.
        - "Where is X published?", "who funds X?" (venue and funder landscape).

    How it works:
        Crossref has no phrase search, so a plain query for "retrieval augmented generation" matches
        about a million works. This tool scans the top `scan` relevance-ranked hits and keeps only
        works whose title contains the exact phrase, then aggregates them. It is a sample of the most
        relevant works, not a complete count; older years may be under-represented.

    Args:
        phrase: The topic phrase, e.g. "retrieval augmented generation" (hyphens/case ignored).
        scan: Relevance hits to scan, 100-2000 (default 500). Larger = slower but more complete.
        from_year: Optional earliest publication year.
        until_year: Optional latest publication year.
        work_type: Optional Crossref type, e.g. "journal-article".

    Returns:
        {"phrase", "fuzzy_total" (all keyword matches, for context), "scanned", "matched",
         "per_year": {year: count}, "top_venues", "top_publishers", "top_funders": [[name, count]],
         "most_cited": [compact work]}

    Counts describe metadata only; they do not rank venue or funder quality.
    """
    scan = max(100, min(scan, 2000))
    filters = build_crossref_filter({
        "from-pub-date": str(from_year) if from_year else None,
        "until-pub-date": f"{until_year}-12-31" if until_year else None,
        "type": work_type,
    })
    fuzzy_total = (await crossref_api("/works", {"query": phrase, "filter": filters, "rows": 0})).get("total-results", 0)
    matched, scanned, cursor = [], 0, "*"
    while scanned < scan and cursor:
        message = await crossref_api("/works", {
            "query": phrase,
            "filter": filters,
            "rows": min(100, scan - scanned),
            "cursor": cursor,
            # cursor paging does not rank by relevance unless asked to
            "sort": "relevance",
            "select": "DOI,title,subtitle,author,container-title,publisher,issued,published,funder,is-referenced-by-count",
        })
        items = message.get("items", [])
        if not items:
            break
        scanned += len(items)
        cursor = message.get("next-cursor")
        for item in items:
            work = normalize_crossref_work(item, include_abstract=False)
            if contains_phrase(f"{work['title']} {work['subtitle'] or ''}", phrase):
                matched.append(work)
    unique = list({w["doi"]: w for w in matched}.values())
    return {
        "phrase": phrase,
        "fuzzy_total": fuzzy_total,
        "scanned": scanned,
        "matched": len(unique),
        "per_year": dict(sorted(Counter(w["year"] for w in unique if w["year"]).items())),
        "top_venues": Counter(w["journal"] for w in unique if w["journal"]).most_common(8),
        "top_publishers": Counter(w["publisher"] for w in unique if w["publisher"]).most_common(8),
        "top_funders": Counter(f["name"] for w in unique for f in w["funders"] if f["name"]).most_common(8),
        "most_cited": [compact_work(w) for w in sorted(unique, key=lambda w: -w["cited_by_count"])[:10]],
    }


@mcp.tool()
async def get_crossref_author(name: str, orcid: str = "", max_works: int = 100) -> dict[str, Any]:
    """
    Build an author profile: publications, years active, frequent co-authors, venues, affiliations
    and ORCID iDs seen in Crossref metadata.

    When to use:
        - "Who is this researcher / what do they work on / who do they collaborate with?"
        - Finding collaborators or research groups around a person.

    Args:
        name: Author name, e.g. "Geoffrey Hinton". Matching is fuzzy, so namesakes can be mixed in.
        orcid: ORCID iD (e.g. "0000-0002-1825-0097"). When given, only works carrying this iD are used,
            which removes namesakes. If the result lists several ORCID iDs, rerun with one of them.
        max_works: Works to scan, 20-500 (default 100).

    Returns:
        {"name", "orcid_filter", "scanned", "matched", "total_citations", "years": {year: count},
         "orcids_seen", "affiliations", "coauthors", "venues": [[name, count]],
         "most_cited": [compact work]}

    Counts describe Crossref metadata, not research impact; centrality is not quality.
    """
    max_works = max(20, min(max_works, 500))
    # cursor paging does not rank by relevance unless asked to
    params: dict[str, Any] = {"select": ",".join(CROSSREF_SELECT_FIELDS), "rows": min(100, max_works), "sort": "relevance"}
    if orcid:
        params["filter"] = build_crossref_filter({"orcid": orcid})
    else:
        params["query.author"] = name
    raw, cursor = [], "*"
    while len(raw) < max_works and cursor:
        message = await crossref_api("/works", {**params, "cursor": cursor, "rows": min(100, max_works - len(raw))})
        items = message.get("items", [])
        if not items:
            break
        raw += items
        cursor = message.get("next-cursor")

    name_tokens = tokenize(name)
    works, seen = [], set()
    coauthors, venues, years, orcids, affiliations = Counter(), Counter(), Counter(), Counter(), Counter()
    for item in raw:
        work = normalize_crossref_work(item, include_abstract=False)
        if orcid:
            me = next((a for a in work["authors"] if a["orcid"] == orcid), None)
        else:
            # surname must match exactly, given names by initial
            me = next((a for a in work["authors"] if name_tokens and tokenize(a["name"])[-1:] == name_tokens[-1:]
                       and all(any(t.startswith(n[0]) for t in tokenize(a["name"])) for n in name_tokens[:-1])), None)
        if not me or work["doi"] in seen:
            continue
        seen.add(work["doi"])
        works.append(work)
        coauthors.update(a["name"] for a in work["authors"] if a is not me and a["name"])
        if work["journal"]:
            venues[work["journal"]] += 1
        if work["year"]:
            years[work["year"]] += 1
        if me["orcid"]:
            orcids[me["orcid"]] += 1
        affiliations.update(me["affiliations"])
    return {
        "name": name,
        "orcid_filter": orcid or None,
        "scanned": len(raw),
        "matched": len(works),
        "total_citations": sum(w["cited_by_count"] for w in works),
        "years": dict(sorted(years.items())),
        "orcids_seen": orcids.most_common(5),
        "affiliations": affiliations.most_common(5),
        "coauthors": coauthors.most_common(15),
        "venues": venues.most_common(10),
        "most_cited": [compact_work(w) for w in sorted(works, key=lambda w: -w["cited_by_count"])[:10]],
    }


@mcp.tool()
async def get_crossref_journal(issn_or_name: str, latest: int = 5) -> dict[str, Any]:
    """
    Look up a journal: publisher, ISSNs, subjects, DOI counts, metadata coverage and latest works.

    When to use:
        - "Tell me about journal X", "what does X publish lately?", "does X deposit abstracts/ORCIDs?"
        - Finding a journal's ISSN from its name (then filter search_crossref by issn).

    Args:
        issn_or_name: An ISSN like "0957-4174" for the full profile, or a name like
            "expert systems with applications" to list matching journals with their ISSNs.
        latest: Number of most recent works to include for an ISSN lookup (0-20, default 5).

    Returns:
        For an ISSN: {"title", "publisher", "issn", "subjects", "total_dois", "coverage" (share of
        current works with abstracts, ORCIDs, references, licenses, ...), "dois_by_year", "latest"}.
        For a name: {"matches": [{"title", "publisher", "issn", "total_dois"}]}.

    Coverage is metadata completeness, not a journal quality ranking.
    """
    value = issn_or_name.strip()
    if not re.fullmatch(r"\d{4}-\d{3}[\dXx]", value):
        message = await crossref_api("/journals", {"query": value, "rows": 10})
        return {
            "matches": [
                {
                    "title": j.get("title"),
                    "publisher": j.get("publisher"),
                    "issn": j.get("ISSN", []),
                    "total_dois": j.get("counts", {}).get("total-dois", 0),
                }
                for j in message.get("items", [])
            ]
        }
    journal = await crossref_api(f"/journals/{value}")
    latest_works = []
    if latest > 0:
        message = await crossref_api(f"/journals/{value}/works", {
            "rows": min(latest, 20), "sort": "published", "order": "desc", "select": ",".join(CROSSREF_SELECT_FIELDS),
        })
        latest_works = [compact_work(normalize_crossref_work(i, include_abstract=False)) for i in message.get("items", [])]
    coverage = journal.get("coverage", {})
    return {
        "title": journal.get("title"),
        "publisher": journal.get("publisher"),
        "issn": journal.get("ISSN", []),
        "subjects": [s.get("name") for s in journal.get("subjects", [])],
        "total_dois": journal.get("counts", {}).get("total-dois", 0),
        "coverage": {k.removesuffix("-current"): round(v, 2) for k, v in sorted(coverage.items()) if k.endswith("-current")},
        "dois_by_year": dict(sorted(journal.get("breakdowns", {}).get("dois-by-issued-year") or [])[-15:]),
        "latest": latest_works,
    }


@mcp.tool()
async def get_crossref_funder(name_or_id: str, latest: int = 5) -> dict[str, Any]:
    """
    Look up a research funder and the works it funded (as declared by publishers in Crossref).

    When to use:
        - "What has LPDP / NSF / Horizon Europe funded?", grant output tracking, funder landscape.

    Args:
        name_or_id: Funder name ("LPDP", "National Science Foundation") or Crossref Funder ID
            ("501100014538").
        latest: Number of most recent funded works to include (0-20, default 5).

    Returns:
        {"funder": {"id", "name", "location", "alt_names"}, "other_matches", "funded_works_total",
         "per_year": {year: count}, "top_venues": [[name, count]], "latest": [compact work]}
    """
    if name_or_id.strip().isdigit():
        funder = await crossref_api(f"/funders/{name_or_id.strip()}")
        others = []
    else:
        matches = (await crossref_api("/funders", {"query": name_or_id, "rows": 5})).get("items", [])
        if not matches:
            return {"error": f"No funder matches {name_or_id!r}"}
        funder, others = matches[0], matches[1:]
    message = await crossref_api(f"/funders/{funder['id']}/works", {
        "rows": max(0, min(latest, 20)),
        "sort": "published",
        "order": "desc",
        "facet": "published:40,container-title:8",
        "select": ",".join(CROSSREF_SELECT_FIELDS),
    })
    facets = message.get("facets", {})
    return {
        "funder": {
            "id": funder.get("id"),
            "name": funder.get("name"),
            "location": funder.get("location"),
            "alt_names": funder.get("alt-names", [])[:8],
        },
        "other_matches": [{"id": f.get("id"), "name": f.get("name")} for f in others],
        "funded_works_total": message.get("total-results", 0),
        "per_year": dict(sorted((int(y), c) for y, c in facets.get("published", {}).get("values", {}).items() if y.isdigit())),
        "top_venues": list(facets.get("container-title", {}).get("values", {}).items()),
        "latest": [compact_work(normalize_crossref_work(i, include_abstract=False)) for i in message.get("items", [])],
    }


@mcp.tool()
async def snowball_doi(doi: str, rows: int = 10) -> dict[str, Any]:
    """
    Citation snowballing around one paper: its references (backward), the works that cite it
    (forward), and related works, plus open-access status and an abstract when available.

    Forward citations are not available from Crossref, so this tool uses OpenAlex (free; set
    OPENALEX_API_KEY for a 10x daily budget, otherwise it runs keyless).

    When to use:
        - Systematic literature review snowballing from one or two seed papers.
        - "Who built on this paper?", "what newer work cites it?", "is there a free PDF?"

    Args:
        doi: DOI of the seed paper.
        rows: Works per direction, sorted by citation count (1-50, default 10).

    Returns:
        {"seed": work with open_access + abstract, "backward_total", "backward": [...],
         "forward_total", "forward": [...], "related": [...],
         "strong_candidates": works found by more than one direction}

    Notes:
        - OpenAlex matching is automatic and occasionally wrong; sanity-check odd entries.
        - open_access.url is a legal free copy when OpenAlex knows one; DOI != free PDF.
        - Costs about 4 OpenAlex list calls; single lookups are free.
    """
    rows = max(1, min(rows, 50))
    seed = await openalex_api(f"/works/doi:{normalize_doi(doi)}")

    async def by_ids(ids: list[str]) -> list[dict[str, Any]]:
        out = []
        for i in range(0, len(ids), 100):
            chunk = "|".join(x.rsplit("/", 1)[-1] for x in ids[i : i + 100])
            out += (await openalex_api("/works", {"filter": f"openalex:{chunk}", "per_page": 100})).get("results", [])
        return [normalize_openalex_work(w) for w in out]

    backward = await by_ids(seed.get("referenced_works", []))
    citing = await openalex_api("/works", {
        "filter": f"cites:{seed['id'].rsplit('/', 1)[-1]}", "per_page": rows, "sort": "cited_by_count:desc",
    })
    forward = [normalize_openalex_work(w) for w in citing.get("results", [])]
    related = await by_ids(seed.get("related_works", [])[:rows])

    def top(works: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{**compact_work(w), "open_access": w["open_access"]["is_oa"]}
                for w in sorted(works, key=lambda w: -w["cited_by_count"])[:rows]]

    seen: Counter = Counter()
    for group in (backward, forward, related):
        seen.update({w["doi"] for w in group if w["doi"]})
    all_works = {w["doi"]: w for w in backward + forward + related if w["doi"]}
    return {
        "seed": normalize_openalex_work(seed),
        "backward_total": len(backward),
        "backward": top(backward),
        "forward_total": citing.get("meta", {}).get("count", 0),
        "forward": top(forward),
        "related": top(related),
        "strong_candidates": [compact_work(all_works[d]) for d, n in seen.most_common() if n > 1][:rows],
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


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
# Frankfurter currency functionality
#

FRANKFURTER_BASE_URL = "https://api.frankfurter.dev/v2"
FRANKFURTER_TIMEOUT = 15.0


def frankfurter_request(
    path: str,
    params: Dict[str, Any] | None = None,
) -> Any:
    try:
        response = httpx.get(
            f"{FRANKFURTER_BASE_URL}/{path}",
            params=params,
            timeout=FRANKFURTER_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json()
        except ValueError:
            detail = exc.response.text
        raise ValueError(f"Frankfurter API error: {detail}") from exc
    except httpx.RequestError as exc:
        raise ConnectionError(f"Could not reach Frankfurter API: {exc}") from exc


def validate_currency_code(value: str, name: str) -> str:
    code = value.strip().upper()
    if len(code) != 3 or not code.isalpha():
        raise ValueError(f"{name} must be a three-letter currency code")
    return code


def parse_currency_date(value: str, name: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"{name} must use YYYY-MM-DD format") from exc


@mcp.tool()
def list_currencies(
    query: str | None = None,
    include_legacy: bool = False,
) -> Dict[str, Any]:
    """List supported currencies, optionally including legacy currencies."""
    currencies = frankfurter_request(
        "currencies",
        {"scope": "all"} if include_legacy else None,
    )
    if query:
        needle = query.strip().lower()
        currencies = [
            item
            for item in currencies
            if needle in item.get("iso_code", "").lower()
            or needle in item.get("name", "").lower()
        ]
    return {"count": len(currencies), "currencies": currencies}


@mcp.tool()
def get_currency(code: str) -> Dict[str, Any]:
    """Get details and provider coverage for one currency."""
    return frankfurter_request(
        f"currency/{validate_currency_code(code, 'code')}"
    )


@mcp.tool()
def list_exchange_rate_providers(query: str | None = None) -> Dict[str, Any]:
    """List central banks and other exchange-rate data providers."""
    providers = frankfurter_request("providers")
    if query:
        needle = query.strip().lower()
        providers = [
            item
            for item in providers
            if needle in item.get("key", "").lower()
            or needle in item.get("name", "").lower()
            or needle in (item.get("country_code") or "").lower()
        ]
    return {"count": len(providers), "providers": providers}


@mcp.tool()
def get_rates(
    base: str = "EUR",
    quotes: str | None = None,
    rate_date: str | None = None,
    providers: str | None = None,
    include_providers: bool = False,
) -> Dict[str, Any]:
    """Get latest or historical rates for multiple comma-separated quote currencies."""
    params = {"base": validate_currency_code(base, "base")}
    if quotes:
        params["quotes"] = ",".join(
            validate_currency_code(code, "quotes") for code in quotes.split(",")
        )
    if rate_date:
        params["date"] = parse_currency_date(rate_date, "rate_date").isoformat()
    if providers:
        params["providers"] = providers.upper()
    if include_providers:
        params["expand"] = "providers"
    rates = frankfurter_request("rates", params)
    return {"count": len(rates), "rates": rates}


@mcp.tool()
def get_exchange_rate(
    base: str,
    quote: str,
    rate_date: str | None = None,
    providers: str | None = None,
) -> Dict[str, Any]:
    """Get a latest or historical rate, optionally from selected providers."""
    base_code = validate_currency_code(base, "base")
    quote_code = validate_currency_code(quote, "quote")
    if base_code == quote_code:
        raise ValueError("base and quote currencies must be different")
    params = {}
    if rate_date:
        params["date"] = parse_currency_date(rate_date, "rate_date").isoformat()
    if providers:
        params["providers"] = providers.upper()
    return frankfurter_request(f"rate/{base_code}/{quote_code}", params or None)


@mcp.tool()
def convert_currency(
    amount: float,
    base: str,
    quote: str,
    rate_date: str | None = None,
    providers: str | None = None,
) -> Dict[str, Any]:
    """Convert an amount using a latest or historical exchange rate."""
    if amount < 0:
        raise ValueError("amount cannot be negative")
    rate = get_exchange_rate(base, quote, rate_date, providers)
    return {
        "date": rate["date"],
        "base": rate["base"],
        "quote": rate["quote"],
        "rate": rate["rate"],
        "amount": amount,
        "converted_amount": amount * rate["rate"],
    }


@mcp.tool()
def get_exchange_rate_history(
    base: str,
    quote: str,
    start_date: str,
    end_date: str,
    group: str | None = None,
    providers: str | None = None,
    include_providers: bool = False,
) -> Dict[str, Any]:
    """Get a daily, weekly, or monthly exchange-rate time series."""
    base_code = validate_currency_code(base, "base")
    quote_code = validate_currency_code(quote, "quote")
    if base_code == quote_code:
        raise ValueError("base and quote currencies must be different")
    start = parse_currency_date(start_date, "start_date")
    end = parse_currency_date(end_date, "end_date")
    if end < start:
        raise ValueError("end_date cannot be before start_date")
    days = (end - start).days
    if group not in {None, "week", "month"}:
        raise ValueError("group must be week, month, or omitted")
    if days > 366 and group is None:
        raise ValueError("ranges over one year must use weekly or monthly grouping")
    if days > 36525:
        raise ValueError("date range cannot exceed 100 years")

    params = {
        "from": start.isoformat(),
        "to": end.isoformat(),
        "base": base_code,
        "quotes": quote_code,
    }
    if group:
        params["group"] = group
    if providers:
        params["providers"] = providers.upper()
    if include_providers:
        params["expand"] = "providers"
    rates = frankfurter_request("rates", params)
    return {
        "base": base_code,
        "quote": quote_code,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "count": len(rates),
        "rates": rates,
    }


def render_data_chart(
    data: pd.DataFrame,
    x_column: str,
    y_columns: List[str],
    chart_type: str,
    title: str,
    x_label: str | None,
    y_label: str | None,
    source_note: str | None,
    filename: str,
    dpi: int,
    annotate: bool,
) -> tuple[Path, Dict[str, Any]]:
    if chart_type not in {"line", "bar", "scatter"}:
        raise ValueError("chart_type must be line, bar, or scatter")
    if not 120 <= dpi <= 400:
        raise ValueError("dpi must be between 120 and 400")
    missing = [column for column in [x_column, *y_columns] if column not in data.columns]
    if missing:
        raise ValueError(f"Columns not found: {', '.join(missing)}")
    if not y_columns:
        raise ValueError("At least one y column is required")

    frame = data[[x_column, *y_columns]].dropna(subset=[x_column]).copy()
    if frame.empty:
        raise ValueError("No rows remain after removing empty x values")
    for column in y_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=y_columns, how="all")
    if frame.empty:
        raise ValueError("Selected y columns contain no numeric values")
    frame = frame.reset_index(drop=True)

    x_values = frame[x_column]
    positions = list(range(len(frame)))
    categorical_x = False
    if not pd.api.types.is_numeric_dtype(x_values):
        text_values = x_values.astype(str)
        looks_like_date = text_values.str.match(
            r"^\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?(?:[ T].*)?$"
        ).all()
        if looks_like_date:
            parsed_dates = pd.to_datetime(x_values, errors="coerce")
            if parsed_dates.notna().all():
                x_values = parsed_dates
        else:
            categorical_x = True
            x_values = positions

    fig, ax = plt.subplots(figsize=(11, 6.2))
    colors = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed"]
    width = 0.8 / len(y_columns)

    for index, column in enumerate(y_columns):
        values = frame[column]
        color = colors[index % len(colors)]
        if chart_type == "line":
            ax.plot(x_values, values, label=column, color=color, linewidth=2.3, marker="o", markersize=3.5)
        elif chart_type == "scatter":
            ax.scatter(x_values, values, label=column, color=color, s=38, alpha=0.85)
        else:
            offsets = [position + (index - (len(y_columns) - 1) / 2) * width for position in positions]
            bars = ax.bar(offsets, values, width=width, label=column, color=color, alpha=0.9)
            if annotate and len(frame) <= 30:
                ax.bar_label(bars, fmt="%g", padding=3, fontsize=8)

        if annotate and chart_type != "bar" and len(frame) <= 30:
            valid = values.dropna()
            for row_index in {valid.index[0], valid.index[-1]} if not valid.empty else set():
                x_value = x_values.iloc[row_index] if hasattr(x_values, "iloc") else x_values[row_index]
                y_value = values.loc[row_index]
                ax.annotate(
                    f"{y_value:,.4g}",
                    (x_value, y_value),
                    xytext=(5, 7),
                    textcoords="offset points",
                    fontsize=8,
                    color=color,
                )

    if chart_type == "bar" or categorical_x:
        ax.set_xticks(positions, [str(value) for value in frame[x_column]], rotation=30, ha="right")
    ax.set_title(title, fontsize=16, fontweight="bold", loc="left", pad=18)
    ax.set_xlabel(x_label or x_column)
    ax.set_ylabel(y_label or "Value")
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    if len(y_columns) > 1:
        ax.legend(frameon=False, ncol=min(len(y_columns), 3))

    statistics = {
        column: {
            "minimum": float(frame[column].min()),
            "maximum": float(frame[column].max()),
            "average": float(frame[column].mean()),
            "latest": float(frame[column].dropna().iloc[-1]),
        }
        for column in y_columns
        if frame[column].notna().any()
    }
    summary = "\n".join(
        f"{column}: min {stats['minimum']:,.4g} | avg {stats['average']:,.4g} | max {stats['maximum']:,.4g}"
        for column, stats in statistics.items()
    )
    ax.text(
        0,
        -0.19,
        summary,
        transform=ax.transAxes,
        fontsize=8.5,
        color="#374151",
        va="top",
    )
    if source_note:
        fig.text(0.99, 0.01, source_note, ha="right", fontsize=8, color="#6b7280")

    CHART_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
    safe_filename = re.sub(r"[^A-Za-z0-9._-]+", "-", filename).strip("-") or "chart.png"
    if not safe_filename.lower().endswith(".png"):
        safe_filename += ".png"
    path = CHART_STORAGE_PATH / safe_filename
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=dpi, format="png", bbox_inches="tight")
    plt.close(fig)
    return path, {"rows": len(frame), "statistics": statistics}


def load_tabular_file(file_path: str, sheet_name: str | None = None) -> tuple[Path, pd.DataFrame]:
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return path, pd.read_csv(path)
    if suffix in {".xlsx", ".xlsm"}:
        return path, pd.read_excel(path, sheet_name=sheet_name or 0)
    raise ValueError("file_path must be a CSV, XLSX, or XLSM file")


@mcp.tool()
def inspect_data_file(
    file_path: str,
    sheet_name: str | None = None,
    preview_rows: int = 5,
) -> Dict[str, Any]:
    """Inspect a CSV or Excel file before analysis or charting.

    Returns sheet names, columns, inferred data types, dimensions, and a small row preview.
    """
    if not 1 <= preview_rows <= 20:
        raise ValueError("preview_rows must be between 1 and 20")
    path = Path(file_path).expanduser().resolve()
    sheet_names = None
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        if not path.is_file():
            raise FileNotFoundError(f"Data file not found: {path}")
        sheet_names = pd.ExcelFile(path).sheet_names
    path, data = load_tabular_file(str(path), sheet_name)
    preview = data.head(preview_rows).where(pd.notna(data), None).to_dict(orient="records")
    return {
        "file_path": str(path),
        "sheet_names": sheet_names,
        "selected_sheet": sheet_name or (sheet_names[0] if sheet_names else None),
        "row_count": len(data),
        "column_count": len(data.columns),
        "columns": [str(column) for column in data.columns],
        "data_types": {str(column): str(dtype) for column, dtype in data.dtypes.items()},
        "preview": preview,
    }


@mcp.tool()
def create_inline_data_chart(
    data: List[Dict[str, Any]],
    x_column: str,
    y_columns: List[str],
    chart_type: str = "line",
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    source_note: str = "Source: generated data",
    filename: str = "inline-data-chart.png",
    dpi: int = 200,
    annotate: bool = True,
) -> List[Any]:
    """Create an informative high-DPI PNG chart from inline records.

    Use this directly for dummy, generated, or small user-provided datasets instead of writing a
    script or temporary CSV. Prefer explicit titles and labels, annotations, and a source note.
    """
    if not data:
        raise ValueError("data cannot be empty")
    if len(data) > 1000:
        raise ValueError("data cannot contain more than 1000 records")
    frame = pd.DataFrame(data)
    chart_path, details = render_data_chart(
        frame,
        x_column,
        y_columns,
        chart_type,
        title or f"{', '.join(y_columns)} by {x_column}",
        x_label,
        y_label,
        source_note,
        filename,
        dpi,
        annotate,
    )
    metadata = {
        "success": True,
        "file_path": str(chart_path),
        "mime_type": "image/png",
        "chart_type": chart_type,
        "x_column": x_column,
        "y_columns": y_columns,
        "dpi": dpi,
        **details,
    }
    return [metadata, Image(path=chart_path)]


@mcp.tool()
def create_data_chart(
    file_path: str,
    x_column: str,
    y_columns: List[str],
    chart_type: str = "line",
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    source_note: str | None = None,
    sheet_name: str | None = None,
    filename: str = "data-chart.png",
    dpi: int = 200,
    annotate: bool = True,
) -> list[Any]:
    """Create an informative high-DPI PNG chart from selected CSV or Excel columns.

    Choose columns that answer the user's question. Prefer an explicit title and axis labels,
    retain annotations unless the chart is dense, and include a short source_note when known. For
    dummy or generated data that is not already in a file, use create_inline_data_chart instead.
    """
    path, data = load_tabular_file(file_path, sheet_name)

    chart_path, details = render_data_chart(
        data,
        x_column,
        y_columns,
        chart_type,
        title or f"{', '.join(y_columns)} by {x_column}",
        x_label,
        y_label,
        source_note or f"Source: {path.name}",
        filename,
        dpi,
        annotate,
    )
    metadata = {
        "success": True,
        "file_path": str(chart_path),
        "mime_type": "image/png",
        "chart_type": chart_type,
        "x_column": x_column,
        "y_columns": y_columns,
        "dpi": dpi,
        **details,
    }
    return [metadata, Image(path=chart_path)]


@mcp.tool()
def create_exchange_rate_chart(
    base: str,
    quote: str,
    start_date: str,
    end_date: str,
    group: str | None = None,
    providers: str | None = None,
    title: str | None = None,
) -> list[Any]:
    """Create and save a PNG line chart for an exchange-rate time series."""
    history = get_exchange_rate_history(
        base,
        quote,
        start_date,
        end_date,
        group,
        providers,
        False,
    )
    if not history["rates"]:
        raise ValueError("No exchange-rate data found for the requested range")

    chart_title = title or f"{history['base']} to {history['quote']} exchange rate"
    filename = (
        f"{history['base']}-{history['quote']}_"
        f"{history['start_date']}_{history['end_date']}.png"
    )
    data = pd.DataFrame(history["rates"])
    path, details = render_data_chart(
        data,
        "date",
        ["rate"],
        "line",
        chart_title,
        "Date",
        f"{history['quote']} per 1 {history['base']}",
        "Source: Frankfurter API",
        filename,
        200,
        True,
    )
    values = data["rate"]

    metadata = {
        "success": True,
        "file_path": str(path.resolve()),
        "mime_type": "image/png",
        "base": history["base"],
        "quote": history["quote"],
        "start_date": history["start_date"],
        "end_date": history["end_date"],
        "data_points": history["count"],
        "minimum_rate": min(values),
        "maximum_rate": max(values),
        "latest_rate": float(values.iloc[-1]),
        **details,
    }
    return [metadata, Image(path=path)]


#
# PlantUML rendering functionality
#

PLANTUML_SERVER = "https://www.plantuml.com/plantuml"
PLANTUML_MAX_SOURCE_BYTES = 256 * 1024
PLANTUML_MAX_OUTPUT_BYTES = 10 * 1024 * 1024


def plantuml_encode6bit(value: int) -> str:
    if value < 10:
        return chr(48 + value)
    value -= 10
    if value < 26:
        return chr(65 + value)
    value -= 26
    if value < 26:
        return chr(97 + value)
    return "-" if value == 26 else "_"


def plantuml_append3bytes(first: int, second: int, third: int) -> str:
    values = (
        first >> 2,
        ((first & 0x3) << 4) | (second >> 4),
        ((second & 0xF) << 2) | (third >> 6),
        third & 0x3F,
    )
    return "".join(plantuml_encode6bit(value & 0x3F) for value in values)


def plantuml_encode(source: str) -> str:
    compressor = zlib.compressobj(level=9, wbits=-15)
    data = compressor.compress(source.encode("utf-8")) + compressor.flush()
    encoded = []
    for index in range(0, len(data), 3):
        chunk = data[index : index + 3]
        encoded.append(
            plantuml_append3bytes(
                chunk[0],
                chunk[1] if len(chunk) > 1 else 0,
                chunk[2] if len(chunk) > 2 else 0,
            )
        )
    return "".join(encoded)


@mcp.tool()
def render_plantuml(source: str, filename: str = "diagram.png") -> List[Any]:
    """Render PlantUML source as PNG using the official PlantUML Server.

    Preserve the source exactly as supplied. The source must include @startuml and @enduml.
    """
    if not source.strip():
        raise ValueError("source cannot be empty")
    if "@startuml" not in source.lower():
        raise ValueError("source must contain @startuml")
    if "@enduml" not in source.lower():
        raise ValueError("source must contain @enduml")
    if len(source.encode("utf-8")) > PLANTUML_MAX_SOURCE_BYTES:
        raise ValueError("source exceeds the 256 KiB limit")

    url = f"{PLANTUML_SERVER}/png/{plantuml_encode(source)}"
    try:
        response = httpx.get(url, timeout=30.0)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        details = (
            exc.response.headers.get("x-plantuml-diagram-description")
            or exc.response.headers.get("x-plantuml-diagram-error")
            or f"HTTP {exc.response.status_code}"
        )
        raise ValueError(f"PlantUML rendering failed: {details}") from exc
    except httpx.RequestError as exc:
        raise ConnectionError(f"Could not reach the official PlantUML Server: {exc}") from exc

    content_type = response.headers.get("content-type", "").split(";", 1)[0]
    if content_type != "image/png":
        raise ValueError(f"PlantUML rendering failed: unexpected content type {content_type}")
    if response.headers.get("x-plantuml-diagram-error"):
        raise ValueError("PlantUML rendering failed because the source contains a syntax error")
    if len(response.content) > PLANTUML_MAX_OUTPUT_BYTES:
        raise ValueError("Rendered PNG exceeds the 10 MiB limit")

    safe_filename = re.sub(r"[^A-Za-z0-9._-]+", "-", filename).strip("-") or "diagram.png"
    if not safe_filename.lower().endswith(".png"):
        safe_filename += ".png"
    PLANTUML_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
    path = PLANTUML_STORAGE_PATH / safe_filename
    path.write_bytes(response.content)
    metadata = {
        "success": True,
        "file_path": str(path),
        "mime_type": "image/png",
        "size_bytes": len(response.content),
        "server": PLANTUML_SERVER,
    }
    return [metadata, Image(path=path)]


#
# Open-Meteo weather functionality
#

OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_TIMEOUT = 15.0

OPEN_METEO_CURRENT_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "is_day",
    "precipitation",
    "rain",
    "weather_code",
    "cloud_cover",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
]
OPEN_METEO_HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation_probability",
    "precipitation",
    "rain",
    "weather_code",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
]
OPEN_METEO_DAILY_VARIABLES = [
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "sunrise",
    "sunset",
    "precipitation_sum",
    "precipitation_probability_max",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
]
OPEN_METEO_WMO_DESCRIPTIONS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def open_meteo_request(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        response = httpx.get(url, params=params, timeout=OPEN_METEO_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        try:
            reason = exc.response.json().get("reason", exc.response.text)
        except ValueError:
            reason = exc.response.text
        raise ValueError(f"Open-Meteo API error: {reason}") from exc
    except httpx.RequestError as exc:
        raise ConnectionError(f"Could not reach Open-Meteo: {exc}") from exc


def validate_weather_coordinates(latitude: float, longitude: float) -> None:
    if not -90 <= latitude <= 90:
        raise ValueError("latitude must be between -90 and 90")
    if not -180 <= longitude <= 180:
        raise ValueError("longitude must be between -180 and 180")


def describe_weather(item: Dict[str, Any]) -> Dict[str, Any]:
    code = item.get("weather_code")
    if code is not None:
        item["weather_description"] = OPEN_METEO_WMO_DESCRIPTIONS.get(code, "Unknown")
    return item


def weather_rows(data: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(data)
    if not keys:
        return []
    return [
        describe_weather(dict(zip(keys, values)))
        for values in zip(*(data[key] for key in keys))
    ]


@mcp.tool()
def search_weather_locations(
    name: str,
    count: int = 5,
    language: str = "en",
) -> Dict[str, Any]:
    """Find cities or places and return coordinates suitable for weather tools."""
    if not name.strip():
        raise ValueError("name is required")
    if not 1 <= count <= 100:
        raise ValueError("count must be between 1 and 100")
    data = open_meteo_request(
        OPEN_METEO_GEOCODING_URL,
        {"name": name, "count": count, "language": language, "format": "json"},
    )
    locations = [
        {
            "id": item.get("id"),
            "name": item.get("name", ""),
            "latitude": item.get("latitude"),
            "longitude": item.get("longitude"),
            "elevation": item.get("elevation"),
            "timezone": item.get("timezone", ""),
            "country": item.get("country", ""),
            "country_code": item.get("country_code", ""),
            "admin1": item.get("admin1", ""),
            "admin2": item.get("admin2", ""),
        }
        for item in data.get("results", [])
    ]
    return {"count": len(locations), "locations": locations}


@mcp.tool()
def get_current_weather(
    latitude: float,
    longitude: float,
    timezone: str = "auto",
) -> Dict[str, Any]:
    """Get current weather conditions for geographic coordinates."""
    validate_weather_coordinates(latitude, longitude)
    data = open_meteo_request(
        OPEN_METEO_FORECAST_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": ",".join(OPEN_METEO_CURRENT_VARIABLES),
            "timezone": timezone,
        },
    )
    return {
        "latitude": data.get("latitude"),
        "longitude": data.get("longitude"),
        "elevation": data.get("elevation"),
        "timezone": data.get("timezone"),
        "units": data.get("current_units", {}),
        "current": describe_weather(data.get("current", {})),
    }


@mcp.tool()
def get_hourly_forecast(
    latitude: float,
    longitude: float,
    forecast_days: int = 2,
    timezone: str = "auto",
) -> Dict[str, Any]:
    """Get an hourly weather forecast for up to 16 days."""
    validate_weather_coordinates(latitude, longitude)
    if not 1 <= forecast_days <= 16:
        raise ValueError("forecast_days must be between 1 and 16")
    data = open_meteo_request(
        OPEN_METEO_FORECAST_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(OPEN_METEO_HOURLY_VARIABLES),
            "forecast_days": forecast_days,
            "timezone": timezone,
        },
    )
    return {
        "latitude": data.get("latitude"),
        "longitude": data.get("longitude"),
        "timezone": data.get("timezone"),
        "units": data.get("hourly_units", {}),
        "forecast": weather_rows(data.get("hourly", {})),
    }


@mcp.tool()
def get_daily_forecast(
    latitude: float,
    longitude: float,
    forecast_days: int = 7,
    timezone: str = "auto",
) -> Dict[str, Any]:
    """Get a daily weather forecast for up to 16 days."""
    validate_weather_coordinates(latitude, longitude)
    if not 1 <= forecast_days <= 16:
        raise ValueError("forecast_days must be between 1 and 16")
    data = open_meteo_request(
        OPEN_METEO_FORECAST_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "daily": ",".join(OPEN_METEO_DAILY_VARIABLES),
            "forecast_days": forecast_days,
            "timezone": timezone,
        },
    )
    return {
        "latitude": data.get("latitude"),
        "longitude": data.get("longitude"),
        "timezone": data.get("timezone"),
        "units": data.get("daily_units", {}),
        "forecast": weather_rows(data.get("daily", {})),
    }


#
# Google Calendar and Drive functionality
#

GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/drive.readonly",
]
GOOGLE_CREDENTIALS = os.getenv("GOOGLE_CREDENTIALS", "credentials.json")
GOOGLE_TOKEN = os.getenv("GOOGLE_TOKEN", "token.json")
GOOGLE_CALENDAR_TIMEZONE = os.getenv("GOOGLE_CALENDAR_TIMEZONE", "Asia/Jakarta")
GOOGLE_DRIVE_DOWNLOAD_PATH = Path(
    os.getenv("GOOGLE_DRIVE_DOWNLOAD_PATH", "downloads/google-drive")
)

google_calendar_service = None
google_drive_service = None
google_credentials = None


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


def get_google_credentials():
    global google_credentials
    if google_credentials is not None and google_credentials.valid:
        return google_credentials

    credentials = None
    if os.path.exists(GOOGLE_TOKEN):
        with open(GOOGLE_TOKEN, encoding="utf-8") as token_file:
            token_data = json.load(token_file)
        granted_scopes = token_data.get("scopes", [])
        if isinstance(granted_scopes, str):
            granted_scopes = granted_scopes.split()
        if set(GOOGLE_SCOPES).issubset(granted_scopes):
            credentials = Credentials.from_authorized_user_info(
                token_data,
                GOOGLE_SCOPES,
            )
    if credentials and credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
    elif not credentials or not credentials.valid:
        if not os.path.exists(GOOGLE_CREDENTIALS):
            raise FileNotFoundError(
                f"Google OAuth credentials not found: {GOOGLE_CREDENTIALS}"
            )
        # Deliberately not run_local_server(): it opens a browser on whatever machine the server
        # runs on and blocks the tool call until someone clicks. On a headless host that fails with
        # "could not locate runnable browser"; even with a browser it stalls an MCP call for
        # minutes. Authorization is an explicit, two-step user action instead.
        raise PermissionError(
            "Google is not authorized yet. Call google_auth_start to get an authorization URL, "
            "then google_auth_complete with the URL you land on."
        )

    with open(GOOGLE_TOKEN, "w", encoding="utf-8") as token_file:
        token_file.write(credentials.to_json())
    google_credentials = credentials
    return google_credentials


def get_google_calendar_service():
    global google_calendar_service
    if google_calendar_service is None:
        google_calendar_service = build(
            "calendar", "v3", credentials=get_google_credentials()
        )
    return google_calendar_service


def get_google_drive_service():
    global google_drive_service
    if google_drive_service is None:
        google_drive_service = build("drive", "v3", credentials=get_google_credentials())
    return google_drive_service


GOOGLE_OAUTH_PORT = int(os.getenv("GOOGLE_OAUTH_PORT", "8765"))
# How long the loopback listener waits for the browser to come back before giving up and leaving
# the paste-back path as the only way to finish.
GOOGLE_OAUTH_LISTEN_SECONDS = 300

google_oauth_flow = None
google_oauth_result: Dict[str, Any] = {}


def google_oauth_redirect_uri() -> str:
    return f"http://localhost:{GOOGLE_OAUTH_PORT}/"


def start_google_oauth_listener() -> bool:
    """Serve the loopback redirect so a desktop browser can finish the flow by itself.

    This is best effort. On a headless host the user's browser is on another machine and can never
    reach this port, which is exactly why `google_auth_complete` exists — the two paths race, and
    whichever completes first wins. Returns whether the listener is actually accepting connections.
    """

    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - name fixed by BaseHTTPRequestHandler
            params = parse_qs(urlparse(self.path).query)
            code = params.get("code", [""])[0]
            error = params.get("error", [""])[0]
            if code:
                try:
                    finish_google_oauth(code)
                    body = "Google account connected. You can close this tab."
                except Exception as exc:  # pragma: no cover - depends on Google's response
                    body = f"Could not complete authorization: {exc}"
            else:
                body = f"Authorization failed: {error or 'no code returned'}"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *args):
            """Keep the HTTP access log out of the MCP server's stderr."""

    try:
        server = HTTPServer(("127.0.0.1", GOOGLE_OAUTH_PORT), CallbackHandler)
    except OSError as exc:
        logger.info("Google OAuth listener could not bind port %s: %s", GOOGLE_OAUTH_PORT, exc)
        return False

    server.timeout = GOOGLE_OAUTH_LISTEN_SECONDS

    def serve_once():
        try:
            server.handle_request()
        finally:
            server.server_close()

    threading.Thread(target=serve_once, daemon=True).start()
    return True


def finish_google_oauth(code: str) -> Dict[str, Any]:
    """Exchange an authorization code for a token and persist it."""
    global google_credentials, google_calendar_service, google_drive_service, google_oauth_flow

    if google_oauth_flow is None:
        raise ValueError("No authorization is in progress; call google_auth_start first")

    google_oauth_flow.fetch_token(code=code)
    credentials = google_oauth_flow.credentials
    with open(GOOGLE_TOKEN, "w", encoding="utf-8") as token_file:
        token_file.write(credentials.to_json())

    google_credentials = credentials
    # Drop cached clients so the next call builds them with the new token.
    google_calendar_service = None
    google_drive_service = None
    google_oauth_flow = None
    google_oauth_result.clear()
    google_oauth_result["connected"] = True
    return {"success": True, "message": "Google account connected."}


@mcp.tool()
def google_auth_start() -> Dict[str, Any]:
    """Begin Google (Calendar and Drive) authorization and return the URL to approve it.

    Works with or without a browser on this machine. If a browser is available it is opened and the
    flow finishes by itself; otherwise open the returned URL anywhere — a phone is fine — approve
    access, and send the address bar's URL to google_auth_complete. That page will fail to load,
    which is expected: the authorization code is in its query string.
    """
    global google_oauth_flow
    try:
        try:
            if get_google_credentials():
                return {
                    "success": True,
                    "already_connected": True,
                    "message": "Google is already authorized; nothing to do.",
                }
        except Exception:
            pass  # Not authorized yet, which is the whole point of this tool.

        if not os.path.exists(GOOGLE_CREDENTIALS):
            raise FileNotFoundError(
                f"Google OAuth credentials not found: {GOOGLE_CREDENTIALS}. Create an OAuth "
                "client of type 'Desktop app' in Google Cloud Console, enable the Calendar and "
                "Drive APIs, and save the downloaded file to that path."
            )

        flow = InstalledAppFlow.from_client_secrets_file(GOOGLE_CREDENTIALS, GOOGLE_SCOPES)
        flow.redirect_uri = google_oauth_redirect_uri()
        auth_url, _ = flow.authorization_url(access_type="offline", prompt="consent")
        google_oauth_flow = flow

        listening = start_google_oauth_listener()
        try:
            opened = webbrowser.open(auth_url)
        except Exception:
            opened = False

        return {
            "success": True,
            "auth_url": auth_url,
            "browser_opened": opened,
            "waiting_for_callback": listening,
            "message": (
                "A browser was opened; approve access there and this finishes by itself."
                if opened and listening
                else "Open this URL, approve access, then send the URL you land on (it will show "
                "a connection error) to google_auth_complete."
            ),
        }
    except Exception as exc:
        logger.warning("Could not start Google authorization: %s", exc)
        return {"success": False, "error": str(exc)}


@mcp.tool()
def google_auth_complete(redirect_url: str = "", code: str = "") -> Dict[str, Any]:
    """Finish Google authorization with the URL the browser landed on, or a bare code.

    Pass the whole address-bar URL from the page that failed to load after approving access
    (`http://localhost:.../?code=...`); a bare `code` value works too.
    """
    try:
        if google_oauth_result.get("connected"):
            return {"success": True, "message": "Google account already connected."}

        value = (code or "").strip()
        if not value and redirect_url.strip():
            query = parse_qs(urlparse(redirect_url.strip()).query)
            value = query.get("code", [""])[0]
            if not value:
                error = query.get("error", [""])[0]
                raise ValueError(
                    f"No authorization code in that URL{': ' + error if error else ''}"
                )
        if not value:
            raise ValueError("Provide the redirect_url you landed on, or the code from it")

        return finish_google_oauth(value)
    except Exception as exc:
        logger.warning("Could not complete Google authorization: %s", exc)
        return {"success": False, "error": str(exc)}


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
# Google Drive functionality
#

GOOGLE_NATIVE_EXPORTS = {
    "application/vnd.google-apps.document": ("text/plain", ".txt"),
    "application/vnd.google-apps.spreadsheet": ("text/csv", ".csv"),
    "application/vnd.google-apps.presentation": (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".pptx",
    ),
    "application/vnd.google-apps.drawing": ("image/png", ".png"),
}


def google_drive_error(exc: Exception) -> str:
    if isinstance(exc, RefreshError):
        return "Google authorization expired or was revoked. Delete token.json and reconnect."
    if isinstance(exc, HttpError):
        status = getattr(exc.resp, "status", None)
        if status == 401:
            return "Google Drive authentication failed. Reauthorize the account."
        if status == 403:
            return "Google Drive denied this operation. Enable the Drive API and check access."
        if status == 404:
            return "Google Drive file not found or not shared with this account."
        return f"Google Drive API error: {exc.reason}"
    return str(exc).strip() or exc.__class__.__name__


def run_google_drive(operation):
    try:
        return operation()
    except Exception as exc:
        logger.warning("Google Drive operation failed: %s", exc)
        return {"success": False, "error": google_drive_error(exc)}


def google_drive_file(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": item.get("id", ""),
        "name": item.get("name", ""),
        "mime_type": item.get("mimeType", ""),
        "size": int(item["size"]) if item.get("size") else None,
        "created_time": item.get("createdTime", ""),
        "modified_time": item.get("modifiedTime", ""),
        "web_view_link": item.get("webViewLink", ""),
        "parents": item.get("parents", []),
        "owners": [owner.get("emailAddress", "") for owner in item.get("owners", [])],
    }


DRIVE_FILE_FIELDS = (
    "id,name,mimeType,size,createdTime,modifiedTime,webViewLink,parents,"
    "owners(emailAddress)"
)


@mcp.tool()
def connect_drive() -> Dict[str, Any]:
    """Authenticate with Google Drive using the shared Google OAuth token."""
    def operation():
        get_google_drive_service().files().list(pageSize=1, fields="files(id)").execute()
        return {"success": True, "message": "Connected to Google Drive."}

    return run_google_drive(operation)


@mcp.tool()
def drive_health() -> Dict[str, Any]:
    """Check Google Drive authentication and API access."""
    def operation():
        about = get_google_drive_service().about().get(fields="user,storageQuota").execute()
        user = about.get("user", {})
        return {
            "healthy": True,
            "user": user.get("emailAddress", ""),
            "storage_quota": about.get("storageQuota", {}),
        }

    return run_google_drive(operation)


@mcp.tool()
def search_drive_files(
    query: str = "",
    limit: int = 20,
    mime_type: str | None = None,
    include_trashed: bool = False,
) -> Dict[str, Any]:
    """Search Google Drive files by name, full text, and optional MIME type."""
    def operation():
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        clauses = []
        if query.strip():
            escaped = query.replace("\\", "\\\\").replace("'", "\\'")
            clauses.append(f"(name contains '{escaped}' or fullText contains '{escaped}')")
        if mime_type:
            escaped_mime = mime_type.replace("'", "\\'")
            clauses.append(f"mimeType = '{escaped_mime}'")
        if not include_trashed:
            clauses.append("trashed = false")
        result = get_google_drive_service().files().list(
            q=" and ".join(clauses) if clauses else None,
            pageSize=limit,
            orderBy="modifiedTime desc",
            fields=f"nextPageToken,files({DRIVE_FILE_FIELDS})",
        ).execute()
        files = [google_drive_file(item) for item in result.get("files", [])]
        return {"success": True, "count": len(files), "files": files}

    return run_google_drive(operation)


@mcp.tool()
def get_drive_file(file_id: str) -> Dict[str, Any]:
    """Get metadata for one Google Drive file."""
    def operation():
        if not file_id.strip():
            raise ValueError("file_id is required")
        item = get_google_drive_service().files().get(
            fileId=file_id,
            fields=DRIVE_FILE_FIELDS,
        ).execute()
        return {"success": True, "file": google_drive_file(item)}

    return run_google_drive(operation)


@mcp.tool()
def read_drive_file(file_id: str, max_characters: int = 100000) -> Dict[str, Any]:
    """Read a Google Doc, Sheet, or text-based Drive file as text."""
    def operation():
        if not 1 <= max_characters <= 1000000:
            raise ValueError("max_characters must be between 1 and 1000000")
        service = get_google_drive_service()
        item = service.files().get(
            fileId=file_id,
            fields="id,name,mimeType,modifiedTime,webViewLink",
        ).execute()
        mime_type = item.get("mimeType", "")
        if mime_type == "application/vnd.google-apps.document":
            content = service.files().export(fileId=file_id, mimeType="text/plain").execute()
        elif mime_type == "application/vnd.google-apps.spreadsheet":
            content = service.files().export(fileId=file_id, mimeType="text/csv").execute()
        elif mime_type.startswith("text/") or mime_type in {
            "application/json",
            "application/xml",
        }:
            content = service.files().get_media(fileId=file_id).execute()
        else:
            raise ValueError(
                f"Unsupported readable MIME type: {mime_type}. Use download_drive_file instead."
            )
        text_content = content.decode("utf-8", errors="replace")
        return {
            "success": True,
            "file": google_drive_file(item),
            "content": text_content[:max_characters],
            "truncated": len(text_content) > max_characters,
        }

    return run_google_drive(operation)


@mcp.tool()
def download_drive_file(file_id: str, filename: str | None = None) -> Dict[str, Any]:
    """Download or export a Google Drive file to the configured download folder."""
    def operation():
        service = get_google_drive_service()
        item = service.files().get(fileId=file_id, fields="id,name,mimeType").execute()
        mime_type = item.get("mimeType", "")
        source_name = filename or item.get("name") or file_id
        if mime_type in GOOGLE_NATIVE_EXPORTS:
            export_type, extension = GOOGLE_NATIVE_EXPORTS[mime_type]
            request = service.files().export_media(fileId=file_id, mimeType=export_type)
            if not Path(source_name).suffix:
                source_name += extension
        elif mime_type.startswith("application/vnd.google-apps"):
            raise ValueError(f"Unsupported Google-native MIME type: {mime_type}")
        else:
            request = service.files().get_media(fileId=file_id)

        safe_name = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(source_name).name).strip()
        if not safe_name:
            raise ValueError("filename must contain valid characters")
        GOOGLE_DRIVE_DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)
        destination = GOOGLE_DRIVE_DOWNLOAD_PATH / safe_name
        buffer = BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        destination.write_bytes(buffer.getvalue())
        return {
            "success": True,
            "file_id": file_id,
            "path": str(destination.resolve()),
            "size": destination.stat().st_size,
        }

    return run_google_drive(operation)


#
# Email functionality
#

EMAIL_PROVIDER = os.getenv("EMAIL_PROVIDER", "gmail")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS", "")
EMAIL_SECRET = os.getenv("EMAIL_SECRET", "")
IMAP_HOST = os.getenv("IMAP_HOST", "imap.gmail.com")
IMAP_PORT = os.getenv("IMAP_PORT", "993")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = os.getenv("SMTP_PORT", "465")

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
        return (
            f"Missing environment variables: {', '.join(missing)}. "
            "Run `python setup_env.py` and configure Email."
        )
    for name, value in {"IMAP_PORT": IMAP_PORT, "SMTP_PORT": SMTP_PORT}.items():
        try:
            port = int(value)
        except ValueError:
            return f"{name} must be a number. Run `python setup_env.py` to fix it."
        if not 1 <= port <= 65535:
            return f"{name} must be between 1 and 65535."
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

    imap_port = int(IMAP_PORT)
    logger.info("Connecting to IMAP server %s:%s", IMAP_HOST, imap_port)
    connection = imaplib.IMAP4_SSL(IMAP_HOST, imap_port, timeout=30)
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


def normalize_email_uid(message_id: str | int) -> str:
    """Validate a message id and return it as a UID string.

    Ids are IMAP UIDs, which are stable for the life of the mailbox — unlike sequence numbers,
    which shift down whenever an earlier message is expunged, so a list of them collected before a
    delete would address the wrong messages afterwards.

    Accepts an int as well as a str because ids look numeric and callers (LLM tool calls in
    particular) routinely send them as JSON numbers. Rejecting anything but digits also keeps the
    value from being interpolated into an IMAP command as extra arguments.
    """
    text = str(message_id).strip()
    if not text:
        raise ValueError("message_id is required")
    if not text.isdigit():
        raise ValueError(f"message_id must be a numeric IMAP UID, got: {message_id!r}")
    return text


def fetch_email_message(message_id: str | int) -> Message:
    connection = open_email_imap()
    select_email_folder(connection)
    uid = normalize_email_uid(message_id)
    status, data = connection.uid("FETCH", uid, "(BODY.PEEK[])")
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


def set_email_seen_flag(message_id: str | int, seen: bool) -> Dict[str, Any]:
    try:
        connection = open_email_imap()
        select_email_folder(connection)
        uid = normalize_email_uid(message_id)
        operation = "+FLAGS" if seen else "-FLAGS"
        status, _ = connection.uid("STORE", uid, operation, "\\Seen")
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
    status, data = connection.uid(
        "FETCH",
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
        status, data = connection.uid("SEARCH", "ALL")
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
def read_email(message_id: str | int) -> Dict[str, Any]:
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
        status, data = connection.uid("SEARCH", *criteria)
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

        with smtplib.SMTP_SSL(SMTP_HOST, int(SMTP_PORT), timeout=30) as smtp:
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
def mark_read(message_id: str | int) -> Dict[str, Any]:
    """Mark an inbox message as read."""
    return set_email_seen_flag(message_id, True)


@mcp.tool()
def mark_unread(message_id: str | int) -> Dict[str, Any]:
    """Mark an inbox message as unread."""
    return set_email_seen_flag(message_id, False)


@mcp.tool()
def delete_email(message_id: str | int) -> Dict[str, Any]:
    """Permanently delete an inbox message using the IMAP Deleted flag."""
    try:
        connection = open_email_imap()
        select_email_folder(connection)
        uid = normalize_email_uid(message_id)
        status, _ = connection.uid("STORE", uid, "+FLAGS", "\\Deleted")
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
def list_attachments(message_id: str | int) -> Dict[str, Any]:
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
def summarize_email(message_id: str | int) -> Dict[str, Any]:
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
    def stop_server(signum, frame) -> None:
        close_email_imap()
        print("\nMCP server stopped.", file=sys.stderr, flush=True)
        os._exit(130)

    signal.signal(signal.SIGINT, stop_server)
    try:
        print("MCP server running. Press Ctrl+C to stop.", file=sys.stderr, flush=True)
        mcp.run()
    except KeyboardInterrupt:
        stop_server(signal.SIGINT, None)
    finally:
        close_email_imap()


# Allow direct execution of the server
if __name__ == "__main__":
    main()
