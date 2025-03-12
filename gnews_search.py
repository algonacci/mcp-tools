from mcp.server.fastmcp import FastMCP, Context
from typing import List, Dict, Any
import gnews
from dataclasses import dataclass
from contextlib import asynccontextmanager
from typing import AsyncIterator

# Define server context for GNews client
@dataclass
class GNewsContext:
    default_client: gnews.GNews

@asynccontextmanager
async def gnews_lifespan(server: FastMCP) -> AsyncIterator[GNewsContext]:
    """Initialize GNews client on startup"""
    # Initialize default GNews client
    default_client = gnews.GNews()
    
    try:
        yield GNewsContext(default_client=default_client)
    finally:
        pass  # No cleanup needed for GNews

# Configure FastMCP with dependencies and lifespan
mcp = FastMCP(
    "GNews Search", 
    dependencies=["gnews"],
    lifespan=gnews_lifespan
)

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
    
    Args:
        language: Language code (e.g., 'en', 'id', 'es', 'fr')
        country: Country code (e.g., 'US', 'ID', 'UK', 'CA')
        max_results: Maximum number of results to return
        period: Time period (None for all time, 'd' for day, 'h' for hour, 'm' for month)
        proxy: Proxy server to use for requests
        exclude_websites: List of websites to exclude from results
        
    Returns:
        Configured GNews client
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
    # Validate topic
    valid_topics = ['world', 'nation', 'business', 'technology', 'entertainment', 'sports', 'science', 'health']
    
    # Create a new client with the specified parameters
    gn = create_gnews_client(
        language=language,
        country=country,
        max_results=max_results,
        proxy=proxy,
        exclude_websites=exclude_websites
    )
    
    # Report progress
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

# Allow direct execution of the server
if __name__ == "__main__":
    mcp.run()