"""
Tavily API Client Wrapper
Provides interface for web search with domain filtering.
"""
import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class TavilyClient:
    """Wrapper for Tavily API with robust error handling."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        
        if not self.api_key:
            logger.warning("TAVILY_API_KEY not set, Tavily client will not work")
    
    def search(
        self,
        query: str,
        include_domains: Optional[List[str]] = None,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform web search with optional domain filtering.
        
        Args:
            query: Search query string
            include_domains: List of domains to include in search
            max_results: Maximum number of results to return
            
        Returns:
            List of search results with 'title', 'url', 'content', 'score'
        """
        if not self.api_key:
            logger.warning("Cannot perform search: API key not available")
            return []
        
        try:
            import requests
            
            url = "https://api.tavily.com/search"
            
            payload = {
                "api_key": self.api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "advanced",
                "include_answer": False,
                "include_raw_content": False
            }
            
            if include_domains:
                payload["include_domains"] = include_domains
            
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            results = data.get('results', [])
            
            logger.info(f"Tavily search returned {len(results)} results for query: {query}")
            
            return results
            
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return []
    
    def search_news(
        self,
        query: str,
        include_domains: Optional[List[str]] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for news articles.
        
        Args:
            query: Search query
            include_domains: News domains to include
            max_results: Maximum results
            
        Returns:
            List of news results
        """
        # Add time filter for recent news
        query_with_time = f"{query} (recent OR latest OR 2024 OR 2025)"
        
        return self.search(
            query=query_with_time,
            include_domains=include_domains,
            max_results=max_results
        )
