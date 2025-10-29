"""
News Adapter
Aggregates recent news for companies and sectors.
"""
import logging
from typing import List, Dict, Any, Optional
from adapters.tavily_client import TavilyClient

logger = logging.getLogger(__name__)


class NewsAdapter:
    """Aggregates news for semantic enrichment."""
    
    def __init__(self, tavily_client: Optional[TavilyClient] = None):
        self.tavily = tavily_client or TavilyClient()
    
    def get_company_news(
        self,
        company_name: str,
        sector: Optional[str] = None,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get recent news about a company.
        
        Args:
            company_name: Company name
            sector: Optional sector for context
            max_results: Maximum results
            
        Returns:
            List of news items with 'title', 'url', 'content', 'published_date'
        """
        query = f"{company_name}"
        if sector:
            query += f" {sector}"
        query += " India startup news"
        
        results = self.tavily.search_news(
            query=query,
            max_results=max_results
        )
        
        # Format results
        news_items = []
        for result in results:
            news_items.append({
                'title': result.get('title', ''),
                'url': result.get('url', ''),
                'content': result.get('content', ''),
                'score': result.get('score', 0)
            })
        
        logger.info(f"Retrieved {len(news_items)} news items for {company_name}")
        
        return news_items
    
    def get_sector_news(
        self,
        sector: str,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get recent news about a sector.
        
        Args:
            sector: Sector name
            max_results: Maximum results
            
        Returns:
            List of news items
        """
        query = f"India {sector} sector trends news 2024 2025"
        
        results = self.tavily.search_news(
            query=query,
            max_results=max_results
        )
        
        # Format results
        news_items = []
        for result in results:
            news_items.append({
                'title': result.get('title', ''),
                'url': result.get('url', ''),
                'content': result.get('content', ''),
                'score': result.get('score', 0)
            })
        
        logger.info(f"Retrieved {len(news_items)} news items for sector {sector}")
        
        return news_items
