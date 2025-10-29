"""
Data Enrichment Service
Orchestrates Tavily-based trends/news and sector context enrichment.
"""
import os
import json
import logging
from typing import Dict, Any, Optional
from adapters.tavily_client import TavilyClient
from adapters.indian_funding_adapter import IndianFundingAdapter
from adapters.news_adapter import NewsAdapter

logger = logging.getLogger(__name__)


class DataEnrichmentService:
    """Orchestrates data enrichment from multiple sources."""
    
    def __init__(self, tavily_api_key: Optional[str] = None):
        self.tavily = TavilyClient(api_key=tavily_api_key)
        self.funding_adapter = IndianFundingAdapter(tavily_client=self.tavily)
        self.news_adapter = NewsAdapter(tavily_client=self.tavily)
        self.india_index = self._load_india_index()
    
    def _load_india_index(self) -> Optional[Dict]:
        """Load India funding index from data directory."""
        index_path = "data/india_funding_index.json"
        
        if not os.path.exists(index_path):
            logger.info(f"India funding index not found at {index_path}")
            return None
        
        try:
            with open(index_path, 'r') as f:
                index = json.load(f)
            logger.info(f"Loaded India funding index with {len(index)} sectors")
            return index
        except Exception as e:
            logger.error(f"Failed to load India funding index: {e}")
            return None
    
    def enrich(
        self,
        company_name: str,
        sector: str,
        product_description: str
    ) -> Dict[str, Any]:
        """
        Perform comprehensive data enrichment.
        
        Args:
            company_name: Company name
            sector: Primary sector
            product_description: Product description
            
        Returns:
            Dictionary with enrichment data:
            - indian_funding_trends: News-derived funding trends
            - recent_news: Company and sector news
            - india_funding_dataset_context: Kaggle-derived context
        """
        enrichment = {
            'indian_funding_trends': {},
            'recent_news': [],
            'india_funding_dataset_context': {}
        }
        
        # Get India funding trends from news
        try:
            trends = self.funding_adapter.get_sector_funding_trends(sector)
            enrichment['indian_funding_trends'] = trends
            logger.info(f"Enriched with India funding trends for {sector}")
        except Exception as e:
            logger.error(f"Failed to get funding trends: {e}")
        
        # Get recent news
        try:
            company_news = self.news_adapter.get_company_news(
                company_name=company_name,
                sector=sector,
                max_results=3
            )
            sector_news = self.news_adapter.get_sector_news(
                sector=sector,
                max_results=3
            )
            enrichment['recent_news'] = company_news + sector_news
            logger.info(f"Enriched with {len(enrichment['recent_news'])} news items")
        except Exception as e:
            logger.error(f"Failed to get news: {e}")
        
        # Get India funding dataset context
        if self.india_index:
            try:
                # Try to find matching sector in index
                sector_context = self._find_sector_context(sector)
                if sector_context:
                    enrichment['india_funding_dataset_context'] = sector_context
                    logger.info(f"Added India funding dataset context for {sector}")
            except Exception as e:
                logger.error(f"Failed to get dataset context: {e}")
        
        return enrichment
    
    def _find_sector_context(self, sector: str) -> Optional[Dict[str, Any]]:
        """Find matching sector in India funding index."""
        if not self.india_index:
            return None
        
        # Direct match
        if sector in self.india_index:
            return self.india_index[sector]
        
        # Fuzzy match - check if sector name is substring
        sector_lower = sector.lower()
        for index_sector, data in self.india_index.items():
            if sector_lower in index_sector.lower() or index_sector.lower() in sector_lower:
                return data
        
        # Return None if no match
        return None
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """Get summary of India funding dataset."""
        if not self.india_index:
            return {'available': False}
        
        total_sectors = len(self.india_index)
        total_rounds = sum(
            data.get('rounds_total', 0)
            for data in self.india_index.values()
        )
        
        return {
            'available': True,
            'total_sectors': total_sectors,
            'total_rounds': total_rounds
        }
