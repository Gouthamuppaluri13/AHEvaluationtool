from typing import Dict, Any
import json
import os

from adapters.tavily_client import TavilyClient
from adapters.indian_funding_adapter import IndianFundingAdapter
from adapters.news_adapter import NewsAdapter

class DataEnrichmentService:
    def __init__(self, tavily_key: str, india_index_path: str = "data/india_funding_index.json"):
        self.tavily = TavilyClient(tavily_key)
        self.funding = IndianFundingAdapter(self.tavily)
        self.news = NewsAdapter(self.tavily)
        self.india_index_path = india_index_path
        self.india_index = None
        if os.path.exists(self.india_index_path):
            try:
                with open(self.india_index_path, "r", encoding="utf-8") as f:
                    self.india_index = json.load(f)
            except Exception:
                self.india_index = None

    def _sector_context(self, sector: str) -> Dict[str, Any]:
        if not self.india_index:
            return {}
        sectors = self.india_index.get("sectors", {})
        keys = list(sectors.keys())
        if not keys:
            return {}
        if sector in sectors:
            return sectors[sector]
        lo = sector.lower()
        for k in keys:
            if k.lower() == lo:
                return sectors[k]
        for k in keys:
            if lo in k.lower() or k.lower() in lo:
                return sectors[k]
        return {}

    def enrich(self, sector: str, company: str) -> Dict[str, Any]:
        india_funding = self.funding.get_funding_trends(sector)
        recent_news = self.news.get_recent_news(company, sector)
        india_dataset_context = self._sector_context(sector)
        return {
            "indian_funding_trends": india_funding,
            "recent_news": recent_news,
            "india_funding_dataset_context": india_dataset_context
        }
