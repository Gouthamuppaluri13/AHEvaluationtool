from typing import Dict, Any
from adapters.tavily_client import TavilyClient

class NewsAdapter:
    def __init__(self, tavily: TavilyClient):
        self.tavily = tavily

    def get_recent_news(self, company: str, sector: str, country: str = "India", max_items: int = 15) -> Dict[str, Any]:
        q = f'{company} {sector} India startup news last 12 months'
        res = self.tavily.search(q, max_results=max_items)
        results = res.get("results", [])
        if not results:
            return {"news": [], "error": res.get("error", "No news found")}
        news = []
        for r in results:
            news.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", "") or r.get("snippet", ""),
                "published_date": r.get("published_date", "")
            })
        return {"news": news}
