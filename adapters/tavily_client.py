import os
import requests
from typing import List, Dict, Any, Optional

class TavilyClient:
    def __init__(self, api_key: Optional[str] = None, timeout: int = 25):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY", "")
        self.timeout = timeout
        self.endpoint = "https://api.tavily.com/search"

    def search(self, query: str, search_depth: str = "advanced", include_domains: Optional[list] = None, max_results: int = 10) -> Dict[str, Any]:
        if not self.api_key:
            return {"results": [], "error": "Missing Tavily API key"}
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results
        }
        if include_domains:
            payload["include_domains"] = include_domains
        try:
            resp = requests.post(self.endpoint, json=payload, timeout=self.timeout)
            if resp.status_code == 401:
                headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                resp = requests.post(self.endpoint, json={k: v for k, v in payload.items() if k != "api_key"}, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return {
                "results": data.get("results", []),
                "query": query
            }
        except Exception as e:
            return {"results": [], "error": str(e), "query": query}
