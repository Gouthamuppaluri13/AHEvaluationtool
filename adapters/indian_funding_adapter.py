from typing import Dict, Any, List
from datetime import datetime
import re

from adapters.tavily_client import TavilyClient

ROUND_KEYWORDS = {
    "Pre-Seed": ["pre-seed", "pre seed"],
    "Seed": ["seed round", "seed funding", "seed"],
    "Series A": ["series a"],
    "Series B": ["series b"],
    "Series C": ["series c"],
    "Series D+": ["series d", "series e", "series f", "series g"]
}

SOURCES = [
    "inc42.com",
    "yourstory.com",
    "entrackr.com",
    "economicstimes.indiatimes.com",
    "moneycontrol.com",
    "livemint.com",
    "business-standard.com",
    "trak.in",
    "startupindia.gov.in"
]

class IndianFundingAdapter:
    def __init__(self, tavily: TavilyClient):
        self.tavily = tavily

    def _detect_stage(self, text: str) -> str:
        t = text.lower()
        for stage, kws in ROUND_KEYWORDS.items():
            if any(kw in t for kw in kws):
                return stage
        return "Undisclosed"

    def _extract_amount(self, text: str) -> float:
        t = text.lower().replace(",", "")
        match = re.search(r'(\$|usd|rs\.?|₹)?\s?([\d\.]+)\s?(m|mn|million|cr|crore|b|bn|billion)?', t)
        if not match:
            return 0.0
        _, num, unit = match.groups()
        val = float(num)
        unit = (unit or "").lower()
        if unit in ["m", "mn", "million"]:
            return val * 1_000_000
        if unit in ["b", "bn", "billion"]:
            return val * 1_000_000_000
        if unit in ["cr", "crore"]:
            return val * 10_000_000
        return val

    def get_funding_trends(self, sector: str, country: str = "India", months: int = 24) -> Dict[str, Any]:
        q = f'{sector} startup funding {country} last {months} months'
        res = self.tavily.search(q, include_domains=SOURCES, max_results=25)
        items = res.get("results", [])
        if not items:
            return {"error": res.get("error", "No results"), "round_counts_by_stage": {}, "median_round_size_inr": None, "top_investors": [], "recent_rounds": []}

        rounds: List[Dict[str, Any]] = []
        stage_counts: Dict[str, int] = {}
        amounts_inr: List[float] = []
        investor_freq: Dict[str, int] = {}

        def parse_date(s):
            try:
                return datetime.fromisoformat(s.replace("Z", "+00:00"))
            except Exception:
                return None

        for it in items:
            title = it.get("title", "")
            snippet = it.get("content", "") or it.get("snippet", "")
            url = it.get("url", "")
            published = parse_date(it.get("published_date", "")) if it.get("published_date") else None

            stage = self._detect_stage(f"{title} {snippet}")
            amt = self._extract_amount(f"{title} {snippet}")
            inv = re.findall(r'led by ([A-Za-z0-9\-\&\s,]+)', f"{title}. {snippet}.")
            inv = [x.strip() for x in inv[0].split(",")] if inv else []

            stage_counts[stage] = stage_counts.get(stage, 0) + 1
            if amt > 0:
                amounts_inr.append(amt)
            for i in inv:
                if i:
                    investor_freq[i] = investor_freq.get(i, 0) + 1

            rounds.append({
                "headline": title,
                "url": url,
                "stage": stage,
                "approx_amount": amt,
                "published_at": published.isoformat() if published else None
            })

        amounts_inr_sorted = sorted([a for a in amounts_inr if a > 0])
        median = amounts_inr_sorted[len(amounts_inr_sorted)//2] if amounts_inr_sorted else None
        top_investors = sorted(investor_freq.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "round_counts_by_stage": stage_counts,
            "median_round_size_inr": median,
            "top_investors": [{"name": k, "mentions": v} for k, v in top_investors],
            "recent_rounds": rounds[:15]
        }
