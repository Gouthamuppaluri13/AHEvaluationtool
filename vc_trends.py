"""
Static 2025 VC trend proxies (update quarterly or via RSS/CSV cron).

- ai_dominance: share of VC dollars to AI startups (global proxy)
- esg_weight: qualitative importance of ESG/impact in diligence
- exit_rate: rebound in exits (M&A/IPO) → affects multiples optimism
"""
from __future__ import annotations

from typing import Dict, List

VC_TRENDS: Dict[str, float] = {
    "ai_dominance": 0.62,  # 62% funding trend for AI/GenAI
    "esg_weight": 0.15,
    "exit_rate": 0.25,
}

def trend_relevance(response: str, trends: Dict[str, float] | None = None) -> float:
    """
    Heuristic relevance: keyword presence weighted by trend strengths.
    """
    t = trends or VC_TRENDS
    keys: List[tuple[str, float]] = [
        ("AI", t.get("ai_dominance", 0.0)),
        ("GenAI", t.get("ai_dominance", 0.0) * 0.9),
        ("ESG", t.get("esg_weight", 0.0)),
        ("sustainability", t.get("esg_weight", 0.0) * 0.8),
        ("exit", t.get("exit_rate", 0.0) * 0.6),
        ("acquisition", t.get("exit_rate", 0.0) * 0.7),
        ("IPO", t.get("exit_rate", 0.0) * 0.8),
    ]
    low = response.lower()
    hits = [(1.0 if k.lower() in low else 0.0) * float(w) for k, w in keys]
    return float(sum(hits) / max(1, len(hits)))
