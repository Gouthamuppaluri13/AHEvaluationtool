from __future__ import annotations
import os
import re
import json
from typing import Any, Dict, Optional

import requests

DEFAULT_GROK_MODEL = os.getenv("GROK_MODEL", "grok-2-latest")
DEFAULT_GROK_API_URL = os.getenv("GROK_API_URL", "https://api.x.ai/v1/chat/completions")

class GrokResearchService:
    """
    Wrapper for xAI Grok chat completions to perform a comprehensive startup deep dive.

    What it does:
    - Instructs Grok to deeply research across X (Twitter), the broader web, filings,
      reputable news, funding databases, job boards, engineering blogs, etc.
    - Returns STRICT JSON with:
        - summary (concise overview)
        - sections (overview/products/business_model/funding/.../tech_stack)
        - sources (title, url, snippet, confidence)
        - memo: executive_summary, bull_case_narrative, bear_case_narrative, recommendation, conviction
    - Uses an OpenAI-compatible endpoint shape (xAI mirrors it).
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, api_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROK_API_KEY", "")
        self.model = model or DEFAULT_GROK_MODEL
        self.api_url = api_url or DEFAULT_GROK_API_URL
        self.enabled = bool(self.api_key)
        self._session = requests.Session()

    def _request(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 2200) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Some Grok deployments support tools:web_search (OpenAI-compatible). If unsupported, server ignores.
            "tools": [{"type": "web_search"}],
            "tool_choice": "auto",
        }
        resp = self._session.post(self.api_url, headers=headers, data=json.dumps(payload), timeout=120)
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except Exception:
            return json.dumps(data)

    def research(self, company: str, sector: str, location: str, description: str) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "notice": "Grok API key not configured. Set GROK_API_KEY in secrets or environment.",
                "summary": "",
                "sections": {},
                "sources": [],
                "memo": {},
            }

        system = (
            "You are Grok, an expert startup analyst with live web access. "
            "Do an exhaustive deep dive on the startup using every public source you can: "
            "X (Twitter), company and product sites, docs, engineering blogs, job boards, "
            "press releases, reputable news, funding databases, and filings. "
            "Normalize currencies/units, include recent dates, avoid speculation, and PROVIDE STRICT JSON ONLY."
        )
        user = f"""
Research the startup thoroughly and produce a structured, citation-rich deep dive and an investment memo.
You MUST browse and synthesize across X and the web. Return ONLY JSON with the following shape:

{{
  "summary": "6–10 sentences capturing product, market, traction, milestones, unit-economics hints, and open questions.",
  "sections": {{
    "overview": "...",
    "products": "...",
    "business_model": "...",
    "funding": "...",
    "investors": "...",
    "leadership": "...",
    "traction": "...",
    "customers": "...",
    "competitors": "...",
    "moat": "...",
    "partnerships": "...",
    "risks": "...",
    "regulatory": "...",
    "controversies": "...",
    "hiring": "...",
    "tech_stack": "..."
  }},
  "sources": [
    {{"title": "Source Title", "url": "https://...", "snippet": "1–2 sentences with date if relevant", "confidence": 0.0}}
  ],
  "memo": {{
    "executive_summary": "A crisp summary tailored for VC IC doc.",
    "bull_case_narrative": "Specific upside levers, catalysts, and conditions for outperformance.",
    "bear_case_narrative": "Key failure modes and downside scenarios.",
    "recommendation": "Invest|Watchlist|Pass",
    "conviction": "High|Medium|Low"
  }}
}}

Entity:
- Name: {company}
- Sector: {sector}
- Location: {location}
- Description: {description}

Rules:
- STRICT JSON ONLY. No prose outside the JSON object.
- Include 10–20 diverse sources with URLs and short snippets (with dates where relevant). Confidence 0–1.
- If a detail is unknown, leave an empty string instead of guessing.
- Prefer info from the last 12–24 months.
""".strip()

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        try:
            content = self._request(messages)
            m = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if not m:
                raise ValueError("No JSON object found in Grok response.")
            data = json.loads(m.group(0))
            if not isinstance(data, dict):
                raise ValueError("Grok returned a non-dict JSON payload.")
            # Defaults + light sanitation
            data.setdefault("summary", "")
            data.setdefault("sections", {})
            data.setdefault("sources", [])
            data.setdefault("memo", {})
            if not isinstance(data["sections"], dict):
                data["sections"] = {}
            if not isinstance(data["sources"], list):
                data["sources"] = []
            if not isinstance(data["memo"], dict):
                data["memo"] = {}
            return data
        except Exception as e:
            return {
                "error": f"Grok research failed: {e}",
                "summary": "",
                "sections": {},
                "sources": [],
                "memo": {},
            }
