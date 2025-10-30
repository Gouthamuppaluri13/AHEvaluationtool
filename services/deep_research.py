from __future__ import annotations
import os
import re
import json
from typing import Any, Dict, Optional

import requests


def _first_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the first JSON object from a string.
    """
    # Try a fenced block first
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Fallback: greedy match of outer object
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


class DeepResearchService:
    """
    Generic web/X research wrapper (provider-agnostic) that returns a comprehensive,
    citation-rich deep dive and an IC-grade memo.

    Configuration (env or pass to __init__):
    - RESEARCH_API_KEY (fallback to GROK_API_KEY to keep current deployments working)
    - RESEARCH_MODEL (default: grok-2-latest)
    - RESEARCH_API_URL (default: https://api.x.ai/v1/chat/completions)
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, api_url: Optional[str] = None):
        key = api_key or os.getenv("RESEARCH_API_KEY") or os.getenv("GROK_API_KEY", "")
        self.api_key = key
        self.model = model or os.getenv("RESEARCH_MODEL", "grok-2-latest")
        self.api_url = api_url or os.getenv("RESEARCH_API_URL", "https://api.x.ai/v1/chat/completions")
        self._responses_url = os.getenv("RESEARCH_RESPONSES_URL", "https://api.x.ai/v1/responses")
        self.enabled = bool(self.api_key)
        self._session = requests.Session()

    def _post(self, url: str, payload: Dict[str, Any]) -> requests.Response:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        return self._session.post(url, headers=headers, data=json.dumps(payload), timeout=120)

    def _chat_try(self, system: str, user: str, with_tools: bool) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": 2200,
        }
        if with_tools:
            payload["tools"] = [{"type": "web_search"}]
            payload["tool_choice"] = "auto"

        resp = self._post(self.api_url, payload)
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except Exception:
            return json.dumps(data)

    def _responses_try(self, system: str, user: str) -> str:
        """
        Fallback to a responses-style endpoint some deployments expose.
        """
        payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_output_tokens": 2200,
        }
        resp = self._post(self._responses_url, payload)
        resp.raise_for_status()
        data = resp.json()
        # Try common shapes
        for k in ("output_text", "content", "response"):
            v = data.get(k)
            if isinstance(v, str) and v.strip():
                return v
        # Try choices-like
        try:
            return data["choices"][0]["message"]["content"] or ""
        except Exception:
            return json.dumps(data)

    def research(self, company: str, sector: str, location: str, description: str) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "notice": "External research service not configured.",
                "summary": "",
                "sections": {},
                "sources": [],
                "memo": {},
            }

        system = (
            "You are an expert VC analyst with live web capabilities. "
            "Perform an exhaustive deep dive across X (Twitter), official sites/docs, engineering blogs, "
            "job boards, filings, reputable news, funding databases, and other public sources. "
            "Normalize currencies/units with dates; avoid speculation. Return STRICT JSON only."
        )

        user = f"""
Research the startup thoroughly and return ONLY JSON with this schema:

{{
  "summary": "6–10 sentences capturing product, market, traction, milestones, unit-economics hints, and open questions.",
  "sections": {{
    "overview": "...",
    "products": "...",
    "business_model": "...",
    "gtm": "...",
    "unit_economics": "...",
    "funding": "...",
    "investors": "...",
    "leadership": "...",
    "hiring": "...",
    "traction": "...",
    "customers": "...",
    "pricing": "...",
    "competitors": "...",
    "moat": "...",
    "partnerships": "...",
    "regulatory": "...",
    "risks": "...",
    "tech_stack": "...",
    "roadmap": "..."
  }},
  "sources": [
    {{"title": "Source Title", "url": "https://...", "snippet": "1–2 sentences with date if relevant", "confidence": 0.0}}
  ],
  "memo": {{
    "executive_summary": "Crisp IC-ready summary",
    "investment_thesis": "Why now/why this team/why this wedge",
    "market": "Market definition, TAM/SAM/SOM with sources, trend vectors",
    "product": "Product depth, differentiation, defensibility",
    "traction": "Logos, usage, revenue signals, pipeline",
    "unit_economics": "Gross margin, LTV/CAC, payback, burn multiple (if public/estimable)",
    "gtm": "Channels, sales motion, sales cycle, pricing, expansion",
    "competition": "Landscape, relative strengths, switching costs",
    "team": "Founders, execs, org, critical hires",
    "risks": "Top risks and mitigations",
    "catalysts": "Milestones/catalysts next 12–18 months",
    "round_dynamics": "Stage, size, valuation markers if public, prior investors",
    "use_of_proceeds": "Planned allocation",
    "valuation_rationale": "Comparable ranges/method with sources",
    "kpis_next_12m": "Key KPIs/OKRs with target ranges",
    "exit_paths": "Potential acquirers and IPO viability",
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
- STRICT JSON ONLY; no prose outside the JSON object.
- Include 10–20 diverse sources (URLs + two-sentence snippets with dates). Confidence 0–1.
- If unknown, leave fields empty. Favor information from last 12–24 months.
""".strip()

        # Attempt with tools -> without tools -> responses fallback
        content = ""
        try:
            content = self._chat_try(system, user, with_tools=True)
        except requests.HTTPError as e:
            code = getattr(e.response, "status_code", None)
            if code in (400, 404, 422):
                # Retry without tools
                try:
                    content = self._chat_try(system, user, with_tools=False)
                except requests.HTTPError:
                    # Final fallback to responses endpoint shape
                    content = self._responses_try(system, user)
            else:
                # Other network error: last chance fallback to responses
                try:
                    content = self._responses_try(system, user)
                except Exception as _:
                    raise
        except Exception:
            try:
                content = self._responses_try(system, user)
            except Exception as _:
                content = ""

        data = _first_json(content or "") or {}
        # Defaults + sanitation
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
