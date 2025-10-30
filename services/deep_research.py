from __future__ import annotations
import os
import re
import json
import time
from typing import Any, Dict, Optional, List

import requests


def _first_json(text: str) -> Optional[Dict[str, Any]]:
    # Try fenced JSON
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Greedy object
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _backoff(attempt: int) -> None:
    time.sleep(min(2.0 * (attempt + 1), 8.0))


class DeepResearchService:
    """
    Provider-agnostic wrapper (internally compatible with Grok's OpenAI-style API).
    - Uses chat completions with optional web_search tool and retries without tools if needed.
    - Falls back to a 'responses' endpoint shape if completions rejects with 4xx/5xx (e.g., 422).
    - Returns STRICT JSON payloads for research and for the IC memo.

    Configuration (env or pass to __init__):
    - RESEARCH_API_KEY (falls back to GROK_API_KEY)
    - RESEARCH_MODEL (default: grok-2-latest)
    - RESEARCH_API_URL (default: https://api.x.ai/v1/chat/completions)
    - RESEARCH_RESPONSES_URL (default: https://api.x.ai/v1/responses)
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, api_url: Optional[str] = None):
        key = api_key or os.getenv("RESEARCH_API_KEY") or os.getenv("GROK_API_KEY", "")
        self.api_key = key
        self.model = model or os.getenv("RESEARCH_MODEL", "grok-2-latest")
        self.api_url = api_url or os.getenv("RESEARCH_API_URL", "https://api.x.ai/v1/chat/completions")
        self.responses_url = os.getenv("RESEARCH_RESPONSES_URL", "https://api.x.ai/v1/responses")
        self.enabled = bool(self.api_key)
        self._session = requests.Session()

    # ---------- Low-level callers with retries/fallbacks ----------
    def _post(self, url: str, payload: Dict[str, Any]) -> requests.Response:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        return self._session.post(url, headers=headers, data=json.dumps(payload), timeout=120)

    def _chat_request(self, messages: List[dict], use_tools: bool) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 2400,
        }
        if use_tools:
            payload["tools"] = [{"type": "web_search"}]
            payload["tool_choice"] = "auto"
        resp = self._post(self.api_url, payload)
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except Exception:
            return json.dumps(data)

    def _responses_request(self, messages: List[dict]) -> str:
        payload = {
            "model": self.model,
            "input": messages,  # many deployments accept this 'input' shape
            "temperature": 0.2,
            "max_output_tokens": 2400,
        }
        resp = self._post(self.responses_url, payload)
        resp.raise_for_status()
        data = resp.json()
        # Try common shapes
        for k in ("output_text", "content", "response"):
            v = data.get(k)
            if isinstance(v, str) and v.strip():
                return v
        try:
            return data["choices"][0]["message"]["content"] or ""
        except Exception:
            return json.dumps(data)

    def _chat_with_retries(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_error = ""
        # 1) With tools
        for attempt in range(2):
            try:
                return self._chat_request(messages, use_tools=True)
            except requests.HTTPError as e:
                last_error = f"chat+tools[{e.response.status_code}]"
                if e.response.status_code in (400, 404, 422, 429, 500, 503):
                    _backoff(attempt)
                    continue
                raise
            except Exception as e:
                last_error = f"chat+tools[{e}]"
                _backoff(attempt)
                continue
        # 2) Without tools
        for attempt in range(2):
            try:
                return self._chat_request(messages, use_tools=False)
            except Exception as e:
                last_error = f"chat-no-tools[{e}]"
                _backoff(attempt)
                continue
        # 3) Responses fallback
        for attempt in range(2):
            try:
                return self._responses_request(messages)
            except Exception as e:
                last_error = f"responses[{e}]"
                _backoff(attempt)
                continue
        # Fail safe
        return json.dumps({"error": f"external_research_unavailable: {last_error}"})

    # ---------- Public API ----------
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
            "You are a senior VC analyst with live web capabilities. "
            "Perform an exhaustive deep dive across X (Twitter), official sites/docs, engineering blogs, job boards, "
            "filings, reputable news, funding databases, app stores, developer forums, and other public sources. "
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

        content = self._chat_with_retries(system, user)
        data = _first_json(content or "") or {}
        # Sanitize
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

    def build_ic_memo(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ask the model to produce a deeply detailed IC memo using:
        - profile (company, sector, stage),
        - derived KPIs (SSQ, Rule of 40, burn multiple, ARR in USD, churn),
        - research summary/sections/sources,
        - valuation range (USD).
        Returns STRICT JSON with the same memo schema as 'research'.
        """
        if not self.enabled:
            return {}

        system = (
            "You are a veteran VC partner preparing an Investment Committee memo. "
            "Use the supplied quantitative KPIs and the research bundle to produce a crisp, specific, and defensible memo. "
            "Cite data inline using [n] tags corresponding to 'sources' if helpful. Output STRICT JSON only."
        )
        user = (
            "Context follows as JSON. Produce ONLY JSON for the memo with fields: "
            "executive_summary, investment_thesis, market, product, traction, unit_economics, gtm, competition, team, "
            "risks, catalysts, round_dynamics, use_of_proceeds, valuation_rationale, kpis_next_12m, exit_paths, "
            "recommendation, conviction.\n\n"
            f"{json.dumps(context)[:18000]}"
        )

        content = self._chat_with_retries(system, user)
        memo = _first_json(content or "") or {}
        if not isinstance(memo, dict):
            return {}
        return memo
