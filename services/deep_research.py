from __future__ import annotations
import os
import re
import json
import time
from typing import Any, Dict, Optional, List, Tuple
import requests
from urllib.parse import urlparse


def _first_json(text: str) -> Optional[Dict[str, Any]]:
    # Fenced JSON first
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Greedy object fallback
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _sleep_backoff(attempt: int) -> None:
    time.sleep(min(1.5 * (attempt + 1), 6.0))


def _normalize_sources(sources: Any, max_n: int = 20) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    if not isinstance(sources, list):
        return out
    for s in sources:
        if not isinstance(s, dict):
            continue
        title = str(s.get("title", "")).strip() or "Source"
        url = str(s.get("url", "")).strip()
        if not url.startswith("http"):
            continue
        try:
            host = urlparse(url).netloc.lower()
        except Exception:
            host = url
        key = f"{host}{url}"
        if key in seen:
            continue
        seen.add(key)
        snippet = str(s.get("snippet", "")).strip()
        conf = s.get("confidence", None)
        if isinstance(conf, (int, float)):
            conf_f = float(conf)
        else:
            conf_f = 0.5
        out.append({"title": title, "url": url, "snippet": snippet, "confidence": conf_f})
        if len(out) >= max_n:
            break
    return out


class DeepResearchService:
    """
    Provider-agnostic wrapper (Grok-compatible, OpenAI-style).
    - Multi-pass: chat+web_search → chat(no tools) → responses endpoint.
    - Strict JSON for research and memo, with source normalization/deduplication.
    - Gracefully degrades with neutral messages (no provider exposed in UI).

    Config (env or __init__):
    - RESEARCH_API_KEY (falls back to GROK_API_KEY)
    - RESEARCH_MODEL (default: grok-2-latest)
    - RESEARCH_API_URL (default: https://api.x.ai/v1/chat/completions)
    - RESEARCH_RESPONSES_URL (default: https://api.x.ai/v1/responses)
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, api_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("RESEARCH_API_KEY") or os.getenv("GROK_API_KEY", "")
        self.model = model or os.getenv("RESEARCH_MODEL", "grok-2-latest")
        self.api_url = api_url or os.getenv("RESEARCH_API_URL", "https://api.x.ai/v1/chat/completions")
        self.responses_url = os.getenv("RESEARCH_RESPONSES_URL", "https://api.x.ai/v1/responses")
        self.enabled = bool(self.api_key)
        self._session = requests.Session()

    # ---------- low-level HTTP ----------
    def _post(self, url: str, payload: Dict[str, Any]) -> requests.Response:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        return self._session.post(url, headers=headers, data=json.dumps(payload), timeout=120)

    def _chat_request(self, messages: List[dict], use_tools: bool) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 3000,
            "stream": False,
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
            "input": messages,
            "temperature": 0.2,
            "max_output_tokens": 3000,
        }
        resp = self._post(self.responses_url, payload)
        resp.raise_for_status()
        data = resp.json()
        for k in ("output_text", "content", "response"):
            v = data.get(k)
            if isinstance(v, str) and v.strip():
                return v
        try:
            return data["choices"][0]["message"]["content"] or ""
        except Exception:
            return json.dumps(data)

    def _chat_with_retries(self, system: str, user: str) -> Tuple[str, str]:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last_err = ""

        for i in range(2):
            try:
                return self._chat_request(messages, use_tools=True), "chat+tools"
            except requests.HTTPError as e:
                code = getattr(e.response, "status_code", None)
                last_err = f"chat+tools[{code}]"
                if code in (400, 404, 409, 422, 429, 500, 503):
                    _sleep_backoff(i)
                    continue
                raise
            except Exception as e:
                last_err = f"chat+tools[{e}]"
                _sleep_backoff(i)
                continue

        for i in range(2):
            try:
                return self._chat_request(messages, use_tools=False), "chat-no-tools"
            except Exception as e:
                last_err = f"chat-no-tools[{e}]"
                _sleep_backoff(i)
                continue

        for i in range(2):
            try:
                return self._responses_request(messages), "responses"
            except Exception as e:
                last_err = f"responses[{e}]"
                _sleep_backoff(i)
                continue

        return json.dumps({"error": f"external_research_unavailable: {last_err}"}), "error"

    # ---------- Public API ----------
    def research(self, company: str, sector: str, location: str, description: str) -> Dict[str, Any]:
        if not self.enabled:
            return {"notice": "External research service is not configured.", "summary": "", "sections": {}, "sources": [], "memo": {}}

        system = (
            "You are a senior VC analyst with live web capabilities. "
            "Perform an exhaustive deep dive across X (Twitter), official sites/docs, engineering blogs, job boards, "
            "filings, reputable news, funding databases, app stores, developer forums, and other public sources. "
            "Normalize currencies/units and include dates. Return STRICT JSON only."
        )

        user = f"""
Return ONLY JSON with this schema:

{{
  "summary": "6–10 sentences covering product, market, traction, milestones, unit-economic hints, and open questions.",
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
    "executive_summary": "IC-ready summary",
    "investment_thesis": "Why now/why this team/why this wedge",
    "market": "TAM/SAM/SOM w/ sources; trend vectors",
    "product": "Depth, differentiation, defensibility",
    "traction": "Logos, usage, revenue signals, pipeline",
    "unit_economics": "Gross margin, LTV/CAC, payback, burn multiple",
    "gtm": "Channels, motion, pricing, expansion",
    "competition": "Landscape + switching costs",
    "team": "Founders, execs, critical hires",
    "risks": "Top risks + mitigations",
    "catalysts": "12–18m milestones",
    "round_dynamics": "Stage, size, valuation markers, prior investors",
    "use_of_proceeds": "Planned allocation",
    "valuation_rationale": "Comparables/method w/ sources",
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
- STRICT JSON ONLY; no prose outside JSON.
- Include 10–20 diverse sources with URLs and two-sentence snippets (include dates); set confidence 0–1.
- If unknown, leave fields empty. Prefer information from the last 12–24 months.
""".strip()

        content, route = self._chat_with_retries(system, user)
        data = _first_json(content or "") or {}
        data.setdefault("summary", "")
        data.setdefault("sections", {})
        data.setdefault("sources", [])
        data.setdefault("memo", {})
        if not isinstance(data["sections"], dict):
            data["sections"] = {}
        data["sources"] = _normalize_sources(data["sources"], max_n=20)
        if not isinstance(data["memo"], dict):
            data["memo"] = {}
        data["_diagnostics"] = {"route": route}
        return data

    def build_ic_memo(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        system = (
            "You are a veteran VC partner preparing an Investment Committee memo. "
            "Use the supplied quantitative KPIs and the research bundle to produce a crisp, specific, and defensible memo. "
            "Cite data inline using [n] tags corresponding to sources if appropriate. Output STRICT JSON only."
        )
        user = (
            "Context JSON follows. Produce ONLY JSON for the memo with fields: "
            "executive_summary, investment_thesis, market, product, traction, unit_economics, gtm, competition, team, "
            "risks, catalysts, round_dynamics, use_of_proceeds, valuation_rationale, kpis_next_12m, exit_paths, "
            "recommendation, conviction.\n\n"
            f"{json.dumps(context)[:18000]}"
        )
        content, route = self._chat_with_retries(system, user)
        memo = _first_json(content or "") or {}
        if isinstance(memo, dict):
            memo["_diagnostics"] = {"route": route}
        return memo if isinstance(memo, dict) else {}
