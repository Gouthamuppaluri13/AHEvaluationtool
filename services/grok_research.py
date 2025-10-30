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
    A thin wrapper around xAI Grok chat completions to perform web research and
    synthesize a structured, citation-rich bundle for a startup.

    Notes:
    - Expects GROK_API_KEY in environment or provided in __init__.
    - Tries to leverage browsing if the API supports it; otherwise asks the model
      to produce sources it used. If the API rejects tool usage, we still get a useful summary.
    - Return shape:
      {
        "summary": str,
        "sections": {
          "overview": str,
          "products": str,
          "business_model": str,
          "funding": str,
          "investors": str,
          "leadership": str,
          "traction": str,
          "customers": str,
          "competitors": str,
          "moat": str,
          "partnerships": str,
          "risks": str,
          "regulatory": str,
          "controversies": str,
          "hiring": str,
          "tech_stack": str
        },
        "sources": [
          {"title": str, "url": str, "snippet": str, "confidence": float}
        ]
      }
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, api_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROK_API_KEY", "")
        self.model = model or DEFAULT_GROK_MODEL
        self.api_url = api_url or DEFAULT_GROK_API_URL
        self.session = requests.Session()
        self.enabled = bool(self.api_key)

    def _request(self, messages: list[dict], use_web: bool = True, temperature: float = 0.2, max_tokens: int = 1400) -> str:
        """
        Calls xAI's chat completions endpoint. This uses an OpenAI-compatible shape as xAI mirrors it.
        If the API rejects tools, we still receive content.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        # Some xAI deployments support browsing via tools; if unsupported, the API will ignore this.
        if use_web:
            payload["tools"] = [{"type": "web_search"}]
            payload["tool_choice"] = "auto"

        resp = self.session.post(self.api_url, headers=headers, data=json.dumps(payload), timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # OpenAI-compatible: choices[].message.content
        content = ""
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception:
            content = json.dumps(data)
        return content or ""

    def research(self, company: str, sector: str, location: str, description: str) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "notice": "Grok API key not configured. Set GROK_API_KEY in secrets or environment.",
                "summary": "",
                "sections": {},
                "sources": []
            }

        system = (
            "You are Grok, an expert startup analyst with live web access. "
            "Conduct thorough research with recent sources and return STRICT JSON only. "
            "Cite diverse sources with title, url, a 1–2 sentence snippet, and a confidence 0–1. "
            "Make sure numbers are normalized and dates are recent."
        )
        user = f"""
Research startup comprehensively and return ONLY JSON with keys: summary, sections, sources.

Entity:
- Name: {company}
- Sector: {sector}
- Location: {location}
- Description: {description}

Sections to produce (concise paragraphs, no bullets):
- overview
- products
- business_model
- funding
- investors
- leadership
- traction
- customers
- competitors
- moat
- partnerships
- risks
- regulatory
- controversies
- hiring
- tech_stack

Return JSON schema:
{{
  "summary": "...",
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
    {{"title": "...", "url": "https://...", "snippet": "...", "confidence": 0.72}}
  ]
}}

Rules:
- Output strictly valid JSON. No prose outside JSON.
- Prefer official pages, filings, reputable news, funding databases.
- If unknown, set empty string and proceed.
        """.strip()

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            content = self._request(messages, use_web=True)
            # Extract first JSON object from the response
            m = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if not m:
                raise ValueError("No JSON in Grok response")
            data = json.loads(m.group(0))
            if not isinstance(data, dict):
                raise ValueError("Invalid Grok JSON payload")
            # Minimal normalization
            data.setdefault("summary", "")
            data.setdefault("sections", {})
            data.setdefault("sources", [])
            return data
        except Exception as e:
            return {
                "error": f"Grok research failed: {e}",
                "summary": "",
                "sections": {},
                "sources": []
            }
