from __future__ import annotations
import io
import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber

# Optional Gemini
try:
    import google.generativeai as genai  # type: ignore
    HAS_GEMINI = True
except Exception:
    genai = None  # type: ignore
    HAS_GEMINI = False

Number = float | int

_UNIT_PATTERNS = [
    (r"\b(cr|crore|crores)\b", 1e7, "INR"),
    (r"\b(lk|lac|lakh|lakhs)\b", 1e5, "INR"),
    (r"\b(mn|million|millions)\b", 1e6, "USD-or-INR"),
    (r"\b(bn|billion|billions)\b", 1e9, "USD-or-INR"),
]

_CCY_HINTS = {
    "₹": "INR", "inr": "INR", "rupee": "INR",
    "$": "USD", "usd": "USD", "dollar": "USD",
}

def _detect_currency(text: str) -> Optional[str]:
    t = text.lower()
    for k, v in _CCY_HINTS.items():
        if k in t:
            return v
    return None

def _parse_money_token(s: str) -> Tuple[Optional[float], Optional[str]]:
    raw = s.strip()
    ccy = _detect_currency(raw)
    num_match = re.search(r"([+-]?\d[\d,]*(?:\.\d+)?)", raw.replace(",", ""))
    if not num_match:
        return None, ccy
    num = float(num_match.group(1))
    text_low = raw.lower()
    mult = 1.0
    unit_ccy_bias = None
    for pat, mul, bias in _UNIT_PATTERNS:
        if re.search(pat, text_low):
            mult = mul
            unit_ccy_bias = bias
            break
    if unit_ccy_bias == "USD-or-INR":
        if ccy is None and "₹" in raw:
            ccy = "INR"
        elif ccy is None and "$" in raw:
            ccy = "USD"
    return num * mult, ccy

def _first_match(text: str, labels: List[str]) -> Optional[str]:
    for label in labels:
        m = re.search(label, text, flags=re.I)
        if m:
            return m.group(1).strip()
    return None

class PDFIngestor:
    def __init__(self, gemini_key: Optional[str] = None):
        self.enabled_llm = False
        self.model = None
        if HAS_GEMINI and gemini_key:
            try:
                genai.configure(api_key=gemini_key)  # type: ignore
                self.model = genai.GenerativeModel("gemini-1.5-flash")  # type: ignore
                self.enabled_llm = True
            except Exception:
                self.enabled_llm = False

    def extract_text(self, file_bytes: bytes, max_pages: int = 40) -> str:
        buf = io.BytesIO(file_bytes)
        chunks: List[str] = []
        with pdfplumber.open(buf) as pdf:
            for i, page in enumerate(pdf.pages):
                if i >= max_pages:
                    break
                try:
                    txt = page.extract_text() or ""
                except Exception:
                    txt = ""
                if txt:
                    chunks.append(txt)
        text = "\n".join(chunks)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{2,}", "\n\n", text).strip()
        return text

    def _coerce_float(self, s: Any, default: float = 0.0) -> float:
        if isinstance(s, (int, float)):
            return float(s)
        if not isinstance(s, str):
            return default
        s_clean = s.replace(",", "").replace("₹", "").replace("$", "").strip()
        m = re.search(r"([+-]?\d+(?:\.\d+)?)", s_clean)
        return float(m.group(1)) if m else default

    def _coerce_int(self, s: Any, default: int = 0) -> int:
        try:
            f = self._coerce_float(s, float(default))
            return int(round(f))
        except Exception:
            return default

    def _fallback_extract(self, text: str, file_name: Optional[str]) -> Dict[str, Any]:
        company = _first_match(text, [r"Company Name[:\-]\s*(.+)", r"Company[:\-]\s*(.+)"]) or \
                  (os.path.splitext(file_name or "")[0] if file_name else "Untitled Company")
        sector = _first_match(text, [r"Sector[:\-]\s*(.+)"]) or "E-commerce & D2C"
        stage = _first_match(text, [r"(Pre-Seed|Seed|Series A|Series B)"]) or "Series A"
        location = _first_match(text, [r"Location[:\-]\s*(.+)"]) or ("India" if "₹" in text or "INR" in text.upper() else "US")

        arr_raw = _first_match(text, [r"ARR[:\-]\s*([^\n]+)", r"Annual Recurring Revenue[:\-]\s*([^\n]+)"]) or ""
        burn_raw = _first_match(text, [r"Burn[:\-]\s*([^\n]+)", r"Monthly Burn[:\-]\s*([^\n]+)"]) or ""
        cash_raw = _first_match(text, [r"Cash[:\-]\s*([^\n]+)", r"Cash Reserves[:\-]\s*([^\n]+)"]) or ""

        arr_val, arr_ccy = _parse_money_token(arr_raw)
        burn_val, burn_ccy = _parse_money_token(burn_raw)
        cash_val, cash_ccy = _parse_money_token(cash_raw)

        doc_ccy = _detect_currency(text) or "INR"
        fx_hint = 83.0

        def to_inr(v: Optional[float], ccy: Optional[str]) -> float:
            if v is None:
                return 0.0
            if ccy == "USD":
                return v * fx_hint
            return v

        arr_inr = to_inr(arr_val, arr_ccy or doc_ccy)
        burn_inr = to_inr(burn_val, burn_ccy or doc_ccy)
        cash_inr = to_inr(cash_val, cash_ccy or doc_ccy)

        team_size = self._coerce_int(_first_match(text, [r"Team Size[:\-]\s*([0-9,\.]+)"]), 40)
        ltv_cac = self._coerce_float(_first_match(text, [r"LTV\s*\:\s*CAC(?:\s*Ratio)?[:\-]?\s*([0-9\.]+)"]), 3.0)
        gross_margin = self._coerce_float(_first_match(text, [r"Gross Margin(?: %| Percent)?[:\-]?\s*([0-9\.]+)"]), 60.0)
        churn = self._coerce_float(_first_match(text, [r"(?:Monthly )?Churn(?: %| Percent)?[:\-]?\s*([0-9\.]+)"]), 2.0)

        founder_bio = _first_match(text, [r"Founder(?:s)?(?: Bio)?[:\-]\s*(.+)"]) or ""
        product_desc = _first_match(text, [r"(?:Product|Solution|Platform)[:\-]\s*(.+)"]) or ""

        data = {
            "company_name": company,
            "focus_area": "Enhance Urban Lifestyle",
            "sector": sector,
            "location": location,
            "stage": stage,
            "founder_type": "technical",
            "founder_bio": founder_bio,
            "product_desc": product_desc,

            "founded_year": 2022,
            "age": 2025 - 2022,
            "total_funding_usd": 5_000_000,
            "team_size": team_size,
            "num_investors": 5,

            "product_stage_score": 7.0,
            "team_score": 8.0,
            "moat_score": 7.0,
            "investor_quality_score": 7.0,

            "ltv_cac_ratio": max(0.1, min(10.0, ltv_cac)),
            "gross_margin_pct": max(0.0, min(95.0, gross_margin)),
            "monthly_churn_pct": max(0.0, min(20.0, churn)),

            "arr": arr_inr if arr_inr > 0 else 80_000_000.0,
            "burn": burn_inr if burn_inr > 0 else 10_000_000.0,
            "cash": cash_inr if cash_inr > 0 else 90_000_000.0,

            "expected_monthly_growth_pct": 5.0,
            "growth_volatility_pct": 3.0,
            "lead_to_customer_conv_pct": 5.0,
            "currency": "INR" if (doc_ccy or "INR") == "INR" else "USD",

            "monthly_web_traffic": [5000, 6200, 8100, 11000, 13500, 16000, 19000, 22000, 25000, 28000, 31000, 35000],
        }
        return data

    def _llm_extract(self, text: str) -> Optional[Dict[str, Any]]:
        if not self.enabled_llm or self.model is None:
            return None
        prompt = (
            "Extract a JSON object for a startup evaluation. Only output raw JSON. "
            "Use these keys (exact names) and fill with best estimates if missing:\n"
            "{\n"
            '  "company_name": str,\n'
            '  "focus_area": "Enhance Urban Lifestyle" | "Live Healthy" | "Mitigate Climate Change",\n'
            '  "sector": str,\n'
            '  "location": str,\n'
            '  "stage": "Pre-Seed" | "Seed" | "Series A" | "Series B",\n'
            '  "founder_type": "technical" | "visionary" | "executor",\n'
            '  "founder_bio": str,\n'
            '  "product_desc": str,\n'
            '  "founded_year": int,\n'
            '  "age": int,\n'
            '  "total_funding_usd": number,\n'
            '  "team_size": int,\n'
            '  "num_investors": int,\n'
            '  "product_stage_score": number,\n'
            '  "team_score": number,\n'
            '  "moat_score": number,\n'
            '  "investor_quality_score": number,\n'
            '  "ltv_cac_ratio": number,\n'
            '  "gross_margin_pct": number,\n'
            '  "monthly_churn_pct": number,\n'
            '  "arr": number,  // INR if Indian context, else USD\n'
            '  "burn": number, // INR per month or USD per month\n'
            '  "cash": number, // INR or USD\n'
            '  "currency": "INR" | "USD",\n'
            '  "expected_monthly_growth_pct": number,\n'
            '  "growth_volatility_pct": number,\n'
            '  "lead_to_customer_conv_pct": number,\n'
            '  "monthly_web_traffic": [int, int, int, int, int, int, int, int, int, int, int, int]\n'
            "}\n\n"
            "Source PDF text follows between <PDF> tags. If a field isn’t present, infer a reasonable default.\n"
            "<PDF>\n"
            f"{text[:18000]}\n"
            "</PDF>"
        )
        try:
            r = self.model.generate_content(prompt)  # type: ignore
            t = getattr(r, "text", "") or ""
            m = re.search(r"\{.*\}", t, flags=re.DOTALL)
            if not m:
                return None
            data = json.loads(m.group(0))
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _normalize(self, d: Dict[str, Any], file_name: Optional[str]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        out["company_name"] = str(d.get("company_name") or (os.path.splitext(file_name or "")[0] or "Untitled Company"))
        fa = str(d.get("focus_area") or "").strip()
        if fa.lower() not in {"enhance urban lifestyle", "live healthy", "mitigate climate change"}:
            fa = "Enhance Urban Lifestyle"
        out["focus_area"] = fa
        out["sector"] = str(d.get("sector") or "E-commerce & D2C")
        out["location"] = str(d.get("location") or "India")
        stage = str(d.get("stage") or "Series A")
        if stage not in {"Pre-Seed", "Seed", "Series A", "Series B"}:
            stage = "Series A"
        out["stage"] = stage

        ft = str(d.get("founder_type") or "technical").lower()
        if ft not in {"technical", "visionary", "executor"}:
            ft = "technical"
        out["founder_type"] = ft
        out["founder_bio"] = str(d.get("founder_bio") or "")
        out["product_desc"] = str(d.get("product_desc") or "")

        out["founded_year"] = int(d.get("founded_year") or 2022)
        out["age"] = int(d.get("age") or (2025 - out["founded_year"]))
        out["total_funding_usd"] = float(d.get("total_funding_usd") or 5_000_000)
        out["team_size"] = int(d.get("team_size") or 50)
        out["num_investors"] = int(d.get("num_investors") or 5)

        def clamp(v: Number, lo: Number, hi: Number) -> float:
            try:
                f = float(v)
            except Exception:
                f = float(lo)
            return float(max(lo, min(hi, f)))

        out["product_stage_score"] = clamp(d.get("product_stage_score", 7.0), 0, 10)
        out["team_score"] = clamp(d.get("team_score", 8.0), 0, 10)
        out["moat_score"] = clamp(d.get("moat_score", 7.0), 0, 10)
        out["investor_quality_score"] = clamp(d.get("investor_quality_score", 7.0), 1, 10)

        out["ltv_cac_ratio"] = clamp(d.get("ltv_cac_ratio", 3.0), 0.1, 10)
        out["gross_margin_pct"] = clamp(d.get("gross_margin_pct", 60.0), 0, 95)
        out["monthly_churn_pct"] = clamp(d.get("monthly_churn_pct", 2.0), 0, 20)

        out["arr"] = float(d.get("arr") or 80_000_000)
        out["burn"] = float(d.get("burn") or 10_000_000)
        out["cash"] = float(d.get("cash") or 90_000_000)
        out["currency"] = (d.get("currency") or ("INR" if "₹" in (d.get("product_desc","")+d.get("founder_bio","")) else "INR")).upper()

        out["expected_monthly_growth_pct"] = float(d.get("expected_monthly_growth_pct") or 5.0)
        out["growth_volatility_pct"] = float(d.get("growth_volatility_pct") or 3.0)
        out["lead_to_customer_conv_pct"] = float(d.get("lead_to_customer_conv_pct") or 5.0)

        mwt = d.get("monthly_web_traffic")
        if not isinstance(mwt, list) or len(mwt) != 12:
            out["monthly_web_traffic"] = [5000, 6200, 8100, 11000, 13500, 16000, 19000, 22000, 25000, 28000, 31000, 35000]
        else:
            out["monthly_web_traffic"] = [int(x) for x in mwt]
        return out

    def extract(self, file_bytes: bytes, file_name: Optional[str] = None) -> Dict[str, Any]:
        text = self.extract_text(file_bytes)
        data = self._llm_extract(text) or self._fallback_extract(text, file_name)
        return self._normalize(data, file_name)
