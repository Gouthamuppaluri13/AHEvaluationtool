"""
VC-aware hallucination evaluation metrics.

Design goals
- Unbiased: entropy-based uncertainty + optional model logits (ensemble-ready).
- VC-Relevant: trend-aware relevance boost (2025 defaults in vc_trends.py).
- Intelligent: math/science verification layer via SymPy.
- Efficient: batch-friendly; graceful fallbacks if heavy deps missing.

Public API
- compute_vc_ahe_score(prompt, response, domain_logits=None, trends=None, projected_cf=None, weights=None)
- compute_vc_ahe_score_batch(prompts, responses, logits_list=None, trends=None, projected_cf_list=None, weights=None)

Returned dict keys (single) or list of dicts (batch):
{
  base_score,            # 0..1
  rouge_l, bert_f,       # components (may be fallback approximations)
  uncertainty,           # 0..1 (higher = more uncertain)
  trend_relevance,       # 0..1
  math_score,            # 0..1
  finance_score,         # 0..1
  vc_ahe_score,          # final 0..1
  bias_risk,             # low|medium|high (from uncertainty vs. fluency)
  valuation_risk         # low|medium|high (from finance check)
}
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import math
import re
import numpy as np

# Optional heavy deps (guarded)
try:
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore
    TORCH_OK = True
except Exception:
    torch = None  # type: ignore
    F = None  # type: ignore
    TORCH_OK = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
    HF_OK = True
except Exception:
    AutoTokenizer = None  # type: ignore
    AutoModelForSequenceClassification = None  # type: ignore
    HF_OK = False

try:
    from bert_score import score as bertscore_score  # type: ignore
    BERTSCORE_OK = True
except Exception:
    bertscore_score = None  # type: ignore
    BERTSCORE_OK = False

try:
    from rouge_score import rouge_scorer  # type: ignore
    ROUGE_OK = True
except Exception:
    rouge_scorer = None  # type: ignore
    ROUGE_OK = False

try:
    from sympy import sympify, simplify  # type: ignore
    SYMPY_OK = True
except Exception:
    sympify = None  # type: ignore
    simplify = None  # type: ignore
    SYMPY_OK = False

# Local modules (lightweight)
try:
    from vc_trends import VC_TRENDS, trend_relevance
except Exception:
    VC_TRENDS = {"ai_dominance": 0.62, "esg_weight": 0.15, "exit_rate": 0.25}

    def trend_relevance(response: str, trends: dict) -> float:
        keys = {
            "AI": trends.get("ai_dominance", 0.0),
            "GenAI": trends.get("ai_dominance", 0.0) * 0.9,
            "ESG": trends.get("esg_weight", 0.0),
            "sustainability": trends.get("esg_weight", 0.0) * 0.8,
            "exit": trends.get("exit_rate", 0.0) * 0.6,
            "acquisition": trends.get("exit_rate", 0.0) * 0.7,
            "IPO": trends.get("exit_rate", 0.0) * 0.8,
        }
        low = response.lower()
        hits: List[float] = [(1.0 if k.lower() in low else 0.0) * float(w) for k, w in keys.items()]
        return float(sum(hits) / max(1, len(hits)))

try:
    from finance import fin_verify
except Exception:
    def fin_verify(response: str, projected_cf: Optional[List[float]] = None) -> float:
        return 0.8  # neutral fallback


# ------------------------------
# Cheap text similarity fallbacks
# ------------------------------
def _lcs(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[n][m]


def _rouge_l_f1(hyp: str, ref: str) -> float:
    if ROUGE_OK:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return float(scorer.score(ref, hyp)["rougeL"].fmeasure)
    # Fallback: word-level LCS F1
    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    lcs = _lcs(ref_tokens, hyp_tokens)
    prec = lcs / (len(hyp_tokens) + 1e-8)
    rec = lcs / (len(ref_tokens) + 1e-8)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec + 1e-8)


def _char_ngram_f1(hyp: str, ref: str, n: int = 3) -> float:
    def grams(s: str) -> Dict[str, int]:
        s = s.lower()
        d: Dict[str, int] = {}
        for i in range(max(0, len(s) - n + 1)):
            g = s[i : i + n]
            d[g] = d.get(g, 0) + 1
        return d

    g_ref = grams(ref)
    g_hyp = grams(hyp)
    if not g_ref or not g_hyp:
        return 0.0
    common = 0
    for k, v in g_hyp.items():
        common += min(v, g_ref.get(k, 0))
    prec = common / (sum(g_hyp.values()) + 1e-8)
    rec = common / (sum(g_ref.values()) + 1e-8)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec + 1e-8)


def _bert_f1(hyp: str, ref: str) -> float:
    if BERTSCORE_OK:
        P, R, F = bertscore_score([hyp], [ref], lang="en")
        return float(F.mean().item())
    # Fallback: char trigram F1
    return _char_ngram_f1(hyp, ref, n=3)


# ------------------------------
# Uncertainty (entropy)
# ------------------------------
def entropy_uncertainty_from_logits(logits: "torch.Tensor") -> float:
    """Expected entropy from classifier logits (higher = more uncertain). Returns 0..1 normalized."""
    if not TORCH_OK:
        return 0.5
    probs = F.softmax(logits, dim=-1)
    ent = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
    C = logits.shape[-1]
    max_ent = math.log(max(C, 2))
    return float(min(1.0, max(0.0, ent / (max_ent + 1e-8))))


_HF_CACHE: Dict[str, Any] = {}


def _load_light_classifier(model_name: str = "distilbert-base-uncased") -> Tuple[Any, Any]:
    if not HF_OK:
        return None, None
    key = f"clf::{model_name}"
    if key in _HF_CACHE:
        return _HF_CACHE[key]
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
        mdl.eval()
        _HF_CACHE[key] = (tok, mdl)
        return tok, mdl
    except Exception:
        return None, None


def entropy_uncertainty_text(text: str) -> float:
    """
    Text-only uncertainty proxy when logits/model unavailable.
    Normalized Shannon entropy of token distribution (0..1).
    Lower entropy → overconfident distribution (riskier).
    """
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return 0.5
    counts: Dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    ent = -sum(p * math.log(p + 1e-9) for p in probs)
    max_ent = math.log(len(counts) + 1e-9)
    return float(min(1.0, max(0.0, ent / (max_ent + 1e-9))))


# ------------------------------
# Science/Math verification
# ------------------------------
def math_verify(response: str) -> float:
    """
    Lightweight math verification proxy:
    - Extracts simple arithmetic segments and tries to parse with SymPy.
    - Returns 1.0 if parsing is consistent; 0.85 if simplified; 0.0 if fails.
    """
    if not SYMPY_OK:
        return 0.8  # neutral if sympy unavailable
    exprs = re.findall(r"[0-9\.\s\+\-\*/\^\(\)x]+", response)
    if not exprs:
        return 1.0
    sample = " ".join(exprs[:2]).strip()
    try:
        parsed = sympify(sample)
        _ = simplify(parsed)
        return 0.85
    except Exception:
        return 0.0


# ------------------------------
# Risk helpers
# ------------------------------
def _bias_risk(uncertainty: float, fluency: float) -> str:
    if uncertainty < 0.25 and fluency >= 0.7:
        return "high"
    if uncertainty < 0.45 and fluency >= 0.6:
        return "medium"
    return "low"


def _valuation_risk(finance_score: float) -> str:
    if finance_score >= 0.9:
        return "low"
    if finance_score >= 0.7:
        return "medium"
    return "high"


_DEFAULT_WEIGHTS = {
    "rouge": 0.35,
    "bert": 0.35,
    "uncertainty": 0.15,  # applied as (1 - unc)
    "math": 0.075,
    "trend": 0.05,
    "finance": 0.025,
}


# ------------------------------
# Main API
# ------------------------------
def compute_vc_ahe_score(
    prompt: str,
    response: str,
    domain_logits: Optional["torch.Tensor"] = None,
    trends: Optional[Dict[str, float]] = None,
    projected_cf: Optional[List[float]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compute VC-aware hallucination score (0..1) and diagnostics.
    """
    w = dict(_DEFAULT_WEIGHTS)
    if isinstance(weights, dict) and weights:
        s = sum(max(0.0, float(v)) for v in weights.values())
        if s > 0:
            for k in list(w.keys()):
                if k in weights:
                    w[k] = float(weights[k]) / s

    # Similarity components
    rouge_l = _rouge_l_f1(response, prompt)
    bert_f = _bert_f1(response, prompt)

    # Uncertainty
    if (domain_logits is not None) and TORCH_OK:
        uncertainty = entropy_uncertainty_from_logits(domain_logits)
    else:
        if HF_OK:
            tok, mdl = _load_light_classifier()
            if tok is not None and mdl is not None:
                try:
                    with torch.no_grad():  # type: ignore
                        inputs = tok(response, return_tensors="pt", truncation=True, max_length=256)
                        logits = mdl(**inputs).logits
                        uncertainty = entropy_uncertainty_from_logits(logits)
                except Exception:
                    uncertainty = entropy_uncertainty_text(response)
            else:
                uncertainty = entropy_uncertainty_text(response)
        else:
            uncertainty = entropy_uncertainty_text(response)

    # Trend relevance
    t = trends if isinstance(trends, dict) else VC_TRENDS
    t_rel = float(trend_relevance(response, t))

    # Math/science
    m_acc = float(math_verify(response))

    # Finance verification
    f_acc = float(fin_verify(response, projected_cf))

    # Aggregate
    parts = {
        "rouge": rouge_l,
        "bert": bert_f,
        "uncertainty": (1.0 - uncertainty),
        "math": m_acc,
        "trend": t_rel,
        "finance": f_acc,
    }
    base_score = w["rouge"] * parts["rouge"] + w["bert"] * parts["bert"] + w["uncertainty"] * parts["uncertainty"]
    vc_ahe_score = base_score + w["math"] * parts["math"] + w["trend"] * parts["trend"] + w["finance"] * parts["finance"]
    vc_ahe_score = float(min(1.0, max(0.0, vc_ahe_score)))

    bias = _bias_risk(uncertainty, fluency=max(bert_f, rouge_l))
    val_risk = _valuation_risk(f_acc)

    return {
        "base_score": round(float(base_score), 4),
        "rouge_l": round(float(rouge_l), 4),
        "bert_f": round(float(bert_f), 4),
        "uncertainty": round(float(uncertainty), 4),
        "trend_relevance": round(float(t_rel), 4),
        "math_score": round(float(m_acc), 4),
        "finance_score": round(float(f_acc), 4),
        "vc_ahe_score": round(vc_ahe_score, 4),
        "bias_risk": bias,
        "valuation_risk": val_risk,
    }


def compute_vc_ahe_score_batch(
    prompts: Iterable[str],
    responses: Iterable[str],
    logits_list: Optional[Iterable[Optional["torch.Tensor"]]] = None,
    trends: Optional[Dict[str, float]] = None,
    projected_cf_list: Optional[Iterable[Optional[List[float]]]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Batch version; iterates efficiently. If torch tensors are provided in logits_list,
    pass-through to single scorer for entropy; otherwise uses text proxy/HF lightweight.
    """
    prompts = list(prompts)
    responses = list(responses)
    N = len(responses)
    logits_list = list(logits_list) if logits_list is not None else [None] * N
    projected_cf_list = list(projected_cf_list) if projected_cf_list is not None else [None] * N

    out: List[Dict[str, Any]] = []
    for i in range(N):
        out.append(
            compute_vc_ahe_score(
                prompt=prompts[i],
                response=responses[i],
                domain_logits=logits_list[i],
                trends=trends,
                projected_cf=projected_cf_list[i],
                weights=weights,
            )
        )
    return out
