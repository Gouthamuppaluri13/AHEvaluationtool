"""
Lightweight startup finance checks for hallucination detection.

Public API:
- dcf_valuation(cashflows, discount_rate=0.12, terminal_growth=0.03) -> float
- fin_verify(response, projected_cf=None) -> float in [0,1] (higher = more plausible)
"""
from __future__ import annotations

from typing import List, Optional
import re

def dcf_valuation(cashflows: List[float], discount_rate: float = 0.12, terminal_growth: float = 0.03) -> float:
    cashflows = [float(x) for x in (cashflows or [])]
    if not cashflows:
        return 0.0
    pv_cf = 0.0
    for t, cf in enumerate(cashflows, start=1):
        pv_cf += cf / ((1.0 + discount_rate) ** t)
    terminal = cashflows[-1] * (1.0 + terminal_growth) / max(1e-6, (discount_rate - terminal_growth))
    return float(pv_cf + terminal / ((1.0 + discount_rate) ** len(cashflows)))

def _extract_amounts(response: str) -> List[float]:
    """
    Parse amounts with optional K/M suffix, e.g., $1.2M, 250k, 3,400,000.
    Returns values in absolute USD.
    """
    amounts: List[float] = []
    for m in re.finditer(r"\$?\s*([0-9][\d,]*(?:\.\d+)?)\s*([kKmM]?)", response):
        num = float(m.group(1).replace(",", ""))
        suf = m.group(2).lower()
        mult = 1.0
        if suf == "k":
            mult = 1e3
        elif suf == "m":
            mult = 1e6
        amounts.append(num * mult)
    return amounts

def fin_verify(response: str, projected_cf: Optional[List[float]] = None) -> float:
    """
    Compare implied valuation to simple rule-of-thumb (e.g., 10x first CF/ARR). Returns [0,1].
    """
    amounts = _extract_amounts(response)
    if not amounts and not projected_cf:
        return 0.8  # neutral if no signals
    # Use provided CF, else derive simple proxy CFs from amounts (first 5 → 5 years)
    cfs = [float(x) for x in (projected_cf or [max(0.0, a * 0.2) for a in amounts[:5]])]
    if not cfs:
        return 0.8
    val_est = dcf_valuation(cfs)
    # Heuristic ground truth: ~10x first CF as ARR-like proxy (conservative)
    gt = max(1e6, cfs[0] * 10.0)
    rel_err = abs(val_est - gt) / gt
    if rel_err <= 0.15:
        return 0.95
    if rel_err <= 0.3:
        return 0.8
    if rel_err <= 0.6:
        return 0.65
    return 0.5
