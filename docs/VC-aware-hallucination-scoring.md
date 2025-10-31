# VC-aware Hallucination Scoring

This module adds an unbiased, VC-relevant, intelligent, and efficient evaluation for LLM outputs in startup/VC contexts.

## What it does

- Unbiased: entropy-based uncertainty (from logits, a tiny HF classifier, or a text entropy proxy).
- VC-Relevant: trend scoring against static 2025 proxies (AI dominance, ESG, exits) in `vc_trends.py`.
- Intelligent: math/science verification via SymPy (graceful fallback when unavailable).
- Financial sanity: DCF-based plausibility check with K/M amount parsing.
- Efficient: batch API + CLI that runs on CPU and degrades gracefully if optional deps are missing.

## Quickstart

```bash
python evaluator.py --data samples/samples.jsonl --out results.jsonl --csv results.csv
```

- Input is JSONL with `prompt`, `response`, optional `projected_cf` (list of cash flows in USD).
- Output is JSONL (and optional CSV) with per-row `vc_ahe_score` and diagnostics.

## Optional dependencies

- If available: `rouge_score`, `bert_score`, `torch`, `transformers`, `sympy` will be used automatically.
- If missing, the scorer falls back to lightweight approximations so it still runs.

## Advanced

- Pass domain logits to `compute_vc_ahe_score(..., domain_logits=...)` to compute uncertainty directly from your classifier.
- Override weights by passing `weights={'rouge':..., 'bert':..., ...}` (they will be renormalized).
