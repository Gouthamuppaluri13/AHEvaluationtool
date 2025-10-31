"""
CLI for VC-aware hallucination evaluation.

Usage:
  python evaluator.py --data samples/samples.jsonl --out results.jsonl --csv results.csv

Input JSONL rows:
  {"prompt": "...", "response": "...", "projected_cf": [..]}  # projected_cf optional
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

from metrics import compute_vc_ahe_score_batch, VC_TRENDS  # type: ignore

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    import csv
    if not rows:
        return
    keys = sorted(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Input JSONL with 'prompt' and 'response'.")
    ap.add_argument("--out", required=True, help="Output JSONL path.")
    ap.add_argument("--csv", default="", help="Optional CSV path.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--no-trends", action="store_true", help="Disable trend relevance boost.")
    args = ap.parse_args()

    rows = _read_jsonl(args.data)
    prompts = [r.get("prompt", "") for r in rows]
    responses = [r.get("response", "") for r in rows]
    projecteds = [r.get("projected_cf") for r in rows]

    logits_list: List[Optional["torch.Tensor"]] = [None] * len(rows)

    out = []
    for i in range(0, len(rows), args.batch_size):
        batch_prompts = prompts[i : i + args.batch_size]
        batch_resps = responses[i : i + args.batch_size]
        batch_logits = logits_list[i : i + args.batch_size]
        batch_cf = projecteds[i : i + args.batch_size]

        batch_scores = compute_vc_ahe_score_batch(
            prompts=batch_prompts,
            responses=batch_resps,
            logits_list=batch_logits,
            trends=None if args.no_trends else VC_TRENDS,
            projected_cf_list=batch_cf,
            weights=None,
        )
        for j, sc in enumerate(batch_scores):
            merged = dict(rows[i + j])
            merged.update(sc)
            out.append(merged)

    _write_jsonl(args.out, out)
    if args.csv:
        _write_csv(args.csv, out)
    print(f"Wrote {len(out)} scores to {args.out}" + (f" and {args.csv}" if args.csv else ""))

if __name__ == "__main__":
    main()
