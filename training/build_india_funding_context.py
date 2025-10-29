"""
Build sector-aware India funding context from Kaggle:
  sandeepnandi48/startup-funding-india-cleaned

Aggregates rounds by sector and year, top investors, and typical amounts,
and saves a compact JSON for RAG-like use in prompts.

Usage:
  python -m models.training.build_india_funding_context --dataset-ref sandeepnandi48/startup-funding-india-cleaned

Env:
  KAGGLE_USERNAME, KAGGLE_KEY

Output:
  data/india_funding_index.json
"""
import os
import json
import argparse
import tempfile
from typing import Dict, Any

import pandas as pd
import numpy as np

def download_kaggle_dataset(dataset_ref: str, out_dir: str):
    import subprocess
    cmd = ["kaggle", "datasets", "download", "-d", dataset_ref, "-p", out_dir, "--unzip"]
    subprocess.check_call(cmd)

def normalize_amount(val: str) -> float:
    if pd.isna(val):
        return np.nan
    s = str(val).lower().replace(",", "").replace("rs.", "").replace("rs", "").replace("inr", "").strip()
    mult = 1.0
    if "crore" in s or "cr" in s:
        mult = 10_000_000
        s = s.replace("crore", "").replace("cr", "")
    if "lakh" in s or "lac" in s:
        mult = 100_000
        s = s.replace("lakh", "").replace("lac", "")
    if "mn" in s or "million" in s:
        mult = 1_000_000
        s = s.replace("mn", "").replace("million", "")
    num = "".join([c for c in s if c.isdigit() or c == "." or c == "-"])
    try:
        return float(num) * mult
    except Exception:
        return np.nan

def build_index(df: pd.DataFrame) -> Dict[str, Any]:
    df = df.copy()
    if "Date" in df.columns:
        df["year"] = pd.to_datetime(df["Date"], errors="coerce").dt.year
    else:
        df["year"] = np.nan

    if "Amount" in df.columns:
        df["amount_inr"] = df["Amount"].apply(normalize_amount)
    elif "Amount in USD" in df.columns:
        try:
            df["amount_inr"] = df["Amount in USD"].astype(float) * 83.0
        except Exception:
            df["amount_inr"] = np.nan
    else:
        df["amount_inr"] = np.nan

    sector_col = "Industry Vertical" if "Industry Vertical" in df.columns else ("Sector" if "Sector" in df.columns else None)
    sub_col = "SubVertical" if "SubVertical" in df.columns else None
    inv_col = "Investors Name" if "Investors Name" in df.columns else None
    round_col = "Investment Type" if "Investment Type" in df.columns else None

    df["sector"] = df[sector_col].fillna("Unknown") if sector_col else "Unknown"
    df["subvertical"] = df[sub_col].fillna("") if sub_col else ""
    df["round"] = df[round_col].fillna("Undisclosed") if round_col else "Undisclosed"

    by_sector = df.groupby("sector").agg(
        rounds=("Startup Name", "count"),
        median_amount_inr=("amount_inr", "median")
    ).reset_index()

    by_sector_year = df.groupby(["sector", "year"]).size().reset_index(name="rounds")

    top_inv = {}
    if inv_col:
        inv_df = df.dropna(subset=[inv_col]).copy()
        inv_df[inv_col] = inv_df[inv_col].astype(str).str.split(",")
        inv_df = inv_df.explode(inv_col)
        inv_df[inv_col] = inv_df[inv_col].str.strip()
        top_inv = (
            inv_df.groupby(["sector", inv_col]).size().reset_index(name="count")
            .sort_values(["sector", "count"], ascending=[True, False])
        )

    rounds_mix = df.groupby(["sector", "round"]).size().reset_index(name="count")

    index = {"sectors": {}}
    for sector in by_sector["sector"].unique():
        s_block = {}
        s_block["rounds_total"] = int(by_sector.loc[by_sector["sector"] == sector, "rounds"].iloc[0])
        med = by_sector.loc[by_sector["sector"] == sector, "median_amount_inr"].iloc[0]
        s_block["median_amount_inr"] = None if pd.isna(med) else float(med)

        years = by_sector_year[by_sector_year["sector"] == sector].sort_values("year")
        s_block["yearly_rounds"] = [
            {"year": int(y) if not pd.isna(y) else None, "rounds": int(r)} for y, r in zip(years["year"], years["rounds"])
        ]

        mix = rounds_mix[rounds_mix["sector"] == sector].sort_values("count", ascending=False)
        s_block["round_mix"] = [{"round": str(r), "count": int(c)} for r, c in zip(mix["round"], mix["count"])]

        if isinstance(top_inv, pd.DataFrame) and not top_inv.empty:
            tdf = top_inv[top_inv["sector"] == sector].head(15)
            s_block["top_investors"] = [{"name": str(n), "mentions": int(c)} for n, c in zip(tdf["Investors Name"], tdf["count"])]
        else:
            s_block["top_investors"] = []

        index["sectors"][str(sector)] = s_block

    return index

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-ref", default="sandeepnandi48/startup-funding-india-cleaned")
    ap.add_argument("--out", default="data/india_funding_index.json")
    args = ap.parse_args()

    os.makedirs("data", exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        download_kaggle_dataset(args.dataset_ref, td)
        csvs = [os.path.join(td, f) for f in os.listdir(td) if f.endswith(".csv")]
        path = max(csvs, key=lambda p: os.path.getsize(p))
        df = pd.read_csv(path)

    index = build_index(df)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"Wrote India funding index to {args.out}")

if __name__ == "__main__":
    main()
