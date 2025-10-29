"""
Build India Funding Context
Downloads India startup funding dataset from Kaggle and creates sector-wise index.
Usage: python build_india_funding_context.py
"""
import os
import logging
import json
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_india_dataset(dataset_ref: str = "sandeepnandi48/startup-funding-india-cleaned"):
    """Download India startup funding dataset from Kaggle."""
    try:
        import kaggle
        
        download_path = "./kaggle_data_india"
        logger.info(f"Downloading dataset: {dataset_ref}")
        kaggle.api.dataset_download_files(
            dataset_ref,
            path=download_path,
            unzip=True
        )
        logger.info(f"Dataset downloaded to {download_path}")
        return download_path
        
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        logger.error("Make sure KAGGLE_USERNAME and KAGGLE_KEY env vars are set")
        raise


def parse_amount_inr(amount_str):
    """Parse funding amount string to INR value."""
    if pd.isna(amount_str) or amount_str == '':
        return np.nan
    
    try:
        amount_str = str(amount_str).upper().strip()
        
        # Remove commas and currency symbols
        amount_str = amount_str.replace(',', '').replace('₹', '').replace('INR', '').strip()
        
        # Handle millions (M), crores (CR), lakhs (L)
        multiplier = 1
        if 'CR' in amount_str or 'CRORE' in amount_str:
            multiplier = 10000000  # 1 crore = 10 million
            amount_str = amount_str.replace('CR', '').replace('CRORE', '')
        elif 'M' in amount_str or 'MILLION' in amount_str:
            multiplier = 1000000
            amount_str = amount_str.replace('M', '').replace('MILLION', '')
        elif 'L' in amount_str or 'LAKH' in amount_str:
            multiplier = 100000  # 1 lakh
            amount_str = amount_str.replace('L', '').replace('LAKH', '')
        elif 'K' in amount_str or 'THOUSAND' in amount_str:
            multiplier = 1000
            amount_str = amount_str.replace('K', '').replace('THOUSAND', '')
        
        # Parse the numeric value
        value = float(amount_str.strip())
        return value * multiplier
        
    except:
        return np.nan


def build_sector_index(df: pd.DataFrame) -> dict:
    """Build sector-wise funding index."""
    
    # Standardize column names (case-insensitive)
    df.columns = df.columns.str.lower().str.strip()
    
    # Find relevant columns
    sector_col = None
    amount_col = None
    investors_col = None
    round_col = None
    year_col = None
    
    for col in df.columns:
        if 'sector' in col or 'industry' in col or 'vertical' in col:
            sector_col = col
        if 'amount' in col or 'funding' in col:
            amount_col = col
        if 'investor' in col:
            investors_col = col
        if 'round' in col or 'stage' in col:
            round_col = col
        if 'year' in col or 'date' in col:
            year_col = col
    
    logger.info(f"Using columns: sector={sector_col}, amount={amount_col}, investors={investors_col}, round={round_col}, year={year_col}")
    
    # Parse amounts
    if amount_col:
        df['amount_inr'] = df[amount_col].apply(parse_amount_inr)
    else:
        df['amount_inr'] = np.nan
    
    # Extract year if date column exists
    if year_col and 'year' not in str(year_col).lower():
        try:
            df['year'] = pd.to_datetime(df[year_col], errors='coerce').dt.year
            year_col = 'year'
        except:
            pass
    
    index = {}
    
    if sector_col:
        # Group by sector
        for sector, group in df.groupby(sector_col):
            if pd.isna(sector) or sector == '':
                continue
            
            sector_data = {
                'median_amount_inr': float(group['amount_inr'].median()) if 'amount_inr' in group.columns else 0,
                'yearly_rounds': {},
                'top_investors': [],
                'round_mix': {},
                'rounds_total': len(group)
            }
            
            # Yearly rounds count
            if year_col:
                yearly = group.groupby(year_col).size().to_dict()
                sector_data['yearly_rounds'] = {str(k): int(v) for k, v in yearly.items() if not pd.isna(k)}
            
            # Top investors
            if investors_col:
                investors = []
                for inv_str in group[investors_col].dropna():
                    # Split by common delimiters
                    inv_list = str(inv_str).split(',')
                    investors.extend([inv.strip() for inv in inv_list if inv.strip()])
                
                from collections import Counter
                top_inv = Counter(investors).most_common(5)
                sector_data['top_investors'] = [inv for inv, count in top_inv]
            
            # Round mix
            if round_col:
                round_mix = group[round_col].value_counts().to_dict()
                sector_data['round_mix'] = {str(k): int(v) for k, v in round_mix.items() if not pd.isna(k)}
            
            index[str(sector)] = sector_data
    
    return index


def export_index(index: dict, output_path: str = "data/india_funding_index.json"):
    """Export funding index to JSON."""
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save index
    with open(output_path, 'w') as f:
        json.dump(index, f, indent=2)
    
    logger.info(f"Index exported to {output_path}")
    logger.info(f"Total sectors: {len(index)}")
    
    return output_path


def main():
    # Check for Kaggle credentials
    if not os.getenv('KAGGLE_USERNAME') or not os.getenv('KAGGLE_KEY'):
        logger.warning("KAGGLE_USERNAME and KAGGLE_KEY environment variables not set")
        logger.warning("Please set them to download data from Kaggle")
        return
    
    try:
        # Download dataset
        download_path = download_india_dataset()
        
        # Find CSV file
        csv_files = list(Path(download_path).glob('*.csv'))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {download_path}")
        
        csv_path = str(csv_files[0])
        logger.info(f"Loading data from {csv_path}")
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows")
        
        # Build index
        index = build_sector_index(df)
        
        # Export
        export_index(index)
        
        logger.info("India funding index build complete!")
        
    except Exception as e:
        logger.error(f"Build failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
