"""
Kaggle Training Script
Downloads Kaggle dataset and trains a logistic regression model.
Usage: python train_kaggle.py --dataset-ref justinas/startup-success-prediction --csv-name train.csv --target-col status
"""
import argparse
import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_kaggle_dataset(dataset_ref: str, download_path: str = "./kaggle_data"):
    """Download dataset from Kaggle."""
    try:
        import kaggle
        
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


def prepare_features(df: pd.DataFrame, target_col: str):
    """Prepare minimal feature set for training."""
    # Calculate age if founded_year exists
    if 'founded_year' in df.columns:
        current_year = 2025
        df['age'] = current_year - df['founded_year']
    
    # Define feature columns
    numeric_features = []
    categorical_features = []
    
    # Common numeric features
    for col in ['age', 'total_funding_usd', 'team_size', 'num_investors']:
        if col in df.columns:
            numeric_features.append(col)
    
    # Common categorical features
    for col in ['sector', 'location', 'stage']:
        if col in df.columns:
            categorical_features.append(col)
    
    # Fill missing values
    for col in numeric_features:
        df[col] = df[col].fillna(df[col].median())
    
    for col in categorical_features:
        df[col] = df[col].fillna('unknown')
    
    return df, numeric_features, categorical_features


def train_model(df: pd.DataFrame, target_col: str, numeric_features, categorical_features):
    """Train logistic regression model with preprocessing pipeline."""
    
    # Prepare target
    y = df[target_col].astype(int)
    X = df[numeric_features + categorical_features]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    
    # Create full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # Train model
    logger.info("Training model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_proba)
    logger.info(f"Model AUC: {auc:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))
    
    return pipeline, auc


def export_model(model, output_path: str = "models/startup_model.joblib"):
    """Export trained model in OnlinePredictor-compatible format."""
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save model
    joblib.dump(model, output_path)
    logger.info(f"Model exported to {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Train startup success model from Kaggle dataset")
    parser.add_argument(
        '--dataset-ref',
        default='justinas/startup-success-prediction',
        help='Kaggle dataset reference (owner/dataset-name)'
    )
    parser.add_argument(
        '--csv-name',
        default='train.csv',
        help='CSV filename within the dataset'
    )
    parser.add_argument(
        '--target-col',
        default='status',
        help='Target column name'
    )
    parser.add_argument(
        '--output',
        default='models/startup_model.joblib',
        help='Output path for trained model'
    )
    
    args = parser.parse_args()
    
    # Check for Kaggle credentials
    if not os.getenv('KAGGLE_USERNAME') or not os.getenv('KAGGLE_KEY'):
        logger.warning("KAGGLE_USERNAME and KAGGLE_KEY environment variables not set")
        logger.warning("Please set them to download data from Kaggle")
        return
    
    try:
        # Download dataset
        download_path = download_kaggle_dataset(args.dataset_ref)
        
        # Load CSV
        csv_path = os.path.join(download_path, args.csv_name)
        if not os.path.exists(csv_path):
            # Try to find the first CSV file
            csv_files = list(Path(download_path).glob('*.csv'))
            if csv_files:
                csv_path = str(csv_files[0])
                logger.info(f"Using CSV file: {csv_path}")
            else:
                raise FileNotFoundError(f"No CSV files found in {download_path}")
        
        logger.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows")
        
        # Prepare features
        df, numeric_features, categorical_features = prepare_features(df, args.target_col)
        
        # Train model
        model, auc = train_model(df, args.target_col, numeric_features, categorical_features)
        
        # Export model
        export_model(model, args.output)
        
        logger.info("Training complete!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
