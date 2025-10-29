# Training Utilities

This directory contains offline training scripts for the ML models.

## Scripts

### train_kaggle.py

Trains a logistic regression model from Kaggle startup datasets.

**Usage:**
```bash
# Set Kaggle credentials
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"

# Run training
python models/training/train_kaggle.py \
  --dataset-ref justinas/startup-success-prediction \
  --csv-name train.csv \
  --target-col status \
  --output models/startup_model.joblib
```

**Output:**
- `models/startup_model.joblib` - Trained model compatible with OnlinePredictor

### build_india_funding_context.py

Downloads and processes India startup funding data to create a sector-wise index.

**Usage:**
```bash
# Set Kaggle credentials
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"

# Run build
python models/training/build_india_funding_context.py
```

**Output:**
- `data/india_funding_index.json` - Sector-wise funding statistics

## Requirements

Both scripts require:
- `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables
- Internet connection to download datasets
- Sufficient disk space for downloaded data

## Notes

- These scripts are meant for offline training and data preparation
- The generated artifacts (model and index) are used by the main application
- Training data is downloaded to temporary directories (`kaggle_data/`, `kaggle_data_india/`)
