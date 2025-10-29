# Anthill AI+ Evaluation Tool - ML & Enrichment Features

## Overview

This enhanced version adds online ML predictions, India-specific market enrichment, and fundraising forecasting capabilities to the Anthill AI+ Evaluation Tool.

## New Features

### 1. Online ML Model Loader
- Loads models from multiple sources with priority fallback:
  1. Hugging Face Hub (env: `HUGGINGFACE_MODEL_ID`, `MODEL_ARTIFACT_NAME`)
  2. Direct URL (env: `MODEL_ASSET_URL`)
  3. Local path (env: `LOCAL_MODEL_PATH`)
- Graceful fallback to heuristics when models are unavailable
- See `services/model_registry.py` and `models/online_predictor.py`

### 2. India-Specific Market Enrichment
- Real-time news analysis from Indian startup ecosystem sources
- Sector-wise funding trends and investor intelligence
- Dataset-derived context from Kaggle India funding data
- See `adapters/` and `services/data_enrichment.py`

### 3. Fundraising Forecast
- Calibrated probability predictions for 6m and 12m funding rounds
- Time-to-next-financing estimates
- Models trained on historical deal data
- See `models/round_models.py` and `services/fundraise_forecast.py`

### 4. New UI Tabs
- **💸 Fundraise Forecast**: Round probabilities and timing predictions
- **🤖 ML Predictions**: Online model outputs with feature importance charts
- **Enhanced Market Deep-Dive**: India funding trends, recent news, and dataset context

## Configuration

### Required API Keys (Already Set)
```bash
TAVILY_API_KEY=your_key
ALPHA_VANTAGE_KEY=your_key
GEMINI_API_KEY=your_key
```

### Model Configuration
```bash
# Option 1: Hugging Face
HUGGINGFACE_MODEL_ID=Saigouthamuppaluri/StartupEvaluator
MODEL_ARTIFACT_NAME=startup_model.joblib

# Option 2: Direct URL
MODEL_ASSET_URL=https://example.com/model.joblib

# Option 3: Local path
LOCAL_MODEL_PATH=/path/to/model.joblib
```

### Training-Only Configuration
```bash
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

## Training Utilities

### Train Model from Kaggle
```bash
python models/training/train_kaggle.py
```

### Build India Funding Context
```bash
python models/training/build_india_funding_context.py
```

See `models/training/README.md` for details.

## Running the Application

```bash
streamlit run app.py
```

The app will gracefully handle missing secrets/models with informative fallbacks.

## Architecture

```
.
├── adapters/              # External service adapters
│   ├── tavily_client.py          # Tavily API wrapper
│   ├── indian_funding_adapter.py # India funding news parser
│   └── news_adapter.py           # General news aggregator
├── services/              # Business logic services
│   ├── model_registry.py         # Multi-source model loader
│   ├── data_enrichment.py        # Market enrichment orchestrator
│   └── fundraise_forecast.py     # Fundraising prediction service
├── models/                # ML models and training
│   ├── online_predictor.py       # Online prediction interface
│   ├── round_models.py           # Round likelihood & time models
│   └── training/                 # Offline training scripts
│       ├── train_kaggle.py
│       └── build_india_funding_context.py
├── data/                  # Data artifacts
│   └── india_funding_index.json  # Sector-wise funding stats
├── app.py                 # Streamlit UI (updated)
├── next_gen_vc_engine.py  # Core engine (updated)
└── requirements.txt       # Dependencies (updated)
```

## Test Plan

1. **Without Secrets**: App runs with fallbacks
   ```bash
   streamlit run app.py
   ```

2. **With Kaggle Training**: Generate model
   ```bash
   export KAGGLE_USERNAME=xxx
   export KAGGLE_KEY=xxx
   python models/training/train_kaggle.py
   export LOCAL_MODEL_PATH=models/startup_model.joblib
   streamlit run app.py
   ```

3. **With India Index**: Generate and view context
   ```bash
   python models/training/build_india_funding_context.py
   streamlit run app.py
   ```

## Key Design Decisions

- **Graceful Degradation**: All new features have fallbacks to ensure the app never crashes
- **Parallel Execution**: Online and legacy ML models run in parallel
- **Minimal Breaking Changes**: All existing functionality preserved
- **Modular Design**: Clear separation of concerns (adapters, services, models)

## Dependencies Added

- `requests`: HTTP client for API calls
- `lifelines`: Survival analysis for time-to-financing
- `joblib`: Model serialization
- `huggingface_hub`: Model downloads from HF Hub
- `kaggle`: Dataset downloads
- `scipy`: Scientific computing (implicit dependency)
- `python-dotenv`: Environment variable management
