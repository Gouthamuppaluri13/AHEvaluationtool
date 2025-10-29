# System Architecture & Data Flow

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Streamlit UI (app.py)                    │
│  Tabs: Investment Memo | Risk | Market | Fundraise | ML | Input │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│               NextGenVCEngine (next_gen_vc_engine.py)            │
│  Orchestrates all analysis components in parallel                │
└──┬──────────────┬─────────────┬───────────────┬─────────────────┘
   │              │             │               │
   │              │             │               │
   ▼              ▼             ▼               ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐
│ Legacy   │  │ Online   │  │ Data     │  │ Fundraise        │
│ AI+      │  │ ML       │  │ Enrich   │  │ Forecast         │
│ Model    │  │ Predictor│  │ Service  │  │ Service          │
└──────────┘  └────┬─────┘  └────┬─────┘  └────┬─────────────┘
                   │             │              │
                   ▼             │              ▼
            ┌──────────────┐    │       ┌─────────────────┐
            │ Model        │    │       │ Round           │
            │ Registry     │    │       │ Models          │
            └──────────────┘    │       │ - Likelihood    │
                   │             │       │ - Time          │
                   │             │       └─────────────────┘
                   ▼             │
            ┌──────────────┐    │
            │ HuggingFace  │    │
            │ Hub / URL /  │    │
            │ Local        │    │
            └──────────────┘    │
                                │
                                ▼
                         ┌──────────────┐
                         │ Adapters     │
                         │ - Tavily     │
                         │ - Indian     │
                         │   Funding    │
                         │ - News       │
                         └──────────────┘
```

## Data Flow for Comprehensive Analysis

### Step 1: Input Collection (app.py)
```
User Input → {
  company_name, sector, stage, founder_bio,
  product_desc, arr, burn, cash, team_size,
  num_investors, funding, metrics...
}
```

### Step 2: Parallel Processing (engine)
```
                    ┌─────────────────┐
                    │   Engine Init   │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │ Legacy Model │ │ Online Model │ │ Enrichment   │
    │ Prediction   │ │ Prediction   │ │ Service      │
    └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
           │                │                │
           │                │                ├─► Tavily News Search
           │                │                ├─► India Funding Trends
           │                │                └─► Dataset Context
           │                │
           └────────────────┴────────┐
                                     │
                    ┌────────────────▼────────────┐
                    │  Market Intelligence (LLM)  │
                    └────────────────┬────────────┘
                                     │
                    ┌────────────────▼────────────┐
                    │   Fundraise Forecast        │
                    └────────────────┬────────────┘
                                     │
                    ┌────────────────▼────────────┐
                    │   Investment Memo (LLM)     │
                    └────────────────┬────────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │ Complete Report │
                            └─────────────────┘
```

### Step 3: Report Assembly
```
Report = {
  'final_verdict': {...},
  'investment_memo': {...},
  'risk_matrix': {...},
  'simulation': {...},
  'market_deep_dive': {
    ...existing fields...,
    'indian_funding_trends': {...},      # NEW
    'recent_news': [...],                # NEW
    'india_funding_dataset_context': {}  # NEW
  },
  'public_comps': {...},
  'profile': {...},
  'ssq_report': {...},
  'online_ml_prediction': {...},         # NEW
  'legacy_ml_prediction': {...},
  'fundraise_forecast': {...}            # NEW
}
```

### Step 4: UI Rendering
```
Report → Streamlit Components → User Interface

Tab 1: Investment Memo (memo data)
Tab 2: Risk & Simulation (risk, simulation data)
Tab 3: Market Deep-Dive (market_deep_dive + enrichment)
Tab 4: Fundraise Forecast (fundraise_forecast)        # NEW
Tab 5: ML Predictions (online/legacy predictions)     # NEW
Tab 6: Inputs (raw input JSON)
```

## Model Loading Priority

```
┌──────────────────────────────────────┐
│        Model Registry                │
└──────────┬───────────────────────────┘
           │
           ▼
    Try #1: HuggingFace Hub
           │ (HUGGINGFACE_MODEL_ID)
           ├─► Success → Return path
           ├─► Fail → Next
           │
           ▼
    Try #2: Direct URL
           │ (MODEL_ASSET_URL)
           ├─► Success → Return path
           ├─► Fail → Next
           │
           ▼
    Try #3: Local Path
           │ (LOCAL_MODEL_PATH)
           ├─► Success → Return path
           ├─► Fail → None
           │
           ▼
    OnlinePredictor:
    if model_path:
        use_model()
    else:
        use_heuristic_fallback()
```

## Training Pipeline (Offline)

```
┌────────────────────────────────────────┐
│  Kaggle Dataset                        │
│  (justinas/startup-success-prediction) │
└────────────┬───────────────────────────┘
             │
             ▼
      ┌──────────────┐
      │ Download CSV │
      └──────┬───────┘
             │
             ▼
      ┌──────────────────┐
      │ Feature Engineer │
      │ - age            │
      │ - funding        │
      │ - team_size      │
      │ - categoricals   │
      └──────┬───────────┘
             │
             ▼
      ┌─────────────────────┐
      │ Train Pipeline:     │
      │ StandardScaler +    │
      │ OneHotEncoder +     │
      │ LogisticRegression  │
      └──────┬──────────────┘
             │
             ▼
      ┌─────────────────┐
      │ Export .joblib  │
      └──────┬──────────┘
             │
             ▼
      Upload to HuggingFace
      or Set LOCAL_MODEL_PATH
```

## India Funding Context Pipeline (Offline)

```
┌──────────────────────────────────────┐
│  Kaggle Dataset                      │
│  (startup-funding-india-cleaned)     │
└────────────┬─────────────────────────┘
             │
             ▼
      ┌──────────────┐
      │ Download CSV │
      └──────┬───────┘
             │
             ▼
      ┌────────────────────┐
      │ Parse by Sector:   │
      │ - median_amount    │
      │ - yearly_rounds    │
      │ - top_investors    │
      │ - round_mix        │
      └──────┬─────────────┘
             │
             ▼
      ┌──────────────────────────┐
      │ Export JSON:             │
      │ data/india_funding_      │
      │ index.json               │
      └──────────────────────────┘
```

## Error Handling Strategy

```
Every Component:
┌─────────────────┐
│ Try Operation   │
└────────┬────────┘
         │
    ┌────▼────┐
    │ Success?│
    └────┬────┘
         │
    Yes  │  No
    ▼    │  ▼
┌────────┐  ┌──────────────┐
│ Return │  │ Log Warning  │
│ Result │  │ Return       │
└────────┘  │ Fallback     │
            └──────────────┘

Examples:
- Model not found → Use heuristics
- API key missing → Skip enrichment
- Training data absent → Use defaults
- JSON parse error → Return empty dict
```

## Secrets Management

```
Required (Already Set):
- TAVILY_API_KEY
- ALPHA_VANTAGE_KEY  
- GEMINI_API_KEY

Optional (For ML Models):
- HUGGINGFACE_MODEL_ID
- MODEL_ARTIFACT_NAME
- MODEL_ASSET_URL
- LOCAL_MODEL_PATH

Optional (For Training):
- KAGGLE_USERNAME
- KAGGLE_KEY
```
