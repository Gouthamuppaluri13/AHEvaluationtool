# UI Changes Summary

## New UI Tabs

The application now includes **6 tabs** (previously 4):

### 1. 📝 Investment Memo (Existing - No changes)
- Executive Summary
- Bull Case narrative with subsections
- Bear Case narrative with subsections

### 2. 📈 Risk & Financial Simulation (Existing - No changes)
- Heuristic Risk Profile spider chart
- Financial Runway Simulation with narrative and line chart

### 3. 🌐 Market Deep-Dive & Comps (Enhanced)
**New Additions:**

#### India Funding Trends (News-Derived)
- **3-column metrics display:**
  - Median Round Size (₹X Cr)
  - Total Rounds (from news)
  - Top Investors Found (count)
- **Round Distribution by Stage:**
  - List showing: Series A: X, Seed: Y, etc.
- **Top Investors:**
  - Comma-separated list of top 5 investors

#### Recent News
- **Expandable sections** for each news article
- Shows: Title, Content, "Read more" link
- Up to 5 most relevant news items

#### India Funding Dataset Context (Kaggle-derived)
- **2-column layout:**
  - Left: Median Round Size (Dataset), Total Rounds (Dataset)
  - Right: Yearly Distribution (last 5 years)
- **Top Investors (Historical):** List from dataset
- **Round Type Mix:** Breakdown by round types

### 4. 💸 Fundraise Forecast (NEW)
**3-column metrics display:**
- **Round Probability (6 months):** XX.X%
- **Round Probability (12 months):** XX.X%
- **Expected Time to Next Round:** X.X months

Info box explaining predictions are based on calibrated models.

### 5. 🤖 ML Predictions (NEW)

#### Online Model Results
**3-column metrics:**
- Round Probability (12m): XX.X%
- Round Probability (6m): XX.X%
- Predicted Valuation: $X,XXX,XXX (or N/A)

#### Feature Importances (Top 20)
- **Horizontal bar chart** using Plotly
- Features sorted by importance (highest to lowest)
- Red accent color (#E60023) matching app theme
- Dark theme background

#### Model Metadata
- Model Type: LogisticRegression / GradientBoostingRegressor / etc.
- Source: online_model / fallback

#### Legacy Model Results (For Comparison)
**2-column metrics:**
- Success Probability: XX.X%
- Next Valuation: $X,XXX,XXX

### 6. 📥 Submitted Inputs (Existing - No changes)
- JSON display of all submitted input parameters

## Theme Consistency

All new components maintain the existing dark theme:
- Background: #0D1117
- Text: #E6EDF3
- Accent: #E60023 (red)
- Containers: #161B22
- Borders: #30363D
- Font: Roboto Mono for headers, Inter for body

## Responsive Design

- All metrics use Streamlit's native columns for responsive layout
- Charts adapt to container width with `use_container_width=True`
- Info/warning boxes provide context when data is unavailable

## Graceful Degradation

When models/APIs/data are unavailable:
- Fundraise Forecast: Shows warning message
- ML Predictions: Shows warning message
- Market Deep-Dive sections: Only show when data is available
- No crashes or errors - always shows fallback UI

## Example States

### With Full Data
- All 6 tabs populated
- India trends showing multiple metrics
- Feature importance chart with 20 features
- Both online and legacy predictions visible

### Without Models
- Fundraise Forecast: Warning about missing training data
- ML Predictions: Warning about unavailable predictions
- Other tabs work normally

### Without API Keys
- India funding trends: Empty (no news data)
- Recent news: Empty
- Other analyses continue with what's available

## Visual Elements

### New Charts
1. **Feature Importance Horizontal Bar Chart**
   - Plotly horizontal bar chart
   - Up to 20 features
   - Sorted descending by importance
   - Dark theme with red accent

### New Metrics Cards
2. **Fundraise Forecast Cards**
   - 3 metrics in a row
   - Percentage formatting for probabilities
   - Decimal formatting for months

3. **ML Prediction Cards**
   - 3 metrics for online model
   - 2 metrics for legacy model
   - Currency formatting for valuations

### New Expandables
4. **Recent News Expanders**
   - Title as expander header
   - Content inside
   - Link at bottom

## Summary

The UI now provides:
- **2 completely new analysis tabs** with rich visualizations
- **3 new sections** in the Market Deep-Dive tab
- **Consistent dark theme** throughout
- **Graceful handling** of missing data
- **Professional layout** with proper spacing and organization
