# 💡 Smart Salary App — Frontend

A unified Streamlit application that merges three sub-systems:
- 💰 **Financial Model** — XGBoost salary allocation
- 🛒 **Smart Shopping List** — AI-powered monthly shopping from real Carrefour data
- 📊 **Smart Budget Insights** — Expense tracker with smart recommendations

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and configure environment
cp .env.example .env
# Edit .env → set DATA_GITHUB_RAW_BASE and GOOGLE_API_KEY

# 3. Run
streamlit run app.py
```

## 📡 Data Source

This app reads **two CSV files** produced by the backend pipeline:

| File | Description |
|------|-------------|
| `data/latest_products.csv` | Latest Carrefour products (scraped) |
| `data/predictions.csv` | Price predictions from ML ensemble |

These files are populated by the **`prediction-pipeline`** repository and pushed to GitHub.
Streamlit Cloud reads them on each session.

## 🏗️ Architecture

```
app.py                    ← Navigation entry point
pages/
  financial.py            ← Financial allocation model (XGBoost)
  shopping.py             ← Smart shopping list (reads CSVs)
  insights.py             ← Budget tracker (SQLite, local state)
core/
  data_loader.py          ← ONLY module that reads CSV files
  financial_engine.py     ← ML inference logic (no Streamlit)
  constants.py            ← Shared constants
shopping_modules/         ← Business logic from Shopping_List project
insights_modules/         ← Business logic from Smart Insights project
models/
  multioutput_xgb_model.pkl  ← Financial allocation model
data/
  latest_products.csv     ← Written by pipeline, read by shopping page
  predictions.csv         ← Written by pipeline, read by shopping page
```

## ⚠️ Rules

- ❌ Streamlit does NOT run scraping
- ❌ Streamlit does NOT train ML models
- ❌ Streamlit does NOT generate CSV files
- ✅ Streamlit ONLY reads pre-generated files from `data/`

## 🌐 Deploying on Streamlit Cloud

1. Push this repo to GitHub
2. Connect to [share.streamlit.io](https://share.streamlit.io)
3. Set secrets in Streamlit Cloud:
   - `DATA_GITHUB_RAW_BASE`
   - `GOOGLE_API_KEY` (if using NLP features)
# MUDABER
