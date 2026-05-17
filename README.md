# 📊 Macro Economic Dashboard

An interactive macroeconomic dashboard for the European economy, built with Python and Streamlit. A personal data science project applying machine learning and econometric methods to macroeconomics and financial markets.

🔗 **[Live Demo →](https://your-app-name.streamlit.app)** *(update link after deploy)*

---

## 🎯 Motivation

The European economy is driven by a complex network of interdependent indicators — inflation, interest rates, GDP growth, unemployment, money supply, sovereign bond spreads, commodity prices, and market volatility. This project was built to understand these relationships from data, not from textbooks.

The goal is a tool that allows an analyst or investor to:
- Monitor the current state of the European economy in a single dashboard
- Understand which indicators statistically lead others (Granger causality)
- Forecast inflation and BTP-Bund spread using machine learning
- Evaluate ECB monetary policy against the theoretical Taylor Rule

---

## 🔍 What the project does

### 1. Automated data collection
Downloads historical data from 2009 to the present from two sources:
- **FRED API** (Federal Reserve Bank of St. Louis) — HICP inflation (eurozone + Italy), GDP, unemployment (Italy), ECB deposit rate, M3, BTP and Bund 10Y yields
- **yFinance** — FTSE MIB, EuroStoxx 50, EUR/USD, **Brent Crude Oil**, **VIX**

> **Note on eurozone unemployment:** The FRED series `LRHUTTTTEZM156S` (EA19) was officially discontinued in January 2023 when Croatia joined the eurozone (EA20). The project uses Italian unemployment as a proxy, which has full coverage through 2026.

### 2. Exploratory Data Analysis (EDA)
- Time series charts with crisis period highlighting (Great Recession, sovereign debt crisis, pandemic, energy crisis)
- Correlation matrix across all macro and market indicators (including Brent and VIX)
- Descriptive statistics for each series

### 3. Relationship Analysis
- ADF stationarity tests on all series
- Granger causality tests on all indicator pairs (including Brent and VIX)
- Interactive causal relationship heatmap

### 4. Predictive Models
- **Target 1**: Eurozone HICP change — 3 months ahead (`diff(1).shift(-3)`)
- **Target 2**: BTP-Bund spread change — 1 month ahead (`diff(1).shift(-1)`)
- Feature engineering: lag 1 and lag 3 months for all series (reduced from 1/3/6 to improve feature/observation ratio)
- Features excluded from ML: `disoccupazione_eurozona` (discontinued series), `m3_eurozona` (negligible importance)
- Features added: `brent` (energy inflation driver), `vix` (risk-off spread driver)
- Models tested: Ridge (α=300), Random Forest, Gradient Boosting
- **Auto-selects best model** per target based on test MAE
- Evaluation with TimeSeriesSplit (k=5) — no look-ahead bias

### 5. ECB Monetary Policy Analysis
- Taylor Rule: `r = π + 2 + 0.5·(π−2) + 0.5·output_gap`
- Potential GDP estimated with Hodrick-Prescott filter (λ=14400 for monthly data)
- ECB actual rate vs. theoretical optimal rate comparison
- Historical deviation analysis

### 6. Interactive Dashboard
Streamlit web app with 4 sections, dark theme, inline unit explanations and tooltips:
- **Macro Overview** — KPI cards with tooltips + real-time charts with Y-axis labels
- **Forecasts** — model output with automatic and manual (slider) mode
- **Relationship Analysis** — interactive Granger heatmap + HTML significance table (no pyarrow dependency)
- **Monetary Policy** — Taylor Rule chart and historical deviations

---

## 📈 Key Results

| Model | Target | MAE (split 80/20) | MAE (CV k=5) | Baseline MAE | Improvement |
|-------|--------|-------------------|--------------|--------------|-------------|
| **Ridge (α=300)** | Eurozone HICP change (3m) | 0.36 | 0.36 | 0.60 | **+40%** |
| Ridge (α=300) | BTP-Bund spread change (1m) | 0.19 | 0.18 | 0.16 | ~flat |

**Why Ridge beats Gradient Boosting here:**
With ~120 observations and ~28 features, GB consistently overfits. Ridge with strong L2 regularization (α=300) forces coefficients toward zero, generalizing better on this narrow dataset. This is the statistically correct model choice for wide, short time series.

**On inflation (+40%):**
Ridge outperforms the naive baseline consistently across all 5 CV folds. The dominant feature is `inflazione_italia_lag3` (GB importance: 0.56) — Italian inflation leads eurozone inflation by ~3 months, directly confirmed by Granger causality.

**On spread (~flat):**
Monthly spread changes are largely unpredictable with macroeconomic lag features. 3 out of 5 CV folds beat the baseline; fold 4 fails significantly. This is consistent with partial bond market efficiency — spread changes in normal conditions have some predictable component, but stress events (ECB decisions, geopolitical shocks) dominate and are not anticipated by macro lags.

**Granger causality findings:**
- GDP eurozone → Italian unemployment (Okun's Law)
- ECB rates → eurozone and Italian inflation
- Italian inflation → eurozone inflation (Italy leads by ~3 months)
- Brent → eurozone inflation (energy channel)
- VIX → BTP-Bund spread (flight to quality)

**ECB analysis:**
- BCE held rates systematically below the Taylor Rule from 2009 to 2022 (negative rates 2014-2022)
- Maximum deviation ~-15pp in 2022 (inflation at 10%, rates at 0%)
- Since 2022, rates converged toward Taylor Rule values (+450bp in 14 months — fastest cycle in ECB history)

---

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| Data manipulation | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Machine Learning | scikit-learn (Ridge, RF, GB, StandardScaler, TimeSeriesSplit) |
| Time series / Econometrics | statsmodels (ADF, Granger, HP filter) |
| Market data | yfinance, fredapi |
| Dashboard | streamlit |
| Environment | python-dotenv, joblib |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Free FRED API key: [fred.stlouisfed.org](https://fred.stlouisfed.org)

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/macro-economic-dashboard
cd macro-economic-dashboard
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
# Windows — use --only-binary to avoid numpy DLL issues
pip install --only-binary=:all: numpy==2.2.0
pip install -r requirements.txt
pip uninstall pyarrow -y   # causes silent pandas crash
```

### 4. Set up the API key
Create a `.env` file in the project root:
```
FRED_API_KEY=your_fred_key_here
```

### 5. Run the pipeline in order
```bash
python 01_data_collecting.py    # download and save data (levels, not transformed)
python 02_eda.py                # exploratory analysis and charts
python 03_relazioni.py          # ADF and Granger tests
python 04_modelli.py            # model training + auto-selection
python 05_politica_monetaria.py # Taylor Rule
streamlit run 06_dashboard.py   # launch dashboard
```

---

## 📁 Project Structure

```
macro-economic-dashboard/
├── 01_data_collecting.py   # download, clean, save levels to data/clean/
├── 02_eda.py               # charts, correlation matrix
├── 03_relazioni.py         # ADF, Granger (applies transformations in memory)
├── 04_modelli.py           # feature engineering, Ridge/RF/GB, auto-selection
├── 05_politica_monetaria.py # Taylor Rule, HP filter
├── 06_dashboard.py         # Streamlit dashboard
├── utils.py                # shared functions: load, clean, plot, transform
├── requirements.txt
├── packages.txt
├── .env                    (not in repo)
├── .gitignore
├── README.md
└── NOTE.md
```

### Key design decisions
- **Clean files store levels** (not transformed): each analytical script applies its own transformations in memory. This allows the dashboard to display actual values while models use differenced series.
- **Transformations applied in-memory**: `03_relazioni.py`, `04_modelli.py`, `05_politica_monetaria.py` all call `resample_dati()` + `applica_trasformazioni()` after loading.
- **No pyarrow dependency**: all Streamlit table displays use HTML rendering to avoid the pyarrow crash on Windows.

---

## ⚠️ Notes

- Data is not included in the repository — downloaded automatically by `01_data_collecting.py`
- On Windows, always use `--only-binary=:all:` when installing numpy and uninstall pyarrow
- The eurozone unemployment FRED series (EA19) was discontinued January 2023 — Italian series used instead
- M3 eurozone excluded from ML models (feature importance < 0.04) but kept in EDA and Granger analysis

---

## 👤 Author

Vincenzo — Data Science student

## 📄 Data Sources

- [FRED — Federal Reserve Bank of St. Louis](https://fred.stlouisfed.org)
- [Yahoo Finance](https://finance.yahoo.com)
- [ECB Data Portal](https://data.ecb.europa.eu)