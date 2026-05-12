# 📊 Macro Economic Dashboard

Dashboard interattiva sull'economia europea costruita con Python e Streamlit. Progetto personale sviluppato per applicare tecniche di data science e machine learning al dominio della macroeconomia e dei mercati finanziari.

---

# 🎯 Motivazione

L'economia europea è guidata da una rete complessa di indicatori che si influenzano a vicenda — inflazione, tassi di interesse, crescita del PIL, disoccupazione, massa monetaria, spread tra titoli di stato. Questo progetto nasce dalla volontà di capire queste relazioni dai dati, non dai libri.

L'obiettivo finale è uno strumento che permetta a un analista o a un investitore di:
- Monitorare lo stato attuale dell'economia europea in un'unica dashboard
- Capire quali indicatori anticipano gli altri (causalità di Granger)
- Prevedere inflazione e spread BTP-Bund con modelli di machine learning
- Valutare la politica monetaria della BCE rispetto alla Taylor Rule teorica

---

# 🔍 Cosa fa il progetto

# 1. Raccolta dati automatizzata
Scarica automaticamente dati storici dal 2009 ad oggi da due fonti:
- **FRED API** (Federal Reserve Bank of St. Louis) — inflazione HICP, PIL, disoccupazione, tasso BCE, M3, BTP e Bund a 10 anni
- **yFinance** — FTSE MIB, EuroStoxx 50, EUR/USD

# 2. Analisi Esplorativa (EDA)
- Grafici di serie storiche con evidenziazione dei periodi di crisi (Grande Recessione, crisi del debito sovrano, pandemia, crisi energetica)
- Matrice di correlazione tra tutti gli indicatori macro e di mercato
- Statistiche descrittive per ogni serie

# 3. Analisi delle Relazioni
- Test di stazionarietà ADF su tutte le serie
- Test di causalità di Granger su tutte le 56 coppie di indicatori
- Heatmap delle relazioni causali tra variabili

# 4. Modelli Predittivi
- **Target 1**: Inflazione eurozona a 3 mesi
- **Target 2**: Spread BTP-Bund a 1 mese
- Modelli testati: Ridge Regression, Random Forest, Gradient Boosting
- Valutazione con walk-forward validation per evitare look-ahead bias

# 5. Analisi Politica Monetaria BCE
- Implementazione della Taylor Rule
- Stima del PIL potenziale con filtro Hodrick-Prescott
- Confronto tasso BCE reale vs tasso ottimale teorico
- Analisi delle deviazioni storiche dalla regola

# 6. Dashboard Interattiva
Dashboard web costruita con Streamlit con 4 sezioni:
- **Panoramica Macro** — grafici in tempo reale di tutti gli indicatori
- **Previsioni** — output dei modelli con modalità automatica e manuale
- **Analisi Relazioni** — heatmap di Granger interattiva
- **Politica Monetaria** — grafico Taylor Rule e deviazioni storiche

---

# 📈 Risultati Principali

| Modello | Target | MAE | Baseline MAE | Miglioramento |
|---------|--------|-----|--------------|---------------|
| Gradient Boosting | Inflazione eurozona (3m) | 0.57 | 0.70 | +19% |
| Gradient Boosting | Spread BTP-Bund (1m) | 0.56 | 2.00 | +72% |

**Relazioni causali trovate (Granger):**
- PIL eurozona → Disoccupazione eurozona e Italia
- Tassi BCE → Inflazione eurozona e Italia
- M3 eurozona → Inflazione eurozona e Italia
- Inflazione Italia → Inflazione eurozona

**Analisi BCE:**
- La BCE ha tenuto i tassi sistematicamente al di sotto della Taylor Rule dal 2009 al 2022
- La deviazione massima è stata di circa -15 punti percentuali nel 2022
- Dal 2022 i tassi BCE sono convergiti verso i valori suggeriti dalla regola

---

# 🛠️ Tecnologie Utilizzate

| Categoria | Librerie |
|-----------|----------|
| Data manipulation | pandas, numpy |
| Visualizzazione | matplotlib, seaborn |
| Machine Learning | scikit-learn |
| Serie temporali | statsmodels |
| Dati di mercato | yfinance, fredapi |
| Dashboard | streamlit |
| Ambiente | python-dotenv |

---

# 🚀 Come Avviare il Progetto

# Prerequisiti
- Python 3.13
- Account FRED gratuito per la API key: fred.stlouisfed.org

# 1. Clona il repository
```bash
git clone https://github.com/tuousername/macro-economic-dashboard
cd macro-economic-dashboard
```

# 2. Crea e attiva il virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

# 3. Installa le dipendenze
```bash
pip install --only-binary=:all: numpy==2.2.0 pandas matplotlib seaborn fredapi yfinance pandas-datareader requests statsmodels scikit-learn streamlit plotly python-dotenv
pip uninstall pyarrow -y
```

# 4. Configura la API key
Crea un file .env nella cartella principale:
```
FRED_API_KEY=la_tua_chiave_fred
```

# 5. Scarica i dati
```bash
python 01_data_collecting.py
```

# 6. Avvia la dashboard
```bash
streamlit run 06_dashboard.py
```

---

# 📁 Struttura del Progetto

```
macro-economic-dashboard/
├── 01_data_collecting.py
├── 02_eda.py
├── 03_relazioni.py
├── 04_modelli.py
├── 05_politica_monetaria.py
├── 06_dashboard.py
├── utils.py
├── .env                  (non incluso nel repo)
├── .gitignore
└── README.md
```

---

# ⚠️ Note

- I dati non sono inclusi nel repository e vengono scaricati automaticamente
- Su Windows usare sempre --only-binary=:all: durante l'installazione e disinstallare pyarrow

---

# 👤 Autore

Vincenzo Arno

# 📄 Fonti Dati

- FRED - Federal Reserve Bank of St. Louis
- Yahoo Finance
- ECB Data Portal
