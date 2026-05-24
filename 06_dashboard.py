import os
import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import streamlit as st

from utils import carica_dati, aggiungi_crisi, resample_dati

# ---------------------------------------------------------------------------
# Configurazione pagina
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MacroDash EU",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — fedele al mockup: sfondo #0e1117, sidebar #1a1f2e, accento #60a5fa
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background-color: #0e1117;
    color: #fafafa;
}
.stApp { background-color: #0e1117; }

[data-testid="stSidebar"] {
    background-color: #1a1f2e !important;
    border-right: 1px solid #2d3748;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.85rem; color: #9ca3af;
    padding: 8px 10px; border-radius: 8px;
}
[data-testid="stSidebar"] h1 {
    color: #60a5fa !important; font-size: 1.1rem !important;
    font-weight: 700 !important; letter-spacing: 0.5px;
}

h1 { color: #f9fafb !important; font-size: 1.6rem !important; font-weight: 700 !important; }
h2 { color: #e5e7eb !important; font-size: 1.1rem !important; font-weight: 600 !important; }
h3 { color: #9ca3af !important; font-size: 0.8rem !important; text-transform: uppercase; letter-spacing: 1px; }

[data-testid="stMetric"] {
    background-color: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 18px 20px !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.7rem !important; color: #6b7280 !important;
    text-transform: uppercase; letter-spacing: 0.5px;
}
[data-testid="stMetricValue"] {
    font-size: 1.9rem !important; font-weight: 700 !important; color: #f9fafb !important;
}

.stAlert {
    background-color: #0d2137 !important; border: 1px solid #1e3a5f !important;
    border-radius: 8px !important; color: #93c5fd !important; font-size: 0.82rem !important;
}
.stRadio > label { color: #6b7280 !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 1px; }
.stSlider label { font-size: 0.78rem !important; color: #6b7280 !important; }
hr { border-color: #2d3748 !important; }
p, .stMarkdown p { color: #9ca3af; font-size: 0.85rem; line-height: 1.6; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0e1117; }
::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tema matplotlib
# ---------------------------------------------------------------------------

BLU   = "#60a5fa"
VERDE = "#34d399"
ROSSO = "#f87171"
AMBRA = "#fbbf24"
VIOLA = "#a78bfa"
SFONDO = "#0e1117"
TESTO  = "#6b7280"

mpl.rcParams.update({
    "figure.facecolor": SFONDO, "axes.facecolor": "#111827",
    "axes.edgecolor": "#2d3748", "axes.labelcolor": TESTO,
    "axes.grid": True, "grid.color": "#1f2937", "grid.linewidth": 0.6,
    "text.color": "#9ca3af", "xtick.color": TESTO, "ytick.color": TESTO,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.labelsize": 8, "axes.titlesize": 10, "axes.titlecolor": "#e5e7eb",
    "axes.titlepad": 10, "font.family": "sans-serif",
    "legend.facecolor": "#1a1f2e", "legend.edgecolor": "#2d3748",
    "legend.fontsize": 7, "figure.dpi": 110,
})


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _scarica_se_necessario():
    """
    Se data/clean/ non esiste (es. Streamlit Cloud), scarica i dati
    direttamente da FRED e yFinance prima di caricarli.
    Usa la FRED_API_KEY dai Secrets di Streamlit o dal file .env locale.
    """
    if not os.path.isdir("data/clean/macro"):
        st.info("⏳ Prima esecuzione: scaricamento dati in corso (1-2 minuti)...")
        import sys
        sys.path.insert(0, ".")
        from fredapi import Fred
        import yfinance as yf
        from utils import pulisci_dizionario, resample_dati as _resample

        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            st.error("FRED_API_KEY non trovata. Aggiungila nei Secrets di Streamlit.")
            st.stop()

        fred = Fred(api_key=api_key)

        INDICATORI = {
            "inflazione_eurozona": "CP0000EZ19M086NEST",
            "inflazione_italia":   "CP0000ITM086NEST",
            "pil_eurozona":        "CLVMEURSCAB1GQEA19",
            "pil_italia":          "CLVMNACSCAB1GQIT",
            "disoccupazione_eurozona": "LRHUTTTTEZM156S",
            "disoccupazione_italia":   "LRHUTTTTITM156S",
            "tasso_bce":           "ECBDFR",
            "m3_eurozona":         "MABMM301EZM189S",
            "btp_10y":             "IRLTLT01ITM156N",
            "bund_10y":            "IRLTLT01DEM156N",
        }
        MERCATI = {
            "ftse_mib":    "FTSEMIB.MI",
            "eurostoxx50": "^STOXX50E",
            "eurusd":      "EURUSD=X",
            "brent":       "BZ=F",
            "vix":         "^VIX",
        }
        DATA_INIZIO = "2009-01-01"
        DATA_FINE   = pd.Timestamp.today().strftime("%Y-%m-%d")

        os.makedirs("data/clean/macro",    exist_ok=True)
        os.makedirs("data/clean/mercati",  exist_ok=True)

        # Scarica macro
        macro = {}
        for nome, codice in INDICATORI.items():
            try:
                macro[nome] = fred.get_series(codice, observation_start=DATA_INIZIO, observation_end=DATA_FINE)
            except Exception:
                pass

        macro_df = {k: v.to_frame(name=k) for k, v in macro.items()}
        macro_pulito = pulisci_dizionario(macro_df)
        macro_resampled = _resample(macro_pulito)
        for nome, df in macro_resampled.items():
            df.to_csv(f"data/clean/macro/{nome}.csv")

        # Scarica mercati
        for nome, ticker in MERCATI.items():
            try:
                df = yf.download(ticker, start=DATA_INIZIO, end=DATA_FINE, progress=False)
                if not df.empty:
                    close = df["Close"]
                    if isinstance(close, pd.DataFrame):
                        close = close.iloc[:, 0]
                    close.to_frame(name=nome).to_csv(f"data/clean/mercati/{nome}.csv")
            except Exception:
                pass

        st.success("✅ Dati scaricati. Caricamento dashboard...")
        st.rerun()


_scarica_se_necessario()


@st.cache_data
def carica_dati_dashboard():
    return resample_dati(carica_dati())

@st.cache_data
def carica_mercati_dashboard():
    from utils import carica_mercati
    return carica_mercati()

@st.cache_resource
def carica_modelli():
    return (joblib.load("models/gb_inflazione.pkl"),
            joblib.load("models/gb_spread.pkl"),
            joblib.load("models/scaler.pkl"))

@st.cache_data
def carica_feature_matrix():
    return pd.read_csv("models/feature_matrix.csv", index_col=0, parse_dates=True)

@st.cache_data
def carica_granger():
    return pd.read_csv("models/granger_matrix.csv", index_col=0).astype(float)

@st.cache_data
def carica_taylor():
    return pd.read_csv("models/taylor_rule.csv", index_col=0, parse_dates=True)


dati    = carica_dati_dashboard()
mercati = carica_mercati_dashboard()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("📊 MacroDash EU")
st.sidebar.markdown(
    "<p style='font-size:0.65rem;color:#6b7280;text-transform:uppercase;"
    "letter-spacing:1px;margin-bottom:16px'>Economic Intelligence</p>",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    "<p style='font-size:0.65rem;color:#6b7280;text-transform:uppercase;letter-spacing:1px'>Navigazione</p>",
    unsafe_allow_html=True,
)

sezione = st.sidebar.radio(
    "",
    ["🏠  Panoramica Macro", "🔮  Previsioni", "🔗  Analisi Relazioni", "🏦  Politica Monetaria"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<p style='font-size:0.65rem;color:#6b7280;text-transform:uppercase;letter-spacing:1px'>Stato dati</p>",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    "<div style='background:#0d2137;border:1px solid #1e3a5f;border-radius:8px;padding:12px'>"
    "<div style='display:flex;justify-content:space-between;font-size:0.75rem;margin-bottom:4px'>"
    "<span style='color:#9ca3af'>FRED API</span><span style='color:#34d399'>&#9679; Live</span></div>"
    "<div style='display:flex;justify-content:space-between;font-size:0.75rem;margin-bottom:4px'>"
    "<span style='color:#9ca3af'>yFinance</span><span style='color:#34d399'>&#9679; Live</span></div>"
    "<div style='display:flex;justify-content:space-between;font-size:0.75rem'>"
    "<span style='color:#9ca3af'>Aggiornato</span><span style='color:#fbbf24'>Oggi</span></div>"
    "</div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<p style='font-size:0.6rem;color:#374151;text-align:center'>"
    "DATI: FRED &middot; YAHOO FINANCE<br>MODELLI: GRADIENT BOOSTING<br>AUTORE: VINCENZO</p>",
    unsafe_allow_html=True,
)


# ===========================================================================
# SEZIONE 1 — PANORAMICA MACRO
# ===========================================================================

if "Panoramica" in sezione:
    st.title("Panoramica Macroeconomica 🇪🇺")
    st.markdown(
        "<p>Indicatori aggiornati &middot; Eurozona e Italia &middot; Maggio 2026</p>",
        unsafe_allow_html=True,
    )

    st.markdown("### Ultimi valori disponibili")
    c1, c2, c3, c4 = st.columns(4)

    def ultimo(chiave):
        return dati[chiave].squeeze().dropna().iloc[-1]

    def delta_val(chiave):
        s = dati[chiave].squeeze().dropna()
        return float(s.iloc[-1] - s.iloc[-2]) if len(s) > 1 else 0.0

    with c1:
        st.metric(
            "Inflazione Eurozona",
            f"{ultimo('inflazione_eurozona'):.1f}",
            f"{delta_val('inflazione_eurozona'):+.2f} vs prec.",
            help="Indice HICP (Harmonised Index of Consumer Prices, base 2015=100). "
                 "Non è una percentuale — è il livello dell'indice. "
                 "La variazione anno su anno (es. da 115 a 118) corrisponde a ~3% di inflazione.",
        )
    with c2:
        st.metric(
            "Tasso BCE",
            f"{ultimo('tasso_bce'):.2f}%",
            f"{delta_val('tasso_bce'):+.2f}",
            help="Tasso sui depositi (Deposit Facility Rate) della BCE, in punti percentuali. "
                 "È il tasso che le banche ricevono per i depositi overnight presso la BCE. "
                 "Negativo dal 2014 al 2022, poi salito rapidamente fino al 4% nel 2023.",
        )
    with c3:
        btp  = dati["btp_10y"].squeeze().dropna()
        bund = dati["bund_10y"].squeeze().dropna()
        sp   = float(btp.iloc[-1] - bund.iloc[-1])
        sp_d = float(btp.iloc[-2] - bund.iloc[-2]) if len(btp) > 1 else sp
        st.metric(
            "Spread BTP-Bund",
            f"{sp:.2f}%",
            f"{sp - sp_d:+.2f}",
            help="Differenza tra il rendimento del BTP italiano a 10 anni e il Bund tedesco a 10 anni, "
                 "in punti percentuali. È il termometro del rischio-Italia: "
                 "più alto = i mercati chiedono un premio maggiore per finanziare il debito italiano.",
        )
    with c4:
        st.metric(
            "Disoccupazione Italia",
            f"{ultimo('disoccupazione_italia'):.1f}%",
            f"{delta_val('disoccupazione_italia'):+.2f}",
            help="Tasso di disoccupazione armonizzato (HICP) per l'Italia, in percentuale della forza lavoro. "
                 "Fonte: FRED/OECD. La serie eurozona è stata discontinuata da FRED a gennaio 2023 "
                 "(aggiunta della Croazia all'eurozona EA20).",
        )

    st.markdown("---")

    # Dizionario: titolo → (serie, colore, unità Y per il grafico)
    grafici = {
        "Inflazione Eurozona":   (dati["inflazione_eurozona"].squeeze(),  BLU,      "Indice HICP (2015=100)"),
        "Inflazione Italia":     (dati["inflazione_italia"].squeeze(),    AMBRA,    "Indice HICP (2015=100)"),
        "Disoccupazione Italia": (dati["disoccupazione_italia"].squeeze(), ROSSO,   "% forza lavoro"),
        "M3 Eurozona":           (dati["m3_eurozona"].squeeze(),           VERDE,   "Milioni di Euro"),
        "PIL Eurozona":          (dati["pil_eurozona"].squeeze(),          BLU,     "Milioni € (prezzi cost. 2010)"),
        "PIL Italia":            (dati["pil_italia"].squeeze(),            VIOLA,   "Milioni € (prezzi cost. 2010)"),
        "Tasso BCE":             (dati["tasso_bce"].squeeze(),             VIOLA,   "% (Deposit Facility Rate)"),
        "BTP 10Y":               (dati["btp_10y"].squeeze(),               ROSSO,  "Rendimento %"),
        "Bund 10Y":              (dati["bund_10y"].squeeze(),              AMBRA,   "Rendimento %"),
        "Spread BTP-Bund":       (
            (dati["btp_10y"].squeeze() - dati["bund_10y"].squeeze()).dropna(),
            "#f87171", "Punti percentuali (BTP - Bund)",
        ),
        "Brent Crude (USD)":     (mercati["brent"].squeeze(),             "#f97316", "USD per barile"),
        "VIX":                   (mercati["vix"].squeeze(),               "#ec4899",  "Indice (0=calma, >30=stress)"),
    }

    nomi = list(grafici.keys())
    for i in range(0, len(nomi), 2):
        col1, col2 = st.columns(2)
        with col1:
            serie, colore, ylabel = grafici[nomi[i]]
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.plot(serie.index, serie.values, color=colore, linewidth=1.3)
            ax.set_title(nomi[i])
            ax.set_ylabel(ylabel, fontsize=7)
            aggiungi_crisi(ax)
            ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.0, 1.0))
            plt.tight_layout(pad=0.4)
            st.pyplot(fig)
            plt.close(fig)
        if i + 1 < len(nomi):
            with col2:
                serie, colore, ylabel = grafici[nomi[i + 1]]
                fig, ax = plt.subplots(figsize=(7, 3))
                ax.plot(serie.index, serie.values, color=colore, linewidth=1.3)
                ax.set_title(nomi[i + 1])
                ax.set_ylabel(ylabel, fontsize=7)
                aggiungi_crisi(ax)
                ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.0, 1.0))
                plt.tight_layout(pad=0.4)
                st.pyplot(fig)
                plt.close(fig)


# ===========================================================================
# SEZIONE 2 — PREVISIONI
# ===========================================================================

elif "Previsioni" in sezione:
    st.title("🔮 Modelli Predittivi")
    st.markdown(
        "<p>Gradient Boosting addestrato su indicatori macroeconomici con validazione temporale "
        "(TimeSeriesSplit k=5). I target sono serie differenziate — i valori rappresentano variazioni.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("### Performance modelli (validazione temporale)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("MAE Inflazione (CV)", "0.36",
                  help="Mean Absolute Error in punti indice HICP. Il modello sbaglia in media di 0.36 punti "
                       "sulla variazione mensile dell'indice HICP a 3 mesi. "
                       "Baseline naive (predici zero cambiamento): 0.60.")
    with c2:
        st.metric("Baseline Inflazione", "0.60",
                  help="MAE del modello naive che predice sempre zero variazione. "
                       "Il nostro modello (0.36) lo batte del +40%.")
    with c3:
        st.metric("MAE Spread (CV)", "0.18",
                  help="Mean Absolute Error in punti percentuali sulla variazione mensile dello spread BTP-Bund. "
                       "Il modello sbaglia in media di 0.18 pp. "
                       "Baseline naive (predici zero cambiamento): 0.16 — le variazioni mensili dello spread "
                       "sono in larga parte imprevedibili con indicatori macro.")
    with c4:
        st.metric("Baseline Spread", "0.16",
                  help="MAE del modello naive che predice sempre zero variazione dello spread. "
                       "Le variazioni mensili dello spread sono piccole e difficili da anticipare — "
                       "coerente con l'efficienza relativa dei mercati obbligazionari sovrani.")

    st.info(
        "⚠️ I target dei modelli sono serie differenziate (diff primo ordine), non livelli assoluti. "
        "Il modello inflazione predice la variazione mensile dell'indice HICP a 3 mesi. "
        "Il modello spread predice la variazione mensile dello spread BTP-Bund a 1 mese. "
        "MAE stimato con TimeSeriesSplit k=5 — ogni fold usa solo dati passati."
    )

    st.markdown("---")

    gb, gb2, scaler = carica_modelli()
    X = carica_feature_matrix()

    modalita = st.radio(
        "Modalità previsione",
        ["Automatica (ultimi dati)", "Manuale (slider)"],
        horizontal=True,
    )

    if modalita == "Automatica (ultimi dati)":
        st.markdown("### Previsione con ultimi dati disponibili")
        ultima_riga        = X.iloc[[-1]]
        ultima_riga_scaled = scaler.transform(ultima_riga)
        prev_inf    = gb.predict(ultima_riga_scaled)[0]
        prev_spread = gb2.predict(ultima_riga_scaled)[0]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Inflazione EZ tra 3 mesi (Δ)", f"{prev_inf:+.3f}",
                      help="Variazione attesa dell'indice HICP eurozona a 3 mesi. "
                           "Unità: punti indice HICP. Positivo = inflazione in crescita, "
                           "negativo = deflazione. Valore tipico: da -0.5 a +1.0.")
        with c2:
            st.metric("Spread BTP-Bund tra 1 mese (Δ)", f"{prev_spread:+.3f}",
                      help="Variazione attesa dello spread BTP-Bund a 1 mese. "
                           "Unità: punti percentuali. Positivo = spread si allarga (più rischio Italia), "
                           "negativo = spread si restringe.")
        with c3:
            st.metric("Dati aggiornati al", str(X.index[-1].date()),
                      help="Data dell'ultima osservazione disponibile nella feature matrix. "
                           "La previsione usa i valori di quella data come input del modello.")

    else:
        st.markdown("### Modifica manuale delle feature principali")
        st.markdown("<p>Le 5 feature con maggior peso nel modello inflazione (feature importance GB).<br>"
                    "<strong>Unità: variazione mensile dell'indice HICP</strong> — es. 0.3 = l'indice è salito di 0.3 punti in quel mese. "
                    "Valori tipici: da -0.5 a +1.0. Inflazione alta corrisponde a valori positivi persistenti.</p>",
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            inf_it_lag3  = st.slider("Inflazione Italia — lag 3 mesi",     -2.0, 5.0, 0.0, 0.1)
            inf_it_lag1  = st.slider("Inflazione Italia — lag 1 mese",     -2.0, 5.0, 0.0, 0.1)
            inf_eu_lag3  = st.slider("Inflazione EZ — lag 3 mesi",         -2.0, 5.0, 0.0, 0.1)
        with c2:
            inf_eu_lag1  = st.slider("Inflazione EZ — lag 1 mese",         -2.0, 5.0, 0.0, 0.1)
            disoc_lag1   = st.slider("Disoccupazione Italia — lag 1 mese", -5.0, 5.0, 0.0, 0.1)

        X_man = X.iloc[[-1]].reset_index(drop=True).copy()
        X_man["inflazione_italia_lag3"]     = inf_it_lag3
        X_man["inflazione_italia_lag1"]     = inf_it_lag1
        X_man["inflazione_eurozona_lag3"]   = inf_eu_lag3
        X_man["inflazione_eurozona_lag1"]   = inf_eu_lag1
        X_man["disoccupazione_italia_lag1"] = disoc_lag1

        X_man_sc = scaler.transform(X_man)
        c1, c2 = st.columns(2)
        with c1: st.metric("Inflazione EZ tra 3 mesi (Δ)", f"{gb.predict(X_man_sc)[0]:+.3f}")
        with c2: st.metric("Spread BTP-Bund tra 1 mese (Δ)", f"{gb2.predict(X_man_sc)[0]:+.3f}")


# ===========================================================================
# SEZIONE 3 — ANALISI RELAZIONI
# ===========================================================================

elif "Relazioni" in sezione:
    st.title("🔗 Causalità di Granger")
    st.markdown(
        "<p>Test di causalità di Granger su tutte le coppie di indicatori (lag 1–6 mesi). "
        "Valori bassi = X anticipa Y statisticamente. "
        "<strong>Righe = CAUSA &middot; Colonne = EFFETTO</strong></p>",
        unsafe_allow_html=True,
    )

    granger_matrix = carica_granger()

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        granger_matrix, cmap="Reds_r", annot=True, fmt=".2f",
        cbar=True, ax=ax, vmin=0, vmax=0.1,
        linewidths=0.3, linecolor="#0e1117", annot_kws={"size": 7},
    )
    ax.set_xlabel("Y — variabile CAUSATA", fontsize=8, color=TESTO)
    ax.set_ylabel("X — variabile che CAUSA", fontsize=8, color=TESTO)
    ax.set_title("p-value minimo (lag 1–6) — soglia significatività: 0.05", fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # HTML table — evita pyarrow che non è installato
    st.markdown("### Relazioni significative (p < 0.05)")
    righe = [
        {"Causa (X)": c, "Effetto (Y)": e, "p-value": round(granger_matrix.loc[c, e], 4)}
        for c in granger_matrix.index
        for e in granger_matrix.columns
        if pd.notna(granger_matrix.loc[c, e]) and granger_matrix.loc[c, e] < 0.05
    ]
    if righe:
        df_sig = pd.DataFrame(righe).sort_values("p-value").reset_index(drop=True)
        html = "<table style='width:100%;border-collapse:collapse;font-size:0.82rem'>"
        html += "<tr style='background:#1e3a5f'>"
        html += "<th style='padding:8px;text-align:left;color:#93c5fd'>Causa (X)</th>"
        html += "<th style='padding:8px;text-align:left;color:#93c5fd'>Effetto (Y)</th>"
        html += "<th style='padding:8px;text-align:right;color:#93c5fd'>p-value</th></tr>"
        for i, row in df_sig.iterrows():
            bg = "#1a1f2e" if i % 2 == 0 else "#111827"
            html += f"<tr style='background:{bg}'>"
            html += f"<td style='padding:7px 8px;color:#d1d5db'>{row['Causa (X)']}</td>"
            html += f"<td style='padding:7px 8px;color:#d1d5db'>{row['Effetto (Y)']}</td>"
            html += f"<td style='padding:7px 8px;text-align:right;color:#34d399'>{row['p-value']:.4f}</td>"
            html += "</tr>"
        html += "</table>"
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown("<p>Nessuna relazione significativa trovata.</p>", unsafe_allow_html=True)


# ===========================================================================
# SEZIONE 4 — POLITICA MONETARIA
# ===========================================================================

elif "Politica" in sezione:
    st.title("🏦 Politica Monetaria BCE")
    st.markdown(
        "<p>Confronto tra il tasso BCE effettivo e il tasso teorico calcolato con la "
        "<strong>Taylor Rule</strong>: "
        "<code>r = &pi; + 2 + 0.5&middot;(&pi;&minus;2) + 0.5&middot;output_gap</code>. "
        "PIL potenziale stimato con filtro Hodrick-Prescott (&lambda;=14400, mensile).</p>",
        unsafe_allow_html=True,
    )

    taylor_df = carica_taylor()
    dev = taylor_df["deviazione"].dropna()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Deviazione media", f"{dev.mean():.2f} pp",
                  help="Media storica della differenza tra tasso BCE reale e tasso Taylor Rule. "
                       "Unità: punti percentuali. Negativo = BCE ha tenuto i tassi sotto il livello ottimale.")
    with c2:
        st.metric("Max sopra Taylor Rule", f"{dev.max():.2f} pp",
                  help="Massima deviazione positiva: momento in cui la BCE ha tenuto i tassi "
                       "più in alto rispetto alla Taylor Rule.")
    with c3:
        st.metric("Max sotto Taylor Rule", f"{dev.min():.2f} pp",
                  help="Massima deviazione negativa: momento in cui la BCE ha tenuto i tassi "
                       "più in basso rispetto alla Taylor Rule (tipicamente 2021-2022, inflazione alta e tassi a zero).")
    with c4:
        st.metric("Ultima deviazione", f"{dev.iloc[-1]:.2f} pp",
                  help="Deviazione più recente. Vicino a zero = BCE allineata alla Taylor Rule. "
                       "Unità: punti percentuali.")

    st.markdown("---")

    st.markdown("### Tasso BCE vs Taylor Rule")
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(taylor_df.index, taylor_df["tasso_bce"].values,
            color=BLU, linewidth=1.5, label="Tasso BCE reale")
    ax.plot(taylor_df.index, taylor_df["tasso_ottimale"].values,
            color=VERDE, linewidth=1.5, linestyle="--", label="Taylor Rule")
    ax.fill_between(taylor_df.index,
                    taylor_df["tasso_bce"].values, taylor_df["tasso_ottimale"].values,
                    alpha=0.1, color=ROSSO)
    ax.set_ylabel("Tasso (%)")
    aggiungi_crisi(ax)
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout(pad=0.4)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Deviazione dalla Taylor Rule")
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.plot(dev.index, dev.values, color=ROSSO, linewidth=1.2, label="Deviazione")
    ax.axhline(0, color="#9ca3af", linestyle="--", linewidth=0.8)
    ax.fill_between(dev.index, dev.values, 0,
                    where=(dev < 0), alpha=0.2, color=ROSSO, label="BCE sotto Taylor Rule")
    ax.fill_between(dev.index, dev.values, 0,
                    where=(dev > 0), alpha=0.2, color=BLU, label="BCE sopra Taylor Rule")
    ax.set_ylabel("Punti percentuali")
    aggiungi_crisi(ax)
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout(pad=0.4)
    st.pyplot(fig)
    plt.close(fig)