import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import carica_dati, carica_mercati, plot_timeseries

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

plt.style.use("seaborn-v0_8")
os.makedirs("output/grafici", exist_ok=True)


# ---------------------------------------------------------------------------
# Funzione di supporto: salva figura
# ---------------------------------------------------------------------------

def salva_figura(fig: plt.Figure, titolo: str) -> None:
    """
    Salva la figura in output/grafici/.
    Separare il salvataggio dalla costruzione del grafico permette
    a plot_timeseries di restare generica (non sa nulla di file system).

    Il nome del file viene sanitizzato: caratteri illegali su Windows
    (/ \\ : * ? " < > |) vengono sostituiti con underscore.
    Il titolo del grafico resta invariato — la sanitizzazione riguarda
    solo il percorso su disco.
    """
    nome_file = titolo
    for char in r'/\:*?"<>|':
        nome_file = nome_file.replace(char, "_")
    percorso = f"output/grafici/{nome_file}.png"
    fig.savefig(percorso, dpi=300, bbox_inches="tight")
    print(f"  Salvato: {percorso}")


# ---------------------------------------------------------------------------
# Funzione: grafico singola serie con salvataggio
# ---------------------------------------------------------------------------

def grafico_serie(
    dati: dict,
    chiave: str,
    color: str,
    ylabel: str,
    titolo: str,
) -> None:
    """
    Crea, mostra e salva il grafico di una singola serie dal dizionario dati.
    Centralizza la logica ripetuta nei ~12 plot_timeseries del file originale.
    """
    if chiave not in dati:
        print(f"  [ATTENZIONE] Serie '{chiave}' non trovata — saltata.")
        return

    df = dati[chiave]
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_timeseries(
        ax=ax,
        x=df.index,
        y=df.squeeze().values,
        color=color,
        xlabel="date",
        ylabel=ylabel,
        title=titolo,
    )
    plt.tight_layout()
    salva_figura(fig, titolo)
    plt.close(fig)   # libera memoria — importante quando si generano molti grafici


# ---------------------------------------------------------------------------
# Caricamento dati
# ---------------------------------------------------------------------------

print("Caricamento dati macro...")
dati = carica_dati()

print("Caricamento dati mercati...")
mercati = carica_mercati()

# ---------------------------------------------------------------------------
# Statistiche descrittive
# ---------------------------------------------------------------------------

print("\n" + "=" * 50)
print("STATISTICHE DESCRITTIVE — MACRO")
print("=" * 50)
for nome, df in dati.items():
    print(f"\n{nome}:")
    print(df.describe().round(2))

# ---------------------------------------------------------------------------
# Grafici serie macro
# ---------------------------------------------------------------------------

print("\n" + "=" * 50)
print("GRAFICI SERIE MACRO")
print("=" * 50)

GRAFICI_MACRO = [
    ("disoccupazione_eurozona", "#C0392B", "Disoccupazione (%)",      "Percentuale di disoccupazione eurozona"),
    ("disoccupazione_italia",   "#C0392B", "Disoccupazione (%)",      "Percentuale di disoccupazione italia"),
    ("inflazione_eurozona",     "#E67E22", "Inflazione (HICP)",       "Consumer Price Index europeo"),
    ("inflazione_italia",       "#E67E22", "Inflazione (HICP)",       "Consumer Price Index italiano"),
    ("m3_eurozona",             "#27AE60", "M3 (milioni €)",          "Massa monetaria M3 europea"),
    ("pil_italia",              "#2980B9", "PIL (milioni €)",         "PIL italia (milioni di Euro)"),
    ("pil_eurozona",            "#2980B9", "PIL (milioni €)",         "PIL eurozona (milioni di Euro)"),
    ("tasso_bce",               "#8E44AD", "Tasso BCE (%)",           "Tassi di interesse BCE"),
    ("btp_10y",                 "#E74C3C", "Rendimento (%)",          "Rendimento BTP italiano a 10 anni"),
    ("bund_10y",                "#F39C12", "Rendimento (%)",          "Rendimento Bund tedesco a 10 anni"),
]

for chiave, colore, ylabel, titolo in GRAFICI_MACRO:
    # BTP e Bund sono in dati (macro), non in mercati
    sorgente = dati if chiave in dati else mercati
    grafico_serie(sorgente, chiave, colore, ylabel, titolo)

# ---------------------------------------------------------------------------
# Grafici serie mercati
# ---------------------------------------------------------------------------

print("\n" + "=" * 50)
print("GRAFICI SERIE MERCATI")
print("=" * 50)

GRAFICI_MERCATI = [
    ("eurostoxx50", "#2980B9", "Indice",     "Indice azionario eurozona (EuroStoxx 50)"),
    ("eurusd",      "#27AE60", "EUR/USD",    "Tasso di cambio EUR/USD"),
    ("ftse_mib",    "#2C3E50", "Indice",     "Indice azionario italiano (FTSE MIB)"),
    ("brent",       "#E67E22", "USD/barile", "Brent Crude Oil (USD/barile)"),
    ("vix",         "#8E44AD", "Indice",     "VIX — Indice di volatilità globale"),
]

for chiave, colore, ylabel, titolo in GRAFICI_MERCATI:
    grafico_serie(mercati, chiave, colore, ylabel, titolo)

# ---------------------------------------------------------------------------
# Heatmap correlazione macro + mercati
# ---------------------------------------------------------------------------

print("\n" + "=" * 50)
print("HEATMAP CORRELAZIONE MACRO + MERCATI")
print("=" * 50)

# Ricampiona tutto a frequenza mensile per poter fare il concat
def a_mensile(df: pd.DataFrame) -> pd.Series:
    return df.squeeze().resample("MS").last()

serie_macro = {
    "disoccupazione_eurozona": a_mensile(dati["disoccupazione_eurozona"]) if "disoccupazione_eurozona" in dati else None,
    "disoccupazione_italia":   a_mensile(dati["disoccupazione_italia"]),
    "inflazione_eurozona":     a_mensile(dati["inflazione_eurozona"]),
    "inflazione_italia":       a_mensile(dati["inflazione_italia"]),
    "m3_eurozona":             a_mensile(dati["m3_eurozona"]),
    "pil_italia":              dati["pil_italia"].squeeze().resample("MS").ffill(),
    "pil_eurozona":            dati["pil_eurozona"].squeeze().resample("MS").ffill(),
    "tasso_bce":               dati["tasso_bce"].squeeze().resample("MS").last(),
    "btp_10y":                 a_mensile(dati["btp_10y"]),
    "bund_10y":                a_mensile(dati["bund_10y"]),
}
# Rimuove serie None (disoccupazione_eurozona se non scaricata)
serie_macro = {k: v for k, v in serie_macro.items() if v is not None}

serie_mercati = {
    "eurostoxx50": a_mensile(mercati["eurostoxx50"]),
    "eurusd":      a_mensile(mercati["eurusd"]),
    "ftse_mib":    a_mensile(mercati["ftse_mib"]),
    "brent":       a_mensile(mercati["brent"]),
    "vix":         a_mensile(mercati["vix"]),
}

df_combinato = pd.DataFrame({**serie_macro, **serie_mercati})
matrice_corr = df_combinato.corr()

fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(
    matrice_corr,
    cmap="coolwarm",
    annot=True,
    fmt=".2f",
    cbar=True,
    ax=ax,
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
plt.tight_layout()
salva_figura(fig, "heatmap_macro_mercati")
plt.close(fig)

print("\nEDA completata.")