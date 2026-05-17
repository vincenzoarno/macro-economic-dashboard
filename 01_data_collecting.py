import os
import pandas as pd
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv

from utils import resample_dati, applica_trasformazioni, pulisci_dizionario

# ---------------------------------------------------------------------------
# Configurazione API
# ---------------------------------------------------------------------------

load_dotenv()
API_KEY = os.getenv("FRED_API_KEY")

if not API_KEY:
    raise EnvironmentError(
        "FRED_API_KEY non trovata. "
        "Crea un file .env con FRED_API_KEY=la_tua_chiave."
    )

fred = Fred(api_key=API_KEY)

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

DATA_INIZIO = "2009-01-01"
DATA_FINE   = pd.Timestamp.today().strftime("%Y-%m-%d")   # sempre aggiornato

# Tutti i nomi usano underscore — coerente con utils.TRASFORMAZIONI e carica_dati()
INDICATORI = {
    # Inflazione
    "inflazione_eurozona":     "CP0000EZ19M086NEST",
    "inflazione_italia":       "CP0000ITM086NEST",
    # Crescita economica
    "pil_eurozona":            "CLVMEURSCAB1GQEA19",
    "pil_italia":              "CLVMNACSCAB1GQIT",
    # Disoccupazione
    # Nota: disoccupazione_eurozona (LRHUTTTTEZM156S) è discontinuata da FRED
    # a gennaio 2023 (Croatia join EA20) — arriva al 2022. Utile per EDA e
    # Granger su dati storici, ma ESCLUSA dalla feature matrix dei modelli ML
    # perché tronca il dataset al 2022 via dropna().
    "disoccupazione_eurozona": "LRHUTTTTEZM156S",
    "disoccupazione_italia":   "LRHUTTTTITM156S",
    # Tassi BCE
    "tasso_bce":               "ECBDFR",
    # Massa monetaria
    "m3_eurozona":             "MABMM301EZM189S",
    # Titoli di stato
    "btp_10y":                 "IRLTLT01ITM156N",
    "bund_10y":                "IRLTLT01DEM156N",
}

MERCATI = {
    "ftse_mib":    "FTSEMIB.MI",
    "eurostoxx50": "^STOXX50E",
    "eurusd":      "EURUSD=X",
    # Brent crude — driver primario dell'inflazione energetica europea
    "brent":       "BZ=F",
    # VIX — indice di volatilità globale, correlato con flight to quality
    # e allargamento spread BTP-Bund nelle fasi di stress
    "vix":         "^VIX",
}

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def scarica_macro() -> dict[str, pd.Series]:
    """
    Scarica tutte le serie macro da FRED.
    Se una serie fallisce, stampa l'errore e continua con le altre
    invece di bloccare tutto il processo.
    """
    risultati = {}
    for nome, codice in INDICATORI.items():
        try:
            serie = fred.get_series(
                codice,
                observation_start=DATA_INIZIO,
                observation_end=DATA_FINE,
            )
            risultati[nome] = serie
            print(f"  [OK] {nome} ({len(serie)} osservazioni)")
        except Exception as e:
            print(f"  [ERRORE] {nome} ({codice}): {e}")

    return risultati


def scarica_mercati() -> dict[str, pd.Series]:
    """
    Scarica i dati di mercato da yFinance (colonna 'Close').
    Stesso pattern di gestione errori di scarica_macro.
    """
    risultati = {}
    for nome, ticker in MERCATI.items():
        try:
            df = yf.download(
                ticker,
                start=DATA_INIZIO,
                end=DATA_FINE,
                progress=False,
            )
            if df.empty:
                print(f"  [ATTENZIONE] {nome} ({ticker}): nessun dato scaricato.")
                continue
            risultati[nome] = df["Close"]
            print(f"  [OK] {nome} ({len(df)} righe)")
        except Exception as e:
            print(f"  [ERRORE] {nome} ({ticker}): {e}")

    return risultati


# ---------------------------------------------------------------------------
# Salvataggio
# ---------------------------------------------------------------------------

def salva_raw(dati: dict, cartella: str) -> None:
    """Salva ogni serie come CSV nella cartella raw specificata."""
    os.makedirs(cartella, exist_ok=True)
    for nome, serie in dati.items():
        percorso = os.path.join(cartella, f"{nome}.csv")
        serie.to_csv(percorso)
        print(f"  Salvato raw: {percorso}")


def salva_clean(dati: dict, cartella: str, usa_trasformazioni: bool = True) -> None:
    """
    Pulisce, ricampiona e salva i dati clean.

    Per i dati macro (usa_trasformazioni=True) applica anche le trasformazioni
    per stazionarietà definite in utils.TRASFORMAZIONI.
    Per i dati di mercato (usa_trasformazioni=False) salva i livelli puliti —
    le trasformazioni vengono applicate al momento del bisogno negli script
    di analisi, non in fase di salvataggio.
    """
    os.makedirs(cartella, exist_ok=True)

    dati_df = {
        nome: serie.to_frame(name=nome) if isinstance(serie, pd.Series) else serie
        for nome, serie in dati.items()
    }

    # Ordine obbligatorio: pulisci prima di resample e trasformazioni
    dati_puliti    = pulisci_dizionario(dati_df)
    dati_resampled = resample_dati(dati_puliti)

    if usa_trasformazioni:
        dati_finali = applica_trasformazioni(dati_resampled)
    else:
        # Mercati: salva i livelli — nessuna trasformazione
        dati_finali = dati_resampled

    for nome, df in dati_finali.items():
        percorso = os.path.join(cartella, f"{nome}.csv")
        df.to_csv(percorso)
        print(f"  Salvato clean: {percorso}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 50)
    print("DOWNLOAD DATI MACRO (FRED)")
    print("=" * 50)
    macro = scarica_macro()

    print("\n" + "=" * 50)
    print("DOWNLOAD DATI MERCATI (yFinance)")
    print("=" * 50)
    mercati = scarica_mercati()

    print("\n" + "=" * 50)
    print("SALVATAGGIO RAW")
    print("=" * 50)
    salva_raw(macro,   "data/raw/macro")
    salva_raw(mercati, "data/raw/mercati")

    print("\n" + "=" * 50)
    print("SALVATAGGIO CLEAN")
    print("=" * 50)
    # Salva LIVELLI per entrambi — le trasformazioni per stazionarietà
    # vengono applicate in memoria da ogni script che le usa (03, 04, 05).
    # Salvare dati trasformati nei clean renderebbe la dashboard e l'EDA errati
    # (mostrerebbero diff invece di livelli assoluti).
    salva_clean(macro,   "data/clean/macro",    usa_trasformazioni=False)
    salva_clean(mercati, "data/clean/mercati",  usa_trasformazioni=False)

    print("\nFatto.")