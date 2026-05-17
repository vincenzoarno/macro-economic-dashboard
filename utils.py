import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

CRISI = [
    {"inizio": "2009-01-01", "fine": "2009-12-31", "colore": "orange", "label": "Grande recessione"},
    {"inizio": "2011-01-01", "fine": "2013-12-31", "colore": "red",    "label": "Crisi debito sovrano"},
    {"inizio": "2020-01-01", "fine": "2021-12-31", "colore": "purple", "label": "Pandemia"},
    {"inizio": "2022-02-01", "fine": "2023-12-31", "colore": "blue",   "label": "Crisi energetica"},
]

WHATEVER_IT_TAKES = pd.Timestamp("2012-07-26")

# Dizionario centralizzato: nome_chiave → trasformazione da applicare per stazionarietà.
# Formato: (tipo, parametro)  dove tipo ∈ {"pct", "diff"}
# Tutti i nomi usano underscore — nessuno spazio.
TRASFORMAZIONI = {
    "pil_eurozona":            ("pct",  12),
    "pil_italia":              ("pct",  12),
    "tasso_bce":               ("diff",  1),
    "disoccupazione_eurozona": ("pct",   1),  # arriva al 2022 — usare solo in EDA/Granger
    "disoccupazione_italia":   ("pct",   1),
    "m3_eurozona":             ("diff",  1),
    "inflazione_eurozona":     ("diff",  1),
    "inflazione_italia":       ("diff",  1),
}


# ---------------------------------------------------------------------------
# Caricamento dati
# ---------------------------------------------------------------------------

def _carica_cartella(percorso: str) -> dict[str, pd.DataFrame]:
    """
    Funzione interna generica: legge tutti i CSV in una cartella
    e restituisce un dizionario  nome_file → DataFrame.

    Il nome della chiave viene normalizzato (spazi → underscore, minuscolo)
    per garantire coerenza con TRASFORMAZIONI e il resto del codice.
    """
    if not os.path.isdir(percorso):
        raise FileNotFoundError(
            f"Cartella non trovata: '{percorso}'. "
            "Hai eseguito 01_data_collecting.py?"
        )

    dati = {}
    for file in os.listdir(percorso):
        if not file.endswith(".csv"):
            continue

        chiave = file.replace(".csv", "").strip().lower().replace(" ", "_")
        df = pd.read_csv(
            os.path.join(percorso, file),
            index_col=0,
            parse_dates=True,
        )

        if df.empty:
            print(f"[ATTENZIONE] '{file}' è vuoto — saltato.")
            continue

        if not isinstance(df.index, pd.DatetimeIndex):
            print(f"[ATTENZIONE] '{file}' non ha un indice datetime — saltato.")
            continue

        dati[chiave] = df

    return dati


def carica_dati() -> dict[str, pd.DataFrame]:
    """Carica i dati macro da data/clean/macro/."""
    return _carica_cartella("data/clean/macro")


def carica_mercati() -> dict[str, pd.DataFrame]:
    """Carica i dati di mercato da data/clean/mercati/."""
    return _carica_cartella("data/clean/mercati")


# ---------------------------------------------------------------------------
# Pulizia
# ---------------------------------------------------------------------------

def pulisci_serie(df: pd.DataFrame, soglia_nan: float = 0.05) -> pd.DataFrame:
    """
    Pulizia standard applicata a ogni serie prima del salvataggio clean.

    Operazioni nell'ordine corretto:
    1. Rimozione timezone dall'indice (yFinance aggiunge UTC, FRED no — va normalizzato)
    2. Deduplicazione dell'indice (tiene l'ultima occorrenza)
    3. Gestione NaN: dropna se sotto soglia, fillna(median) se sopra
    4. Arrotondamento a 2 decimali

    Il parametro soglia_nan controlla la tolleranza: 0.05 = se meno del 5%
    dei valori è NaN, li rimuoviamo; altrimenti imputiamo con la mediana.
    La mediana è più robusta della media per serie finanziarie con outlier.

    Ritorna un nuovo DataFrame — non modifica quello in ingresso.
    """
    df = df.copy()

    # 1. Rimuovi timezone — indici tz-aware e tz-naive non si concatenano
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    # 2. Deduplica indice
    df = df[~df.index.duplicated(keep="last")]

    # 3. Gestione NaN colonna per colonna
    for col in df.columns:
        n_nan = df[col].isna().sum()
        frazione_nan = n_nan / len(df)

        if frazione_nan == 0:
            continue
        elif frazione_nan < soglia_nan:
            df = df.dropna(subset=[col])
        else:
            mediana = df[col].median()
            df[col] = df[col].fillna(mediana)
            print(f"  [INFO] '{col}': {n_nan} NaN imputati con mediana ({mediana:.2f})")

    # 4. Arrotondamento
    df = df.round(2)

    return df


def pulisci_dizionario(
    dati: dict[str, pd.DataFrame],
    soglia_nan: float = 0.05,
) -> dict[str, pd.DataFrame]:
    """
    Applica pulisci_serie a ogni DataFrame nel dizionario.
    Ritorna un nuovo dizionario.
    """
    return {nome: pulisci_serie(df, soglia_nan) for nome, df in dati.items()}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def resample_dati(dati: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Ricampiona tasso BCE (giornaliero → mensile) e PIL (trimestrale → mensile).

    Ritorna un NUOVO dizionario — non modifica quello passato in ingresso.
    In questo modo il chiamante decide se vuole sovrascrivere o conservare
    l'originale.
    """
    risultato = dict(dati)  # copia shallow: i DataFrame non vengono duplicati in memoria

    if "tasso_bce" in risultato:
        risultato["tasso_bce"] = risultato["tasso_bce"].resample("MS").last()

    for serie in ("pil_italia", "pil_eurozona"):
        if serie in risultato:
            risultato[serie] = risultato[serie].resample("MS").ffill()

    return risultato


def applica_trasformazioni(
    dati: dict[str, pd.DataFrame],
    trasformazioni: dict | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Applica le trasformazioni per stazionarietà definite in TRASFORMAZIONI
    (o in un dizionario personalizzato passato come argomento).

    Ritorna un NUOVO dizionario — non modifica quello passato in ingresso.
    Le serie non presenti in 'trasformazioni' vengono copiate invariate.
    """
    if trasformazioni is None:
        trasformazioni = TRASFORMAZIONI

    # Controlla se ci sono nomi nel dizionario trasformazioni che non esistono
    # nel dizionario dati — errore silenzioso molto comune.
    chiavi_mancanti = set(trasformazioni) - set(dati)
    if chiavi_mancanti:
        print(
            f"[ATTENZIONE] Le seguenti serie sono in TRASFORMAZIONI ma non nei dati: "
            f"{sorted(chiavi_mancanti)}"
        )

    risultato = {}
    for nome, df in dati.items():
        if nome not in trasformazioni:
            risultato[nome] = df
            continue

        tipo, parametro = trasformazioni[nome]
        serie = df.squeeze()  # da DataFrame a Series se ha una sola colonna

        if tipo == "pct":
            trasformata = serie.pct_change(parametro)
        elif tipo == "diff":
            trasformata = serie.diff(parametro)
        else:
            raise ValueError(f"Tipo di trasformazione sconosciuto: '{tipo}'. Usa 'pct' o 'diff'.")

        risultato[nome] = (
            trasformata
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_frame()  # riconverte a DataFrame per coerenza con il resto del dizionario
        )

    return risultato


def crea_lag(
    dizionario: dict[str, pd.Series | pd.DataFrame],
    lags: list[int] = None,
) -> dict[str, pd.Series]:
    """
    Crea feature laggiate per ogni serie nel dizionario.

    Parametri
    ---------
    dizionario : dict  nome → Series o DataFrame a colonna singola
    lags       : lista di interi (default [1, 3, 6])

    Ritorna un dizionario piatto  nome_lagN → Series.
    """
    if lags is None:
        lags = [1, 3, 6]

    risultato = {}
    for serie, value in dizionario.items():
        s = value.squeeze() if isinstance(value, pd.DataFrame) else value
        for lag in lags:
            risultato[f"{serie}_lag{lag}"] = s.shift(lag)

    return risultato


# ---------------------------------------------------------------------------
# Visualizzazione
# ---------------------------------------------------------------------------

def aggiungi_crisi(ax: plt.Axes) -> None:
    """
    Aggiunge alle aree shaded dei periodi di crisi e la linea
    'Whatever it takes' all'asse passato.
    """
    for crisi in CRISI:
        ax.axvspan(
            crisi["inizio"],
            crisi["fine"],
            alpha=0.15,
            color=crisi["colore"],
            label=crisi["label"],
        )

    ax.axvline(WHATEVER_IT_TAKES, color="black", linestyle="--", linewidth=1)
    ax.annotate(
        "Whatever it takes",
        xy=(WHATEVER_IT_TAKES, ax.get_ylim()[1] * 0.85),
        xytext=(pd.Timestamp("2013-06-01"), ax.get_ylim()[1] * 0.9),
        arrowprops={"arrowstyle": "->", "color": "black"},
        fontsize=8,
    )


def plot_timeseries(
    ax: plt.Axes,
    x,
    y,
    color: str,
    xlabel: str,
    ylabel: str,
    title: str,
    label: str | None = None,
) -> None:
    """
    Traccia una serie temporale sull'asse ax con aree crisi e griglia.

    Nota: usa ax.tick_params invece di plt.xticks per evitare effetti
    collaterali sugli altri assi della stessa figura.
    """
    ax.plot(x, y, color=color, label=label or xlabel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    aggiungi_crisi(ax)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1))
    ax.tick_params(axis="x", rotation=45)  # agisce solo su questo asse, non su plt globale
    ax.grid(alpha=0.3)