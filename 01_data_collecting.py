import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from fredapi import Fred
import os 
from dotenv import load_dotenv
load_dotenv()
API_KEY= os.getenv("FRED_API_KEY")
fred = Fred(api_key=API_KEY)
INDICATORI= {
    # INFLAZIONE
    "inflazione eurozona": "CP0000EZ19M086NEST",
    "inflazione italia" : "CP0000ITM086NEST",
     # CRESCITA ECONOMICA
    "pil eurozona" : "CLVMEURSCAB1GQEA19",
    "pil_italia": "CLVMNACSCAB1GQIT",
    # DISOCCUPAZIONE
    "disoccupazione_eurozona": "LRHUTTTTEZM156S",
    "disoccupazione_italia":   "LRHUTTTTITM156S",
    # TASSI BCE
    "tasso_bce":              "ECBDFR",

    # MASSA MONETARIA
    "m3_eurozona":            "MABMM301EZM189S",
    # TITOLI DI STATO
    "btp_10y":"IRLTLT01ITM156N",
    "bund_10y": "IRLTLT01DEM156N"
}


def scarica_tutto():
    risultati = {}
    for key, value in INDICATORI.items():
        print(f"Tentativo: '{key}' -> '{value}'")
        risultati[key] = fred.get_series(value, observation_start="2009-01-01", observation_end="2026-04-01")
        print(f"Successo: {key}")
    return risultati

dati= scarica_tutto()
for nome,series in dati.items():
    print(f"\n{nome}:")
    print(series.tail(3))

import os
os.makedirs("data/raw", exist_ok=True)
for nome,serie in dati.items():
    serie.to_csv(f"data/raw/{nome}.csv")
    print(f"Salvato: {nome}")

MERCATI={
    "ftse_mib": "FTSEMIB.MI",
    "eurostoxx50": "^STOXX50E",
    "eurusd": "EURUSD=X",
    
}
def scarica_mercati():
    risultati= {}
    for nome,ticker in MERCATI.items():
        print(f"Scaricando {nome}...")
        df= yf.download(ticker, start="2009-01-01", end="2026-04-01", progress=False)
        risultati[nome]= df["Close"]
        print(f"OK - {len(df)} righe")
    return risultati

dati_mercati= scarica_mercati()
for nome, serie in dati_mercati.items():
    serie.to_csv(f"data/raw/{nome}.csv")
    print(f"salvato:{nome}")
