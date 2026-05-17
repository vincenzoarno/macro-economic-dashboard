import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import permutations
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

from utils import carica_dati, carica_mercati, applica_trasformazioni, resample_dati

# ---------------------------------------------------------------------------
# Caricamento dati
# Le trasformazioni per stazionarietà le applichiamo qui — ADF e Granger
# richiedono serie stazionarie. I livelli originali restano in EDA.
# Includiamo anche brent e vix (mercati) perché sono feature rilevanti:
# brent anticipa l'inflazione energetica, vix anticipa lo spread (risk-off).
# disoccupazione_eurozona è inclusa — arriva al 2022 ma il suo storico
# ha valore per l'analisi causale sul periodo 2009-2022.
# ---------------------------------------------------------------------------

print("Caricamento dati...")
dati    = carica_dati()
dati    = resample_dati(dati)
mercati = carica_mercati()

dati_staz = applica_trasformazioni(dati)

# Aggiungi brent e vix stazionarizzati (pct_change mensile)
# Le serie di mercato giornaliere sono già ricampionate a mensile da carica_mercati
# tramite la pipeline clean di 01_data_collecting.py
import numpy as np
for nome in ("brent", "vix"):
    if nome in mercati:
        serie = mercati[nome].squeeze().resample("MS").last()
        serie_staz = serie.pct_change(1).replace([np.inf, -np.inf], np.nan).dropna()
        dati_staz[nome] = serie_staz.to_frame()


# ---------------------------------------------------------------------------
# Test ADF — verifica stazionarietà dopo le trasformazioni
# ---------------------------------------------------------------------------

print("\n" + "=" * 50)
print("TEST ADF (p-value — soglia: 0.05)")
print("=" * 50)
print(f"{'Serie':<30} {'p-value':>10}  {'Stazionaria?':>14}")
print("-" * 58)

for nome, df in dati_staz.items():
    serie = df.squeeze().dropna()
    p_value = adfuller(serie)[1]
    esito = "SI" if p_value < 0.05 else "NO  <-- attenzione"
    print(f"{nome:<30} {p_value:>10.4f}  {esito:>14}")


# ---------------------------------------------------------------------------
# Test di causalità di Granger
#
# Per ogni coppia ordinata (X, Y) testiamo se i valori passati di X
# migliorano la previsione di Y rispetto a usare solo i valori passati di Y.
# Usiamo il p-value minimo tra i lag 1-6: se almeno un lag è significativo,
# c'è evidenza di causalità a quel lag.
#
# Convenzione della matrice risultante:
#   riga    = X  (la variabile che potenzialmente CAUSA)
#   colonna = Y  (la variabile che viene potenzialmente CAUSATA)
# Leggi: "la riga X causa la colonna Y?"
# ---------------------------------------------------------------------------

print("\n" + "=" * 50)
print("TEST DI GRANGER (56 coppie, lag 1-6)")
print("=" * 50)

coppie = list(permutations(dati_staz.keys(), 2))
risultati_granger = {}

for x, y in coppie:
    df_test = pd.concat(
        [dati_staz[x].squeeze(), dati_staz[y].squeeze()],
        axis=1,
    ).dropna()

    # grangercausalitytests vuole che la variabile da prevedere (Y)
    # sia nella prima colonna e la causa (X) nella seconda
    df_test = df_test.iloc[:, [1, 0]]

    risultato = grangercausalitytests(df_test, maxlag=6, verbose=False)
    p_values = [res[0]["ssr_ftest"][1] for res in risultato.values()]
    risultati_granger[x, y] = min(p_values)

# Coppie significative (p < 0.05)
print("\nRelazioni causali significative (p < 0.05):")
print(f"  {'X (causa)':<30} →  {'Y (effetto)':<30}  p-value")
print("  " + "-" * 72)
for (x, y), p in sorted(risultati_granger.items(), key=lambda kv: kv[1]):
    if p < 0.05:
        print(f"  {x:<30} →  {y:<30}  {p:.4f}")


# ---------------------------------------------------------------------------
# Heatmap causalità di Granger
# ---------------------------------------------------------------------------

granger_matrix = pd.Series(risultati_granger).unstack()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    granger_matrix,
    cmap="Reds_r",
    annot=True,
    fmt=".2f",
    cbar=True,
    ax=ax,
    vmin=0,
    vmax=0.1,   # scala fissa 0-0.1: evidenzia meglio le zone significative
)
ax.set_xlabel("Y — variabile CAUSATA", fontsize=10)
ax.set_ylabel("X — variabile che CAUSA", fontsize=10)
ax.set_title("Causalità di Granger (p-value minimo, lag 1-6)\nValori bassi = evidenza di causalità", fontsize=11)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
plt.tight_layout()
fig.savefig("output/grafici/heatmap_granger.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Salvataggio matrice per la dashboard
# ---------------------------------------------------------------------------

import os
os.makedirs("models", exist_ok=True)
granger_matrix.to_csv("models/granger_matrix.csv")
print("\nSalvato: models/granger_matrix.csv")
print("Salvato: output/grafici/heatmap_granger.png")