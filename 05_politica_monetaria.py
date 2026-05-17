import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.filters.hp_filter import hpfilter

from utils import carica_dati, aggiungi_crisi, resample_dati

os.makedirs("models", exist_ok=True)
os.makedirs("output/grafici", exist_ok=True)


# ---------------------------------------------------------------------------
# Caricamento dati
# ---------------------------------------------------------------------------

print("Caricamento dati...")
dati = carica_dati()
dati = resample_dati(dati)

tasso_bce = dati["tasso_bce"].squeeze().dropna()


# ---------------------------------------------------------------------------
# Filtro Hodrick-Prescott — stima PIL potenziale
#
# lamb=14400 è il parametro convenzionale per dati MENSILI.
# lamb=1600  è per dati trimestrali — valore sbagliato dopo il resample a MS.
# Con lamb troppo basso il trend segue troppo da vicino i dati grezzi,
# l'output gap risulta compresso e la Taylor Rule viene distorta.
# ---------------------------------------------------------------------------

pil = dati["pil_eurozona"].squeeze().dropna()
pil_ciclo, pil_trend = hpfilter(pil, lamb=14400)
output_gap = (pil_ciclo / pil_trend) * 100

print(f"Output gap — media: {output_gap.mean():.2f}%  |  min: {output_gap.min():.2f}%  |  max: {output_gap.max():.2f}%")


# ---------------------------------------------------------------------------
# Taylor Rule
#
# tasso_ottimale = inflazione + 2 + 0.5*(inflazione - 2) + 0.5*output_gap
#
# Inflazione calcolata come variazione YoY (pct_change 12 mesi) sull'indice
# HICP in livelli — produce la variazione percentuale annua, che è la metrica
# che la BCE guarda effettivamente per il suo target del 2%.
# ---------------------------------------------------------------------------

inflazione = dati["inflazione_eurozona"].squeeze().dropna().pct_change(12) * 100
tasso_ottimale = inflazione + 2 + 0.5 * (inflazione - 2) + 0.5 * output_gap

# Allinea le serie sull'indice comune prima di calcolare la deviazione
# (inflazione YoY parte da 2010, tasso BCE da 2009 — senza allineamento
# la sottrazione produce NaN silenziosamente)
indice_comune = tasso_bce.index.intersection(tasso_ottimale.index)
tasso_bce_allineato  = tasso_bce.reindex(indice_comune)
tasso_ottimale_allineato = tasso_ottimale.reindex(indice_comune)
deviazione = tasso_bce_allineato - tasso_ottimale_allineato

print(f"\nDeviazione BCE dalla Taylor Rule:")
print(f"  Media:    {deviazione.mean():.2f} pp")
print(f"  Massima:  {deviazione.max():.2f} pp")
print(f"  Minima:   {deviazione.min():.2f} pp")


# ---------------------------------------------------------------------------
# Grafico 1 — Tasso BCE reale vs Taylor Rule
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(tasso_bce_allineato.index,   tasso_bce_allineato.values,   color="#2980B9", label="Tasso BCE reale")
ax.plot(tasso_ottimale_allineato.index, tasso_ottimale_allineato.values, color="#27AE60", label="Taylor Rule")
ax.set_xlabel("Data")
ax.set_title("Tasso BCE: reale vs Taylor Rule")
aggiungi_crisi(ax)
ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1))
ax.tick_params(axis="x", rotation=45)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig("output/grafici/tasso_bce_vs_taylor_rule.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("\nSalvato: output/grafici/tasso_bce_vs_taylor_rule.png")


# ---------------------------------------------------------------------------
# Grafico 2 — Deviazione dalla Taylor Rule
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(deviazione.index, deviazione.values, color="red", label="Deviazione")
ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.fill_between(deviazione.index, deviazione.values, 0,
                where=(deviazione < 0), alpha=0.15, color="red",    label="Tassi sotto Taylor Rule")
ax.fill_between(deviazione.index, deviazione.values, 0,
                where=(deviazione > 0), alpha=0.15, color="#2980B9", label="Tassi sopra Taylor Rule")
ax.set_xlabel("Data")
ax.set_ylabel("Punti percentuali")
ax.set_title("Deviazione tasso BCE dalla Taylor Rule")
aggiungi_crisi(ax)
ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1))
ax.tick_params(axis="x", rotation=45)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig("output/grafici/deviazione_taylor_rule.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Salvato: output/grafici/deviazione_taylor_rule.png")


# ---------------------------------------------------------------------------
# Salvataggio dati per la dashboard
# ---------------------------------------------------------------------------

taylor_df = pd.DataFrame({
    "tasso_bce":      tasso_bce_allineato,
    "tasso_ottimale": tasso_ottimale_allineato,
    "deviazione":     deviazione,
})
taylor_df.to_csv("models/taylor_rule.csv")
print("Salvato: models/taylor_rule.csv")