import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import carica_dati, carica_mercati
dati= carica_dati()
dati2= carica_mercati()


dati["tasso_bce"]= dati["tasso_bce"].resample("MS").last()
dati["pil_italia"]= dati["pil_italia"].resample("MS").ffill()
dati["pil eurozona"]= dati["pil eurozona"].resample("MS").ffill()
tasso_bce= dati["tasso_bce"].squeeze().dropna()
dati_resampled= {}

for nome,value in dati2.items():
    serie=value.resample("MS").last()
    if isinstance(serie, pd.DataFrame):
        serie= serie.iloc[:,0]
    dati_resampled[nome]= serie

# calcoliamo il tasso teorico seguendo la taylor rule e confrontiamola con il tasso BCE attuale
#tasso_ottimale= inflazione + 2% + 0.5 * (inflazione - 2%) + 0.5 * output_gap (quanto il PIL attuale si distacca da quello potenziale)

#il pil potenziale si stima con il filtro Hodrick-Prescott (HP filter)
from statsmodels.tsa.filters.hp_filter import hpfilter
pil= dati["pil eurozona"].squeeze().dropna()
pil_ciclo, pil_trend= hpfilter(pil, lamb=1600)
#pil ciclo= output gap
#lamb 1600 è il parametro standard per i dati trimestrali
print(pil_ciclo.tail())
#ora dobbiamo esprimerlo in percentuale
output_gap= (pil_ciclo / pil_trend) * 100
#utilizza la variazione di inflazione anno su anno
inflazione= dati["inflazione eurozona"].squeeze().dropna().pct_change(12) * 100
tasso_ottimale= inflazione + 2 + 0.5 * (inflazione - 2) + 0.5 * output_gap
print(tasso_ottimale.tail())
print(dati["tasso_bce"].squeeze().tail(10))
def aggiungi_crisi(ax):
    ax.axvspan("2009-01-01", "2009-12-31", alpha=0.2, color="orange", label="Grande recessione")
    ax.axvspan("2011-01-01", "2013-12-31", alpha=0.2, color="red",label="Crisi debito sovrano")
    ax.axvspan("2020-01-01", "2021-12-31", alpha=0.2, color="purple", label="Pandemia")
    ax.axvspan("2022-02-01", "2023-12-31", alpha=0.2, color="blue",label="Crisi energetica")
    ax.axvline(pd.Timestamp("2012-07-26"),color="black", linestyle="--", linewidth=1)
    ax.annotate("Whatever it takes", xy=(pd.Timestamp("2012-07-26"), 11),
            xytext=(pd.Timestamp("2013-06-01"), 11.5),
            arrowprops={"arrowstyle": "->", "color": "black"},
            fontsize=8)
def plot_timeseries(ax, x, y, x2, y2 , color2 , color, xlabel,title):
    ax.plot(x, y, color=color, label="Tasso BCE reale")
    ax.plot(x2, y2, color=color2, label="Taylor Rule")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    aggiungi_crisi(ax)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25,1))
    plt.xticks(rotation=45)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    
fig,ax= plt.subplots(figsize=(12,5))
plot_timeseries(ax, tasso_bce.index, tasso_bce.values, tasso_ottimale.index, tasso_ottimale.values,
                color="#2980B9",color2="#27AE60" , xlabel="date",
                 title="Tasso BCE: Reale vs tasso Taylor Rule")

fig.savefig(f"output/grafici/Tasso_BCE_Reale_vs_tasso_Taylor_Rule.png", dpi=300, bbox_inches="tight")

deviazione= tasso_bce - tasso_ottimale
fig,ax= plt.subplots(figsize=(12,5))
ax.plot(deviazione.index, deviazione.values, color="red", label= "deviazione") 
ax.set_xlabel("date")
ax.set_ylabel("Punti percentuali")
ax.set_title("Differenza tra taso BCE reale e Taylor rule")
aggiungi_crisi(ax)
ax.legend(loc='upper right', bbox_to_anchor=(1.25,1))
plt.xticks(rotation=45)
ax.grid(alpha=0.3)
ax.axhline(0, color="Black", linestyle="--")
plt.tight_layout()
fig.savefig(f"output/grafici/deviazione.png", dpi=300, bbox_inches="tight")

taylor_df= pd.DataFrame({"tasso_bce" : tasso_bce,
                         "tasso_ottimale": tasso_ottimale,
                         "deviazione" : deviazione})

taylor_df.to_csv("models/taylor_rule.csv")