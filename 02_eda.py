import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
# Aprire e ispezionare i dati
def pulisci_serie(df):
    for col,value in df.items():
        if value.isna().sum() < len(value) * 0.05:
            df= df.dropna(subset=col)
        else: 
            df[col]=value.fillna(value.median())
        df[col]= df[col].round(2)        
    
    df= df[~df.index.duplicated(keep="last")]
    return df

def importa_serie(percorso):
    nome= percorso.split("/")[-1].split(".")[0]
    df= pd.read_csv(percorso, index_col=0, parse_dates=True)
    df.columns= [nome]
    df.index.name= "date"
    return df

dati= {}
for file in os.listdir("data/raw/macro"):
    dati[file.split(".")[0]]= importa_serie(f"data/raw/macro/{file}")

for key, value in dati.items():
    dati[key]= pulisci_serie(value)

for nome, df in dati.items():
    df.to_csv(f"data/clean/macro/{nome}.csv")
    print(f"Salvato: {nome}")

def pulisci_mercati(df):
    for col,value in df.items():
        if value.isna().sum() < len(value) * 0.05:
            df= df.dropna(subset=col)
        else: 
            df[col]=value.fillna(value.median())
        df[col]= df[col].round(2)        
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df= df[~df.index.duplicated(keep="last")]
    df= df.squeeze()
    return df

dati2= {}

for file in os.listdir("data/raw/mercati"):
    dati2[file.split(".")[0]]= importa_serie(f"data/raw/mercati/{file}")

for key,value in dati2.items():
    dati2[key]= pulisci_mercati(value)

for nome,df in dati2.items():
    df.to_csv(f"data/clean/mercati/{nome}.csv")
    print(f"Salvato: {nome}")

# APPLICHIAMO STATISTICHE DESCRITTIVE
for nome,df in dati.items():
    print(f"\n{nome}:")
    print(df.describe())
    
# VISUALIZZIAMO
plt.style.use("seaborn-v0_8")
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
def plot_timeseries(ax, x, y, color, xlabel, ylabel,title):
    ax.plot(x, y, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, color=color)
    ax.set_title(title)
    ax.tick_params("y", colors=color)
    aggiungi_crisi(ax)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25,1))
    plt.xticks(rotation=45)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"output/grafici/{title}.png", dpi=300, bbox_inches="tight")

disoccupazione_eurozona= dati["disoccupazione_eurozona"]
disoccupazione_italia= dati["disoccupazione_italia"]
inflazione_eurozona=dati["inflazione eurozona"]
inflazione_italia= dati["inflazione italia"]
m3_eurozona= dati["m3_eurozona"]
pil_eurozona= dati["pil eurozona"]
pil_italia= dati["pil_italia"]
tasso_bce= dati["tasso_bce"]

fig,ax= plt.subplots(figsize=(10,5))
plot_timeseries(ax, disoccupazione_eurozona.index, disoccupazione_eurozona.values,
                color="#C0392B", xlabel="date", ylabel="disoccupazione eurozona(%)",
                 title="Percentuale di disoccupazione eurozona")


fig,ax= plt.subplots(figsize=(10,5))
plot_timeseries(ax, disoccupazione_italia.index, disoccupazione_italia.values,
                color="#C0392B", xlabel="date", ylabel="disoccupazione italia(%)",
                 title="Percentuale di disoccupazione italia")

fig,ax= plt.subplots(figsize=(10,5))
plot_timeseries(ax, inflazione_eurozona.index, inflazione_eurozona.values,
                color="#E67E22", xlabel="date", ylabel="inflazione eurozona(CPI)",
                 title="consumer price index europeo")


fig,ax= plt.subplots(figsize=(10,5))
plot_timeseries(ax, inflazione_italia.index, inflazione_italia.values,
                color="#E67E22", xlabel="date", ylabel="inflazione italia(CPI)",
                 title="consumer price index italiano")


fig,ax= plt.subplots(figsize=(10,5))
plot_timeseries(ax, inflazione_eurozona.index, inflazione_eurozona.values,
                color="#E67E22", xlabel="date", ylabel="inflazione eurozona(CPI)",
                 title="consumer price index europeo")


fig,ax= plt.subplots(figsize=(10,5))
plot_timeseries(ax, m3_eurozona.index, m3_eurozona.values,
                color="#27AE60", xlabel="date", ylabel="m3 eurozona",
                 title="massa monetaria m3 europea(Euro (€))")


fig,ax= plt.subplots(figsize=(10,5))
plot_timeseries(ax, pil_italia.index, pil_italia.values,
                color="#2980B9", xlabel="date", ylabel="pil ",
                 title="pil italia(miliardi di Euro (€))")

fig,ax= plt.subplots(figsize=(10,5))
plot_timeseries(ax, pil_eurozona.index, pil_eurozona.values,
                color="#2980B9", xlabel="date", ylabel="pil",
                 title="pil eurozona(miliardi di Euro (€))")


fig,ax= plt.subplots(figsize=(10,5))
plot_timeseries(ax, tasso_bce.index, tasso_bce.values,
                color="#8E44AD", xlabel="date", ylabel="tassi bce",
                 title="tassi di interesse BCE (%)")

pil_italia_mensile= dati["pil_italia"].resample("MS").ffill()
pil_europa_mensile= dati["pil eurozona"].resample("MS").ffill()
tasso_bce_mensile= dati["tasso_bce"].resample("MS").last()


btp_10y= dati2["btp_10y"]
bund_10y= dati2["bund_10y"]
eurostoxx50= dati2["eurostoxx50"]
eurusd= dati2["eurusd"]
ftse_mib= dati2["ftse_mib"]
fig,ax= plt.subplots(figsize=(10,5))
plot_timeseries(ax, btp_10y.index, btp_10y.values,
                color="#E74C3C", xlabel="date", ylabel="btp 10y",
                 title="Rendimento BTP italiano a 10 anni (%)")


fig,ax= plt.subplots(figsize=(10,5))
plot_timeseries(ax, bund_10y.index, bund_10y.values,
                color="#F39C12", xlabel="date", ylabel="bund 10y",
                 title="Rendimento Bund tedesco a 10 anni (%)")


fig,ax= plt.subplots(figsize=(10,5))
plot_timeseries(ax, eurostoxx50.index, eurostoxx50.values,
                color="#2980B9", xlabel="date", ylabel="eurostoxx50",
                 title="Indice azionario eurozona (EuroStoxx 50)")


fig,ax= plt.subplots(figsize=(10,5))

plot_timeseries(ax, eurusd.index, eurusd.values,
                color="#27AE60", xlabel="date", ylabel="eurusd",
                 title="Tasso di cambio EURUSD")

fig,ax= plt.subplots(figsize=(10,5))
plot_timeseries(ax, ftse_mib.index, ftse_mib.values,
                color="#2C3E50", xlabel="date", ylabel="ftsemib",
                 title="Indice azionario italiano (FTSE MIB)")


btp_10y_mon=dati2["btp_10y"].resample("MS").last()
bund_10y_mon=dati2["bund_10y"].resample("MS").last()
eurostoxx50mon=dati2["eurostoxx50"].resample("MS").last()
eurusdmon=dati2["eurusd"].resample("MS").last()
ftse_mib_mon=dati2["ftse_mib"].resample("MS").last()
df_macro_mercati= pd.concat([disoccupazione_eurozona, disoccupazione_italia, inflazione_eurozona, inflazione_italia, m3_eurozona, pil_italia_mensile, pil_europa_mensile, tasso_bce_mensile,
                     btp_10y_mon,bund_10y_mon,eurostoxx50mon,eurusdmon,ftse_mib_mon], axis=1)
matrice_correlazione= df_macro_mercati.corr()
fig,ax= plt.subplots(figsize=(10,8))
heatmap_macros= sns.heatmap(matrice_correlazione,cmap="coolwarm", annot=True, cbar=True, fmt=".2f", ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
plt.tight_layout()
fig.savefig(f"output/grafici/heatmap macros and market.png", dpi=300, bbox_inches="tight")