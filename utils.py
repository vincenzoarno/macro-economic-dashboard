import os
import pandas as pd
import matplotlib.pyplot as plt
def carica_dati():
    dati= {}
    for file in os.listdir("data/clean/macro"):
        dati[file.split(".")[0]]= pd.read_csv(
            f"data/clean/macro/{file}",
            index_col= 0,
            parse_dates=True
        )
    return dati

def carica_mercati():
    dati2= {}
    for file in os.listdir("data/clean/mercati"):
        dati2[file.split(".")[0]]= pd.read_csv(
            f"data/clean/mercati/{file}",
            index_col= 0,
            parse_dates=True
        )
    return dati2


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