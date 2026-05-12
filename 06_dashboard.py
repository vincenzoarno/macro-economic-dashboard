import matplotlib.pyplot as plt
import pandas as pd
from utils import carica_dati, aggiungi_crisi, plot_timeseries
import streamlit as st
import seaborn as sns
#avviamo la prima streamlit.
st.title("Macro Economic Dashboard")
st.write("Testo qualsiasi")
st.sidebar.title("Navigazione")
sezione= st.sidebar.radio("Vai a:", 
                          ["Panoramica Macro", "Previsioni", "Analisi Relazioni", "Politica Monetaria"])
#carichiamo i dati macro e i grafici.

@st.cache_data
def carica_dati_dashboard():
    return carica_dati()

dati= carica_dati_dashboard()
#SEZIONE PREVISIONI IMPORTIAMO I MODELLI SALVATI
@st.cache_resource
def carica_modelli():
    import joblib
    gb= joblib.load("models/gb_inflazione.pkl")
    gb2=joblib.load("models/gb_spread.pkl")
    scaler= joblib.load("models/scaler.pkl")
    return gb, gb2, scaler
#SEZIONE PANORAMICA MACRO
gb, gb2, scaler= carica_modelli()
X=pd.read_csv("models/feature_matrix.csv", index_col=0, parse_dates=True)
if sezione == "Panoramica Macro":
    st.header("Panoramica Macroeconomica")
    st.write("Qui andrà la panoramica")
    grafici= {"Inflazione Eurozona": dati["inflazione eurozona"].squeeze(),
          "Inflazione Italia": dati["inflazione italia"].squeeze(),
          "Disoccupazione Eurozona": dati["disoccupazione_eurozona"].squeeze(),
          "Disoccupazione Italia": dati["disoccupazione_italia"].squeeze(),
          "PIL Eurozona": dati["pil eurozona"].squeeze(),
          "PIL Italia": dati["pil_italia"].squeeze(),
          "M3 Eurozona": dati["m3_eurozona"].squeeze(),
          "Tasso BCE": dati["tasso_bce"].squeeze(),
          "BTP 10Y": dati["btp_10y"].squeeze(),
          "Bund 10Y": dati["bund_10y"].squeeze(),
          "Spread BTP-Bund": (dati["btp_10y"].squeeze() - dati["bund_10y"].squeeze()).dropna()}
    nomi= list(grafici.keys())
    for i in range(0, len(nomi), 2):
        col1,col2= st.columns(2)
        with col1:
          fig,ax= plt.subplots(figsize=(10,5))
          ax.plot(grafici[nomi[i]].index, grafici[nomi[i]].values)
          ax.set_title(nomi[i])
          aggiungi_crisi(ax)
          st.pyplot(fig)
          plt.close()
          if i + 1 < len(nomi):
             with col2:
                 fig,ax= plt.subplots(figsize=(10,5))
                 ax.plot(grafici[nomi[i+1]].index, grafici[nomi[i+1]].values)
                 ax.set_title(nomi[i+1])
                 aggiungi_crisi(ax)
                 st.pyplot(fig)
                 plt.close()







elif sezione == "Previsioni":
    st.header("Modelli Predittivi")
    st.write("Qui andranno i modelli predittivi")
    st.subheader("Previsione Inflazione Eurozona (3 mesi)")
    st.write(f"MAE del modello: 0.57 punti percentuali")
    st.subheader("Previsione Spread BTP-Bund (1 mese)")
    st.write(f"MAE del modello: 0.56 punti percentuali")
    modalita= st.radio("Modalità previsione:", ["Automatica", "Manuale"])
    if modalita == "Automatica":
       st.write("Utilizzo gli ultimi dati disponibili")
       X.index.name= "date"
       ultima_riga= X.iloc[[-1]]
       ultima_riga_scaled= scaler.transform(ultima_riga)
       previsione_inflazione= gb.predict(ultima_riga_scaled)[0]
       previsione_spread= gb2.predict(ultima_riga_scaled)[0]
       st.metric("Inflazione tra 3 mesi", f"{previsione_inflazione: .2f}%")
       st.metric("Previsione BTP-Bund tra 1 mese", f"{previsione_spread:.2f}%")
    elif modalita == "Manuale":
        st.subheader("Modifica le feature principali")
        inf_it_lag3= st.slider("Inflazione Italia lag 3 mesi", -2.0, 5.0, 0.0)
        inf_it_lag1= st.slider("Inflazione Italia lag 1 mese", -2.0, 5.0, 0.0)
        inf_eu_lag3= st.slider("Inflazione Eurozona lag 3 mesi", -2.0, 5.0, 0.0)
        m3_lag3= st.slider("M3 Eurozona lag 3 mesi", -50.0, 50.0, 0.0)
        m3_lag1= st.slider("M3 Eurozona lag 1 mese", -50.0, 50.0, 0.0)
        X_manuale= X.iloc[[-1]].reset_index(drop=True).copy()
        X_manuale["inflazione italia_lag3"]= inf_it_lag3
        X_manuale["inflazione italia_lag1"]= inf_it_lag1  
        X_manuale["inflazione eurozona_lag3"]= inf_eu_lag3
        X_manuale["m3_eurozona_lag3"]= m3_lag3 
        X_manuale["m3_eurozona_lag1"]= m3_lag1
        X_manuale_scaled= scaler.transform(X_manuale)
        prev_inf= gb.predict(X_manuale_scaled)[0]
        prev_spread= gb2.predict(X_manuale_scaled)[0]
        st.metric("Inflazione tra 3 mesi", f"{prev_inf: .2f}%")
        st.metric("Spread BTP-Bund tra 1 mese", f"{prev_spread: .2f}%")    
            

elif sezione == "Analisi Relazioni":
    st.header("Analisi delle Relazioni")
    st.write("Qui andrà la heatmap di Granger")
    granger_matrix= pd.read_csv("models/granger_matrix.csv", index_col=0)
    granger_matrix= granger_matrix.astype(float)
    fig,ax= plt.subplots(figsize=(10,8))
    sns.heatmap(granger_matrix, cmap= "Reds_r", annot=True,fmt=".2f", cbar=True, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    

elif sezione == "Politica Monetaria":
    st.header("Politica Monetaria BCE")
    st.write("Qui andrà la Taylor rule")
    taylor_df= pd.read_csv("models/taylor_rule.csv", index_col=0, parse_dates=True)
    fig,ax= plt.subplots(figsize=(12,5))
    plot_timeseries(ax, taylor_df.index, taylor_df["tasso_bce"].values, taylor_df.index, taylor_df["tasso_ottimale"].values,
                color="#2980B9",color2="#27AE60" , xlabel="date",
                 title="Tasso BCE: Reale vs tasso Taylor Rule")
    st.pyplot(fig)
    plt.close()
    fig,ax= plt.subplots(figsize=(12,5))
    ax.plot(taylor_df.index, taylor_df["deviazione"].values, color="red", label= "deviazione") 
    ax.set_xlabel("date")
    ax.set_ylabel("Punti percentuali")
    ax.set_title("Differenza tra taso BCE reale e Taylor rule")
    aggiungi_crisi(ax)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25,1))
    plt.xticks(rotation=45)
    ax.grid(alpha=0.3)
    ax.axhline(0, color="Black", linestyle="--")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

