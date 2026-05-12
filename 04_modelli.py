import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import os
import statsmodels

#costruiamo la feature matrix
#è un DataFrame dove ogni riga è un mese, ogni colonna è una variabile che usi per prevedere, l'ultima colonna è il target cioè quello che vuoi prevedere.
from utils import carica_dati, carica_mercati
dati= carica_dati()
dati2= carica_mercati()
spread= (dati["btp_10y"].squeeze() - dati["bund_10y"].squeeze()).dropna()
dati["tasso_bce"]= dati["tasso_bce"].resample("MS").last()
dati["pil_italia"]= dati["pil_italia"].resample("MS").ffill()
dati["pil eurozona"]= dati["pil eurozona"].resample("MS").ffill()

dati_resampled= {}
#qui c'è un errore: non tutti i df di dati resampled diventano series con lo squeeze
for nome,value in dati2.items():
    serie=value.resample("MS").last()
    if isinstance(serie, pd.DataFrame):
        serie= serie.iloc[:,0]
    dati_resampled[nome]= serie

dati= {k: v.squeeze() for k,v in dati.items()}
#prima di creare i lag rendiamo le serie stazionarie.
trasformazioni= {"pil eurozona": ("pct",12),
                 "pil_italia": ("pct",12),
                 "tasso_bce": ("diff",1),
                 "disoccupazione_eurozona": ("pct",1),
                 "disoccupazione_italia": ("pct",1),
                 "m3_eurozona": ("diff",1),
                 "inflazione eurozona": ("diff",1),
                 "inflazione italia" : ("diff",1)
}

for nome,n in trasformazioni.items():
    if n[0]=="pct":
        dati[nome]= dati[nome].pct_change(n[1]).squeeze().replace([np.inf, -np.inf], np.nan).dropna()
    else:
        dati[nome]= dati[nome].diff(1).squeeze().replace([np.inf, -np.inf], np.nan).dropna()


serie_lag= {}
def crea_lag(dizionario, lags=[1,3,6]):
    risultato={}
    for serie, value in dizionario.items():
        for lag in lags:
            nome_colonna= f"{serie}_lag{lag}"
            risultato[nome_colonna]= value.shift(lag)
    return risultato       


serie_lag.update(crea_lag(dati))
serie_lag.update(crea_lag(dati_resampled))
serie_lag.update(crea_lag({"spread":spread}))
print(list(serie_lag.keys()))
feature_matrix= pd.DataFrame(serie_lag)
feature_matrix.dropna()
# adesso che abbiamo creato la feature matrix dobbiamo aggiungere i target (ciò che vogliamo prevedere)
target_inflazione= dati["inflazione eurozona"].squeeze().shift(-3)
target_spread= spread.shift(-1)
feature_matrix["target_inflazione"]= target_inflazione
feature_matrix["target_spread"]= target_spread
feature_matrix= feature_matrix.dropna()
print(feature_matrix.shape)
x= feature_matrix.drop(columns=["target_inflazione", "target_spread"])
y= feature_matrix["target_inflazione"]
y2= feature_matrix["target_spread"]
#adesso costruiamo il baseline model.
baseline_pred= dati["inflazione eurozona"].squeeze().reindex(feature_matrix.index).dropna()
y_aligned= y.reindex(baseline_pred.index)
#calcoliamo il mean absolute error cioè quanto sbaglia il modello rispetto alla realtà
from sklearn.metrics import mean_absolute_error
mae_baseline= mean_absolute_error(y_aligned, baseline_pred)
print(f"MAE baseline: {mae_baseline: .4f}")
#il baseline sbaglia in media di 0.70 punti percentuale e la std di y è 6.5 quindi il modello baseline è relativamente buono
#dividiamo i dati tra train e test
split= int(len(feature_matrix)* 0.8)
X_train= x.iloc[:split]
X_test= x.iloc[split:]
Y_train= y.iloc[:split]
Y_test= y.iloc[split:]
print(f"TRAIN: {len(X_train)} mesi")
print(f"TEST: {len(X_test)} mesi ")
# ora costruiamo il primo modello: una regressione ridge. Cioè una regresione lineare che evita il problema dell'overfitting, cosi il modello evita di "memorizzare" i dati invece di imparare a capire relazioni
from sklearn.linear_model import Ridge
modello= Ridge(alpha=0.1)
modello.fit(X_train, Y_train)
pred= modello.predict(X_test)
mae_ridge= mean_absolute_error(Y_test, pred)
print(f"MAE ridge: {mae_ridge: .4f}")
print(f"MAE baseline: {mae_baseline: .4f}")
# il modello ovviamente ha un MAE molto peggiore rispetto che il baseline perchè le feature hanno scale diverse, la soluzione è la standardizzazione.
#cioè trasformare ogni feature in modo che abbia media 0 e std 1
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
X_train_scaled= scaler.fit_transform(X_train)
X_test_scaled= scaler.transform(X_test)
modello2= Ridge(alpha=0.1)
modello2.fit(X_train_scaled, Y_train)
pred2= modello2.predict(X_test_scaled)
mae_ridge2= mean_absolute_error(Y_test, pred2)
print(f"MAE ridge: {mae_ridge2: .4f}")
print(f"MAE baseline: {mae_baseline: .4f}")
#il modello continua ad essere instabile nonostante il miglioramento per via delle troppe features (42) rispetto alle osservazioni (130)
#rendiamo il modello molto piu conservativo per penalizzare i coefficienti.
modello3= Ridge(alpha=300)
modello3.fit(X_train_scaled, Y_train)
pred3= modello3.predict(X_test_scaled)
mae_ridge_3= mean_absolute_error(Y_test, pred3)
print(f"MAE ridge: {mae_ridge_3: .4f}")
print(f"MAE baseline: {mae_baseline: .4f}")

#il modello 3 0.59 batte il baseline 0.70 a seguito delle trasformazioni di stazionarietà effettuate all'inizio del codice
#adesso ne creiamo uno ancora piu potente il Random Forest. Un insieme di alberi decisionali che cattura relazioni non lineari che ridge non riesce a vedere
from sklearn.ensemble import RandomForestRegressor
rf= RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, Y_train)
pred_rf= rf.predict(X_test_scaled)
mae_rf= mean_absolute_error(Y_test, pred_rf)
print(F"MAE Random Forest: {mae_rf: .4f}")
# stesso margine di errore rispetto al modello ridge quindi proviamo il gradient boosting, simile al random forest ma costruisce gli alberi in sequenza
#piuttosto che in parallelo cioè ogni albero prova a corregere gli errori del precedente.
from sklearn.ensemble import GradientBoostingRegressor
gb= GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.1) #learning rate è quanto un albero corregge un altro
gb.fit(X_train_scaled, Y_train)
pred_gb= gb.predict(X_test_scaled)
mae_gb= mean_absolute_error(Y_test, pred_gb)
print(F"MAE Gradient Boosting: {mae_gb: .4f}")
#margine di errore 0.57 il migliore fin ora è il gradient boosting
#adesso estrapoliamo la feature importance ovvero quali variabili pesano di piu nelle previsioni del modello
importanze= pd.Series(gb.feature_importances_, index=x.columns)
importanze_top= importanze.nlargest(10)
print(importanze_top)
#il numero rappresenta quanto ogni feature contribuisce alla riduzione totale del modello sommano tutti a 1.
#quella piu dominante è l'inflazione italia come avevamo notato nel test di granger
fig, ax= plt.subplots(figsize=(12,5))
ax.plot(Y_test.index, Y_test.values,label="Reale", color="black", linewidth=2)
ax.plot(Y_test.index, pred_gb, label="Previsto", color="#E74C3C", linestyle="--",linewidth=1.5)
ax.fill_between(Y_test.index, Y_test.values, pred_gb, alpha=0.3, color="red")
ax.set_xlabel("Data")
ax.set_title("Inflazione eurozona: reale vs previsto", fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f"output/grafici/risultati modelli/Inflazione_eurozona_reale-vs_previsto.png", dpi=300, bbox_inches="tight")
#facciamo lo stesso per lo spread.
#prima controlliamo la stazionarietà
from statsmodels.tsa.stattools import adfuller
#applichiamo diff per stazionarlo e controlliamo che frequenza ha (1 mese)
spread_diff= spread.diff(1).squeeze().replace([np.inf, -np.inf], np.nan).dropna()
print(spread.index[:3])
baseline_pred= spread_diff.squeeze().reindex(feature_matrix.index)
y2_aligned= y2.reindex(baseline_pred.index)
mae_baseline= mean_absolute_error(y2_aligned, baseline_pred)
print(f"MAE baseline: {mae_baseline: .4f}")
Y_train2= y2.iloc[:split]
Y_test2= y2.iloc[split:]
gb2= GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.1)
gb2.fit(X_train_scaled, Y_train2)
pred_gb2= gb2.predict(X_test_scaled)
mae_gb2= mean_absolute_error(Y_test2, pred_gb2)
print(f"MAE Gradient boosting spread: {mae_gb2: .4f}")
fig,ax= plt.subplots(figsize=(12,5))
ax.plot(Y_test2.index, Y_test2.values, label="Reale", color="black", linewidth=2)
ax.plot(Y_test2.index, pred_gb2, label="Previsto", color="red", linewidth=1.5, linestyle="--")
ax.fill_between(Y_test2.index, Y_test2.values, pred_gb2, color="red", alpha=0.3)
ax.set_xlabel("Data")
ax.set_title("Spread BTP-Bund: reale vs previsto", fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f"output/grafici/risultati modelli/Spread_BTP-Bund_reale_vs_previsto.png", dpi=300, bbox_inches="tight")

#salviamo il modello che abbiamo creato con joblib

import joblib
import os
os.makedirs("models", exist_ok=True)
joblib.dump(gb, "models/gb_inflazione.pkl")
joblib.dump( gb2, "models/gb_spread.pkl")
joblib.dump(scaler, "models/scaler.pkl")

x.to_csv("models/feature_matrix.csv")