import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
#risolviamo il problema della stazionarietà
#Una serie storica si definisce stazionaria quando la sua
#media e la sua varianza non cambiano nel tempo ma oscillano
#intorno a un livello fisso senza trend
#facciamo un ADF test: l'ipotesi nulla è "la serie non è stazionaria cioè ha una radice unitaria"
#l'ipotesi alternativa è che sia stazionaria.
from utils import carica_dati
dati= carica_dati()
from statsmodels.tsa.stattools import adfuller
risultati= {}
for nome,series in dati.items():
    risultati[nome]= adfuller(series)[1]

print(risultati)

dati["tasso_bce"]= dati["tasso_bce"].resample("MS").last()
dati["pil_italia"]= dati["pil_italia"].resample("MS").ffill()
dati["pil eurozona"]= dati["pil eurozona"].resample("MS").ffill()

trasformazioni= {"pil eurozona": ("pct",12),
                 "pil_italia": ("pct",12),
                 "tasso_bce": ("diff",1),
                 "disoccupazione_eurozona": ("pct",1),
                 "disoccupazione_italia": ("pct",1),
                 "m3_eurozona": ("diff",1),
                 "inflazione eurozona": ("diff",1),
                 "inflazione italia" : ("diff",1)
}
risultati_puliti={}
for nome,n in trasformazioni.items():
    if n[0]=="pct":
        risultati_puliti[nome]= dati[nome].pct_change(n[1]).squeeze().replace([np.inf, -np.inf], np.nan).dropna()
    else:
        risultati_puliti[nome]= dati[nome].diff(1).squeeze().replace([np.inf, -np.inf], np.nan).dropna()


test2={}
for nome,series in risultati_puliti.items():
    test2[nome]= adfuller(series)[1]

print(test2)
#adesso calcoliamo per ogni coppia di serie, se una serie in un certo numero
#di mesi è stata utile per prevedere una sere di ora.
#ssr_ftest è un F-test che confronta due modelli:

#Modello ristretto: prevedo Y usando solo i valori passati di Y
#Modello completo: prevedo Y usando i valori passati di Y più i valori passati di X
#l'F-test dà un p-value basso — significa che X aggiunge informazione predittiva su Y.
#creiamo delle coppie con
from itertools import combinations, permutations
from statsmodels.tsa.stattools import grangercausalitytests
coppie= list(permutations(risultati_puliti.keys(),2))
risultati_granger={}
for x,y in coppie:
    df_test= pd.concat([risultati_puliti[x], risultati_puliti[y]], axis=1).dropna()
    risultato= grangercausalitytests(df_test,maxlag=6, verbose=False)
    p_values= []
    for lag, res in risultato.items():
        p_value= res[0]["ssr_ftest"][1]
        p_values.append(p_value)
    risultati_granger[x,y]= min(p_values)
    

print(risultati_granger)
#costruiamo una matrice quadrata dove righe e colonne sono i nomi delle serie
granger_matrix= pd.Series(risultati_granger).unstack()
fig,ax= plt.subplots(figsize=(10,8))
sns.heatmap(granger_matrix, cmap= "Reds_r", annot=True,fmt=".2f", cbar=True, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
plt.tight_layout()

#nelle righe vedi cosa PREVEDE l'elemento nelle colonne vedi
#da cosa è PREVISTO l'elemento
fig.savefig(f"output/grafici/heatmap granger.png", dpi=300,bbox_inches="tight")
granger_matrix.to_csv("models/granger_matrix.csv")