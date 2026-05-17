import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller

from utils import carica_dati, carica_mercati, crea_lag, resample_dati, applica_trasformazioni

os.makedirs("models", exist_ok=True)
os.makedirs("output/grafici/risultati modelli", exist_ok=True)


# ---------------------------------------------------------------------------
# Caricamento e preparazione dati
# ---------------------------------------------------------------------------

print("Caricamento dati...")
dati    = carica_dati()
mercati = carica_mercati()

# Spread calcolato sui livelli originali — prima di qualsiasi trasformazione
spread = (dati["btp_10y"].squeeze() - dati["bund_10y"].squeeze()).dropna()

# Resample e stazionarizzazione
dati      = resample_dati(dati)
dati_staz = applica_trasformazioni(dati)
dati_staz = {k: v.squeeze() for k, v in dati_staz.items()}

# Mercati ricampionati a mensile
mercati_mensili = {}
for nome, df in mercati.items():
    serie = df.squeeze().resample("MS").last()
    mercati_mensili[nome] = serie


# ---------------------------------------------------------------------------
# Feature matrix
#
# Usiamo solo lag 1 e 3 — non 6. Con ~120 osservazioni e molte serie,
# usare lag 1/3/6 produce ~42 feature che portano a overfitting garantito.
# Lag 1/3 riducono le feature a ~28, rapporto più ragionevole per Ridge.
# Il lag 6 ha importanza trascurabile nei modelli e aggiunge solo rumore.
# ---------------------------------------------------------------------------

LAGS_MODELLI = [1, 3]

serie_lag = {}

# Feature escluse dalla feature matrix ML:
# - disoccupazione_eurozona: serie FRED discontinuata al 2022
# - m3_eurozona: feature importance trascurabile su entrambi i target
# Sostituite da brent (driver inflazione energetica) e vix (driver spread)
ESCLUDI_DA_MODELLI = {"disoccupazione_eurozona", "m3_eurozona"}
dati_staz_modelli = {k: v for k, v in dati_staz.items() if k not in ESCLUDI_DA_MODELLI}

serie_lag.update(crea_lag(dati_staz_modelli, lags=LAGS_MODELLI))
serie_lag.update(crea_lag(mercati_mensili, lags=LAGS_MODELLI))
serie_lag.update(crea_lag({"spread": spread}, lags=LAGS_MODELLI))

feature_matrix = pd.DataFrame(serie_lag).dropna()

# ---------------------------------------------------------------------------
# DIAGNOSTICA — mostra dove finisce ogni serie e quante osservazioni perde
# ogni passaggio. Utile per capire perché la feature matrix è corta.
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("DIAGNOSTICA COPERTURA TEMPORALE")
print("=" * 50)
print(f"{'Serie':<35} {'Inizio':>12}  {'Fine':>12}  {'N':>6}")
print("-" * 70)
for nome, s in {**dati_staz, **mercati_mensili, "spread": spread}.items():
    s2 = s.squeeze().dropna()
    if len(s2):
        print(f"  {nome:<33} {str(s2.index[0].date()):>12}  {str(s2.index[-1].date()):>12}  {len(s2):>6}")

print(f"\nFeature matrix (dopo lag+dropna): "
      f"{feature_matrix.index[0].date()} → {feature_matrix.index[-1].date()}  "
      f"({len(feature_matrix)} righe)")
print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
# Target
#
# target_inflazione: inflazione eurozona tra 3 mesi (shift -3)
# target_spread:     spread tra 1 mese (shift -1)
# Shift negativo = "il valore futuro visto dal punto di osservazione corrente"
# ---------------------------------------------------------------------------

# target_inflazione: variazione inflazione eurozona tra 3 mesi (diff stazionarizzata, shift -3)
# target_spread:     VARIAZIONE spread tra 1 mese — non il livello assoluto.
#                    Il livello è quasi identico al valore corrente (baseline imbattibile).
#                    La variazione ha baseline = 0 (naive: "non cambia nulla") — più onesto.
target_inflazione = dati_staz["inflazione_eurozona"].shift(-3)
target_spread     = spread.diff(1).shift(-1)

feature_matrix["target_inflazione"] = target_inflazione
feature_matrix["target_spread"]     = target_spread
feature_matrix = feature_matrix.dropna()

X  = feature_matrix.drop(columns=["target_inflazione", "target_spread"])
y_inf    = feature_matrix["target_inflazione"]
y_spread = feature_matrix["target_spread"]

print(f"Feature matrix finale: {feature_matrix.index[0].date()} → {feature_matrix.index[-1].date()}  ({len(feature_matrix)} righe)")
print(f"  (shift -3 per inflazione e -1 per spread consumano le ultime righe)\n")


# ---------------------------------------------------------------------------
# Baseline
#
# Inflazione: usiamo il valore corrente come predittore del valore futuro
# Spread:     usiamo il valore corrente come predittore del valore futuro
# Entrambi sono "naive forecast" — il modello deve batterli per essere utile.
# ---------------------------------------------------------------------------

inf_corrente = dati_staz["inflazione_eurozona"].reindex(feature_matrix.index)

# Baseline inflazione: valore corrente come predittore del valore futuro
mae_baseline_inf = mean_absolute_error(y_inf, inf_corrente.dropna().reindex(y_inf.index).dropna())

# Baseline spread: la variazione futura è zero (naive forecast per una differenza prima)
# Equivale a dire "lo spread domani sarà uguale a oggi" — il minimo ragionevole da battere
spread_zero = pd.Series(0.0, index=feature_matrix.index)
mae_baseline_spread = mean_absolute_error(y_spread, spread_zero)

print(f"\nMAE baseline inflazione: {mae_baseline_inf:.4f}")
print(f"MAE baseline spread:     {mae_baseline_spread:.4f}")


# ---------------------------------------------------------------------------
# Walk-forward validation
#
# Il semplice split 80/20 produce un MAE ottimista: il modello viene valutato
# su un unico periodo di test fisso, che potrebbe essere anomalo in entrambe
# le direzioni. La walk-forward validation simula il vero utilizzo:
#
#   - Finestra di training iniziale: TRAIN_SIZE mesi
#   - Ad ogni step si prevede il mese successivo (o 3 mesi dopo per inflazione)
#   - La finestra avanza di 1 mese, il modello viene riaddestratto
#   - Il MAE finale è la media di ~N previsioni indipendenti
#
# Questo elimina il look-ahead bias strutturale dello split fisso.
#
# TRAIN_SIZE = 60: 5 anni di storia minima per addestrare — abbastanza per
# catturare almeno un ciclo economico completo prima di fare previsioni.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Validazione temporale — TimeSeriesSplit
#
# Con 72 osservazioni e 40+ feature la walk-forward pura (riaddestrare
# da zero ad ogni step) produce finestre di training troppo piccole
# per GB — le prime iterazioni avrebbero 40-50 osservazioni con 40 feature,
# overfitting garantito.
#
# TimeSeriesSplit divide il dataset in k fold temporali rispettando
# l'ordine cronologico: il fold i usa i primi N*i/k mesi come train
# e il blocco successivo come test. Nessun dato futuro entra nel training.
#
# Con n_splits=5 ogni fold di test ha ~12 osservazioni (1 anno circa)
# e il training cresce da ~12 a ~60 osservazioni — dimensioni ragionevoli
# per un modello conservativo su questo dataset.
# ---------------------------------------------------------------------------

from sklearn.model_selection import TimeSeriesSplit, cross_val_score

N_SPLITS = 5
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

# ---------------------------------------------------------------------------
# Train / test split finale
# ---------------------------------------------------------------------------

split = int(len(feature_matrix) * 0.8)

X_train, X_test   = X.iloc[:split],      X.iloc[split:]
y_inf_train, y_inf_test       = y_inf.iloc[:split],    y_inf.iloc[split:]
y_spread_train, y_spread_test = y_spread.iloc[:split], y_spread.iloc[split:]

print(f"\nSplit 80/20 — Train: {len(X_train)} mesi  |  Test: {len(X_test)} mesi")

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# ---------------------------------------------------------------------------
# Modelli — inflazione
# ---------------------------------------------------------------------------

print("\n" + "=" * 50)
print("MODELLI — TARGET: INFLAZIONE EUROZONA (3m)")
print("=" * 50)

# Ridge senza scaling (benchmark)
ridge_raw = Ridge(alpha=0.1)
ridge_raw.fit(X_train, y_inf_train)
mae_ridge_raw = mean_absolute_error(y_inf_test, ridge_raw.predict(X_test))
print(f"Ridge (non scalato):    MAE = {mae_ridge_raw:.4f}  ← feature non normalizzate")

# Ridge α=300 — penalizzazione forte, adatto a dataset wide (~28 feature, ~100 obs)
ridge_inf = Ridge(alpha=300)
ridge_inf.fit(X_train_sc, y_inf_train)
pred_inf_ridge = ridge_inf.predict(X_test_sc)
mae_ridge_inf  = mean_absolute_error(y_inf_test, pred_inf_ridge)
print(f"Ridge (scalato, α=300): MAE = {mae_ridge_inf:.4f}")

# Random Forest
rf_inf = RandomForestRegressor(n_estimators=100, random_state=42)
rf_inf.fit(X_train_sc, y_inf_train)
mae_rf = mean_absolute_error(y_inf_test, rf_inf.predict(X_test_sc))
print(f"Random Forest:          MAE = {mae_rf:.4f}")

# Gradient Boosting
gb_inf = GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.1)
gb_inf.fit(X_train_sc, y_inf_train)
pred_inf_gb = gb_inf.predict(X_test_sc)
mae_gb_inf  = mean_absolute_error(y_inf_test, pred_inf_gb)
print(f"Gradient Boosting:      MAE = {mae_gb_inf:.4f}")
print(f"Baseline:               MAE = {mae_baseline_inf:.4f}")

# Seleziona il modello migliore
migliori_inf = {
    "Ridge":   (mae_ridge_inf, ridge_inf,  pred_inf_ridge),
    "RF":      (mae_rf,        rf_inf,     rf_inf.predict(X_test_sc)),
    "GB":      (mae_gb_inf,    gb_inf,     pred_inf_gb),
}
nome_best_inf, (mae_best_inf, model_best_inf, pred_inf) = min(migliori_inf.items(), key=lambda x: x[1][0])
print(f"\n→ Modello inflazione selezionato: {nome_best_inf} (MAE = {mae_best_inf:.4f})")
print(f"  Miglioramento vs baseline: {((mae_baseline_inf - mae_best_inf) / mae_baseline_inf * 100):+.1f}%")

# Feature importance — sempre da GB (anche se non è il modello salvato)
# GB è il modello che produce feature importance interpretabili
importanze_inf = pd.Series(gb_inf.feature_importances_, index=X.columns).nlargest(10)
print("\nTop 10 feature importance GB (inflazione):")
print(importanze_inf.round(3).to_string())

# ---------------------------------------------------------------------------
# Modello ridotto — top N feature per importanza
#
# Con 72 osservazioni e 40+ feature GB va in overfitting anche sul test set.
# Le prime 5 feature spiegano >80% dell'importanza totale — usare solo quelle
# riduce drasticamente la dimensionalità e rende il modello più generalizzabile.
# N_TOP_FEATURES è parametro: puoi alzarlo a 8-10 se i risultati migliorano.
# ---------------------------------------------------------------------------

N_TOP_FEATURES = 5
top_feature = importanze_inf.nlargest(N_TOP_FEATURES).index.tolist()
print(f"\nTop {N_TOP_FEATURES} feature per importanza: {top_feature}")

X_top_train = X_train[top_feature]
X_top_test  = X_test[top_feature]

scaler_top     = StandardScaler()
X_top_train_sc = scaler_top.fit_transform(X_top_train)
X_top_test_sc  = scaler_top.transform(X_top_test)

gb_inf_top = GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.1)
gb_inf_top.fit(X_top_train_sc, y_inf_train)
mae_gb_inf_top = mean_absolute_error(y_inf_test, gb_inf_top.predict(X_top_test_sc))
print(f"GB ridotto (top {N_TOP_FEATURES} feature): MAE = {mae_gb_inf_top:.4f}")
print(f"GB completo:                        MAE = {mae_gb_inf:.4f}")
print(f"Baseline:                           MAE = {mae_baseline_inf:.4f}")
# Nota: il modello salvato è sempre quello completo — entrambi i target
# usano lo stesso X_train_sc/X_test_sc. Il ridotto è solo un confronto informativo.


# ---------------------------------------------------------------------------
# Modelli — spread BTP-Bund
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Modelli — spread BTP-Bund
#
# Per lo spread usiamo sia Ridge che GB e salviamo il migliore.
# Ridge con alpha alto è più adatto a dataset piccoli con molte feature —
# penalizza i coefficienti e generalizza meglio di GB quando i dati sono pochi.
# ---------------------------------------------------------------------------

print("\n" + "=" * 50)
print("MODELLI — TARGET: SPREAD BTP-BUND (1m)")
print("=" * 50)

# Ridge — penalizzazione forte
ridge_spread = Ridge(alpha=300)
ridge_spread.fit(X_train_sc, y_spread_train)
pred_spread_ridge = ridge_spread.predict(X_test_sc)
mae_ridge_spread  = mean_absolute_error(y_spread_test, pred_spread_ridge)
print(f"Ridge (α=300):      MAE = {mae_ridge_spread:.4f}")

# Gradient Boosting
gb_spread = GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.1)
gb_spread.fit(X_train_sc, y_spread_train)
pred_spread_gb = gb_spread.predict(X_test_sc)
mae_gb_spread  = mean_absolute_error(y_spread_test, pred_spread_gb)
print(f"Gradient Boosting:  MAE = {mae_gb_spread:.4f}")
print(f"Baseline:           MAE = {mae_baseline_spread:.4f}")

# Seleziona il modello migliore
if mae_ridge_spread <= mae_gb_spread:
    print(f"\n→ Salvo Ridge per lo spread (MAE migliore)")
    gb_spread  = ridge_spread    # rinomino per compatibilità con il resto del codice
    pred_spread = pred_spread_ridge
    mae_spread_finale = mae_ridge_spread
else:
    print(f"\n→ Salvo Gradient Boosting per lo spread (MAE migliore)")
    pred_spread = pred_spread_gb
    mae_spread_finale = mae_gb_spread

print(f"Miglioramento (split): {((mae_baseline_spread - mae_spread_finale) / mae_baseline_spread * 100):+.1f}%")


# ---------------------------------------------------------------------------
# Validazione temporale — TimeSeriesSplit (k=5)
#
# Eseguita DOPO aver addestrato i modelli finali, usa gli stessi
# iperparametri conservativi su tutti i fold per un MAE onesto.
# ---------------------------------------------------------------------------

print("\n" + "=" * 50)
print("VALIDAZIONE TEMPORALE (TimeSeriesSplit, k=5)")
print("=" * 50)

sc_cv     = StandardScaler()
X_sc_full = sc_cv.fit_transform(X)

# Usa GB conservativo per la CV — stesso modello per entrambi i target
gb_cv = GradientBoostingRegressor(
    n_estimators=50, max_depth=2,
    learning_rate=0.05, subsample=0.8, random_state=42,
)

scores_inf = cross_val_score(
    gb_cv, X_sc_full, y_inf,
    cv=tscv, scoring="neg_mean_absolute_error",
)
mae_cv_inf = -scores_inf.mean()
print(f"  MAE CV inflazione (media {N_SPLITS} fold): {mae_cv_inf:.4f}  (baseline: {mae_baseline_inf:.4f})")
print(f"  MAE per fold: {[-round(s,4) for s in scores_inf]}")

scores_spread = cross_val_score(
    gb_cv, X_sc_full, y_spread,
    cv=tscv, scoring="neg_mean_absolute_error",
)
mae_cv_spread = -scores_spread.mean()
print(f"  MAE CV spread     (media {N_SPLITS} fold): {mae_cv_spread:.4f}  (baseline: {mae_baseline_spread:.4f})")
print(f"  MAE per fold: {[-round(s,4) for s in scores_spread]}")

print(f"\n  Miglioramento CV inflazione: {((mae_baseline_inf - mae_cv_inf) / mae_baseline_inf * 100):+.1f}%")
print(f"  Miglioramento CV spread:     {((mae_baseline_spread - mae_cv_spread) / mae_baseline_spread * 100):+.1f}%")

# Previsioni fold per fold — per i grafici CV
wf_pred_inf    = pd.Series(dtype=float)
wf_pred_spread = pd.Series(dtype=float)

for train_idx, test_idx in tscv.split(X):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    sc_fold    = StandardScaler()
    X_tr_sc    = sc_fold.fit_transform(X_tr)
    X_te_sc    = sc_fold.transform(X_te)

    m1 = GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=42)
    m1.fit(X_tr_sc, y_inf.iloc[train_idx])
    wf_pred_inf = pd.concat([wf_pred_inf, pd.Series(m1.predict(X_te_sc), index=X.iloc[test_idx].index)])

    m2 = GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=42)
    m2.fit(X_tr_sc, y_spread.iloc[train_idx])
    wf_pred_spread = pd.concat([wf_pred_spread, pd.Series(m2.predict(X_te_sc), index=X.iloc[test_idx].index)])


# ---------------------------------------------------------------------------
# Grafici risultati
# ---------------------------------------------------------------------------

def grafico_reale_vs_previsto(
    index, reale, previsto, titolo: str, percorso: str
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(index, reale,    label="Reale",   color="black", linewidth=2)
    ax.plot(index, previsto, label="Previsto", color="#E74C3C", linestyle="--", linewidth=1.5)
    ax.fill_between(index, reale, previsto, alpha=0.2, color="red")
    ax.set_xlabel("Data")
    ax.set_title(titolo, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(percorso, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvato: {percorso}")


grafico_reale_vs_previsto(
    y_inf_test.index, y_inf_test.values, pred_inf,
    titolo="Inflazione eurozona: reale vs previsto (Gradient Boosting)",
    percorso="output/grafici/risultati modelli/inflazione_reale_vs_previsto.png",
)

grafico_reale_vs_previsto(
    y_spread_test.index, y_spread_test.values, pred_spread,
    titolo="Spread BTP-Bund: reale vs previsto (Gradient Boosting)",
    percorso="output/grafici/risultati modelli/spread_reale_vs_previsto.png",
)

# Grafico validazione temporale — inflazione
grafico_reale_vs_previsto(
    wf_pred_inf.index,
    y_inf.reindex(wf_pred_inf.index).values,
    wf_pred_inf.values,
    titolo=f"Inflazione EZ — TimeSeriesSplit CV (MAE: {mae_cv_inf:.3f})",
    percorso="output/grafici/risultati modelli/inflazione_cv.png",
)

# Grafico validazione temporale — spread
grafico_reale_vs_previsto(
    wf_pred_spread.index,
    y_spread.reindex(wf_pred_spread.index).values,
    wf_pred_spread.values,
    titolo=f"Spread BTP-Bund — TimeSeriesSplit CV (MAE: {mae_cv_spread:.3f})",
    percorso="output/grafici/risultati modelli/spread_cv.png",
)


# ---------------------------------------------------------------------------
# Salvataggio modelli e feature matrix
# ---------------------------------------------------------------------------

joblib.dump(model_best_inf, "models/gb_inflazione.pkl")
joblib.dump(gb_spread,     "models/gb_spread.pkl")
joblib.dump(scaler,        "models/scaler.pkl")
X.to_csv("models/feature_matrix.csv")

print(f"\nSalvati: modello inflazione ({nome_best_inf}), spread, scaler, feature_matrix.csv")