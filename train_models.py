"""
train_models.py
---------------
Script untuk melatih model XGBoost dari stockmaster_preprocessed.csv
dan menyimpan model + scaler ke folder model/.

Jalankan sekali:
    python train_models.py

Output:
    model/xgb_model.pkl   — model XGBoost terlatih
    model/scaler.pkl      — MinMaxScaler yang digunakan saat training
    model/model_metrics.pkl — metrik evaluasi (RMSE, MAE, R²)
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# ── Config ────────────────────────────────────────────────────
CSV_FILE   = "stockmaster_preprocessed.csv"
MODEL_DIR  = "model"
FEATURES   = ['Close_Price', 'Basic EPS', 'DER', 'ROA', 'ROE',
               'MA5', 'MA20', 'Volatility20']
TARGET     = 'Target'

XGB_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

# ── Setup ─────────────────────────────────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────
print(f"[INFO] Membaca {CSV_FILE} ...")
df = pd.read_csv(CSV_FILE)
df['Tanggal'] = pd.to_datetime(df['Tanggal'])
print(f"   {len(df):,} baris, {df['Ticker'].nunique()} ticker")

# ── Fitur & target ────────────────────────────────────────────
X = df[FEATURES].copy()
y = df[TARGET].copy()

# Hapus baris dengan NaN
mask = X.notna().all(axis=1) & y.notna()
X, y = X[mask], y[mask]
print(f"   {len(X):,} baris setelah drop NaN")

# ── Train / test split (time-ordered, no shuffle) ─────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
print(f"   Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── Scaling ───────────────────────────────────────────────────
scaler = MinMaxScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── Training XGBoost ──────────────────────────────────────────
print("\n[INFO] Training XGBoost ...")
xgb_model = XGBRegressor(**XGB_PARAMS)
xgb_model.fit(
    X_train_s, y_train,
    eval_set=[(X_test_s, y_test)],
    verbose=50,
)

# ── Evaluasi ──────────────────────────────────────────────────
y_pred = xgb_model.predict(X_test_s)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
mae    = mean_absolute_error(y_test, y_pred)
r2     = r2_score(y_test, y_pred)

print(f"\n[RESULT] Evaluasi Model:")
print(f"   RMSE : {rmse:,.2f}")
print(f"   MAE  : {mae:,.2f}")
print(f"   R2   : {r2:.4f}")

metrics = {"RMSE": round(rmse, 2), "MAE": round(mae, 2), "R2": round(r2, 4)}

# ── Simpan model & scaler ─────────────────────────────────────
xgb_path     = os.path.join(MODEL_DIR, "xgb_model.pkl")
scaler_path  = os.path.join(MODEL_DIR, "scaler.pkl")
metrics_path = os.path.join(MODEL_DIR, "model_metrics.pkl")

with open(xgb_path, "wb")     as f: pickle.dump(xgb_model, f)
with open(scaler_path, "wb")  as f: pickle.dump(scaler, f)
with open(metrics_path, "wb") as f: pickle.dump(metrics, f)

print(f"\n[OK] Model tersimpan:")
print(f"   {xgb_path}")
print(f"   {scaler_path}")
print(f"   {metrics_path}")
print("\nSelesai! Jalankan 'streamlit run app.py' untuk melihat hasilnya.")
