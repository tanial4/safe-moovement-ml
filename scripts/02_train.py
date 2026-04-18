"""
scripts/02_train.py — v2
Entrena IsolationForest con features expandidas (9 reales + 5 simuladas).
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import json
from pathlib import Path
import sys

# Features reales de MmCows
REAL_FEATURES = [
    "mean_accel",
    "std_accel",
    "body_temp",
    "lying_ratio",
    "temp_trend",
    "humidity",
    "ambient_temp",
    "thi_score",
    "elevation_std",
]

# Features simuladas con distribuciones de literatura veterinaria
SIM_FEATURES = [
    "heart_rate_mean",
    "heart_rate_std",
    "respiratory_rate",
    "rumination_min",
    "hydration_freq",
]

ALL_FEATURES  = REAL_FEATURES + SIM_FEATURES
CONTAMINATION = 0.05
N_ESTIMATORS  = 200
RANDOM_STATE  = 42
CLEAN_PATH    = Path("data/clean/mmcows_clean.csv")
MODEL_DIR     = Path("models")
MODEL_PATH    = MODEL_DIR / "ganaderIA_model.pkl"
META_PATH     = MODEL_DIR / "model_meta.json"

# Umbrales clínicos para alertas por reglas duras
# Estas NO van al IF — se evalúan por separado en scorer.py
CLINICAL_THRESHOLDS = {
    "body_temp":        {"leve": 39.4, "fiebre": 39.9, "critico": 40.5},
    "heart_rate_mean":  {"low": 48,    "high": 84},
    "heart_rate_std":   {"alerta": 15},
    "respiratory_rate": {"low": 18,    "high": 44,  "alerta_std": 10},
    "rumination_min":   {"alerta": 280},
    "hydration_freq":   {"alerta_low": 5, "alerta_high": 12},
    "humidity":         {"alerta": 40},
    "thi_score":        {"estres_leve": 72, "estres_alto": 78},
    "his_alerta":       86,
}


def load_data():
    if not CLEAN_PATH.exists():
        print(f"ERROR: No se encontró {CLEAN_PATH}")
        print("Ejecuta primero: python scripts/01_clean_mmcows.py")
        sys.exit(1)

    df = pd.read_csv(CLEAN_PATH)

    # Verificar qué features están disponibles
    available = [f for f in ALL_FEATURES if f in df.columns]
    missing   = [f for f in ALL_FEATURES if f not in df.columns]

    if missing:
        print(f"  AVISO: features no encontradas: {missing}")
        print(f"  Entrenando con: {available}")

    X = df[available].copy()

    # Imputar NaN con mediana por columna
    for col in X.columns:
        if X[col].isna().sum() > 0:
            median = X[col].median()
            X[col] = X[col].fillna(median)
            print(f"  Imputados {X[col].isna().sum()} NaN en {col} con mediana={median:.3f}")

    X = X.dropna()
    print(f"\nDatos: {len(X)} ventanas · {len(available)} features")
    return X, available


def train(X):
    print("\nNormalizando features...")
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Entrenando IsolationForest (n={N_ESTIMATORS}, contamination={CONTAMINATION})...")
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        max_samples="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    return model, scaler


def evaluate(model, scaler, X):
    X_scaled    = scaler.transform(X)
    scores      = model.score_samples(X_scaled)
    preds       = model.predict(X_scaled)
    n_anomalies = (preds == -1).sum()
    pct         = n_anomalies / len(X) * 100

    print(f"\nResultados:")
    print(f"  Ventanas anómalas:  {n_anomalies} / {len(X)} ({pct:.1f}%)")
    print(f"  Score range:        {scores.min():.3f} → {scores.max():.3f}")
    print(f"  Threshold (offset): {model.offset_:.3f}")
    print(f"  Score min usado en normalización: {scores.min():.3f}")
    print(f"  Score max usado en normalización: {scores.max():.3f}")

    if pct < 1.0:
        print("  AVISO: Muy pocas anomalías. Considera bajar contamination.")
    elif pct > 15.0:
        print("  AVISO: Demasiadas anomalías. Considera subir contamination.")
    else:
        print("  OK: Proporción razonable.")

    return float(scores.min()), float(scores.max())


def save(model, scaler, X, features, score_min, score_max):
    MODEL_DIR.mkdir(exist_ok=True)

    bundle = {
        "model":      model,
        "scaler":     scaler,
        "features":   features,
        "score_min":  score_min,
        "score_max":  score_max,
    }
    joblib.dump(bundle, MODEL_PATH)

    meta = {
        "n_windows":          int(len(X)),
        "features":           features,
        "real_features":      REAL_FEATURES,
        "sim_features":       SIM_FEATURES,
        "contamination":      CONTAMINATION,
        "n_estimators":       N_ESTIMATORS,
        "offset":             float(model.offset_),
        "score_min":          score_min,
        "score_max":          score_max,
        "clinical_thresholds": CLINICAL_THRESHOLDS,
        "feature_means":      dict(zip(features, scaler.mean_.tolist())),
        "feature_stds":       dict(zip(features, scaler.scale_.tolist())),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))

    print(f"\nModelo guardado en:  {MODEL_PATH}")
    print(f"Metadatos en:        {META_PATH}")


def main():
    print("=== 02_train.py (v2 — features expandidas) ===\n")
    X, features     = load_data()
    model, scaler   = train(X)
    score_min, score_max = evaluate(model, scaler, X)
    save(model, scaler, X, features, score_min, score_max)
    print("\nListo. Ejecuta: uvicorn api:app --port 8001 --reload")


if __name__ == "__main__":
    main()