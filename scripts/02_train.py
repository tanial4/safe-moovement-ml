import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import json
from pathlib import Path
import sys

FEATURES = [
    "mean_accel", "std_accel", "body_temp", "lying_ratio", "temp_trend",
    "humidity", "ambient_temp", "thi_score", "elevation_std",
    "heart_rate_mean", "heart_rate_std", "respiratory_rate",
    "rumination_min", "hydration_freq",
]

CONTAMINATION = 0.05
N_ESTIMATORS  = 200
RANDOM_STATE  = 42
MIN_WINDOWS   = 50   # mínimo para entrenar modelo individual

CLEAN_PATH = Path("data/clean/mmcows_clean.csv")
MODEL_DIR  = Path("models")
META_PATH  = MODEL_DIR / "model_meta.json"


def load_data():
    if not CLEAN_PATH.exists():
        print(f"ERROR: No se encontró {CLEAN_PATH}")
        print("Ejecuta primero: python scripts/01_clean_mmcows.py")
        sys.exit(1)
    df = pd.read_csv(CLEAN_PATH)
    available = [f for f in FEATURES if f in df.columns]
    missing   = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"  AVISO: features no encontradas: {missing}")
    print(f"Datos totales: {len(df)} ventanas · {df['cow_id'].nunique()} vacas")
    print(f"Features disponibles: {len(available)}\n")
    return df, available


def train_one(X_raw, label):
    X = X_raw.copy()
    for col in X.columns:
        if X[col].isna().sum() > 0:
            X[col] = X[col].fillna(X[col].median())
    X = X.dropna()

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        max_samples="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    scores      = model.score_samples(X_scaled)
    n_anomalies = (model.predict(X_scaled) == -1).sum()

    print(f"  {label}: {len(X)} ventanas · "
          f"{n_anomalies} anómalas ({n_anomalies/len(X)*100:.1f}%) · "
          f"score [{scores.min():.3f}, {scores.max():.3f}]")

    return model, scaler, float(scores.min()), float(scores.max())


def save_model(model, scaler, features, score_min, score_max, path, cow_id=None):
    bundle = {
        "model":     model,
        "scaler":    scaler,
        "features":  features,
        "score_min": score_min,
        "score_max": score_max,
    }
    if cow_id is not None:
        bundle["cow_id"] = cow_id
    joblib.dump(bundle, path)


def main():
    print("=== 02_train.py (v3 — modelos por vaca) ===\n")
    MODEL_DIR.mkdir(exist_ok=True)

    df, features = load_data()
    meta = {"features": features, "contamination": CONTAMINATION,
            "n_estimators": N_ESTIMATORS, "per_cow_models": {}, "global_model": {}}

    # ── Modelos individuales por vaca ──────────────────────────────────────
    print("Entrenando modelos individuales:")
    trained_cows = []

    for cow_id in sorted(df['cow_id'].unique()):
        X_cow = df[df['cow_id'] == cow_id][features]

        if len(X_cow) < MIN_WINDOWS:
            print(f"  C{cow_id:02d}: muy pocas ventanas ({len(X_cow)}), skip")
            continue

        model, scaler, s_min, s_max = train_one(X_cow, f"C{cow_id:02d}")
        path = MODEL_DIR / f"cow_{cow_id:02d}_model.pkl"
        save_model(model, scaler, features, s_min, s_max, path, cow_id=int(cow_id))

        meta["per_cow_models"][str(cow_id)] = {
            "path":      str(path),
            "n_windows": int(len(X_cow)),
            "score_min": s_min,
            "score_max": s_max,
            "offset":    float(model.offset_),
        }
        trained_cows.append(cow_id)

    # ── Modelo global fallback ─────────────────────────────────────────────
    print("\nEntrenando modelo global (fallback para vacas nuevas):")
    X_all = df[features]
    model_g, scaler_g, s_min_g, s_max_g = train_one(X_all, "GLOBAL")
    global_path = MODEL_DIR / "ganaderIA_model.pkl"
    save_model(model_g, scaler_g, features, s_min_g, s_max_g, global_path)

    meta["global_model"] = {
        "path":      str(global_path),
        "n_windows": int(len(X_all)),
        "score_min": s_min_g,
        "score_max": s_max_g,
        "offset":    float(model_g.offset_),
    }

    META_PATH.write_text(json.dumps(meta, indent=2))

    print(f"\nResumen:")
    print(f"  Modelos individuales: {len(trained_cows)} vacas")
    print(f"  Modelo global:        {global_path}")
    print(f"  Metadatos:            {META_PATH}")
    print("\nListo. Ejecuta: uvicorn api:app --port 8001 --reload")


if __name__ == "__main__":
    main()