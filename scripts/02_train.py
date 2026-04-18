"""
scripts/02_train.py
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import json
from pathlib import Path
import sys

FEATURES      = ["mean_accel", "std_accel", "body_temp", "lying_ratio", "temp_trend"]
CONTAMINATION = 0.05
N_ESTIMATORS  = 200
RANDOM_STATE  = 42
CLEAN_PATH    = Path("data/clean/mmcows_clean.csv")
MODEL_DIR     = Path("models")
MODEL_PATH    = MODEL_DIR / "ganaderIA_model.pkl"
META_PATH     = MODEL_DIR / "model_meta.json"


def main():
    print("=== 02_train.py ===\n")

    if not CLEAN_PATH.exists():
        print(f"ERROR: No se encontró {CLEAN_PATH}")
        print("Ejecuta primero: python scripts/01_clean_mmcows.py")
        sys.exit(1)

    df = pd.read_csv(CLEAN_PATH)
    X  = df[FEATURES].dropna()

    if len(X) < 50:
        print(f"ERROR: Solo {len(X)} ventanas — muy pocas para entrenar.")
        sys.exit(1)

    print(f"Datos: {len(X)} ventanas · {df['cow_id'].nunique()} vacas\n")

    print("Normalizando features...")
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Entrenando IsolationForest...")
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        max_samples="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # Evaluación
    scores      = model.score_samples(X_scaled)
    preds       = model.predict(X_scaled)
    n_anomalies = (preds == -1).sum()
    pct         = n_anomalies / len(X) * 100

    print(f"\nResultados:")
    print(f"  Ventanas anómalas: {n_anomalies} / {len(X)} ({pct:.1f}%)")
    print(f"  Score range:       {scores.min():.3f} → {scores.max():.3f}")
    print(f"  Threshold:         {model.offset_:.3f}")

    if pct < 1.0:
        print("  AVISO: Muy pocas anomalías. Considera bajar contamination a 0.03.")
    elif pct > 15.0:
        print("  AVISO: Demasiadas anomalías. Considera subir contamination a 0.08.")
    else:
        print("  OK: Proporción razonable.")

    # Guardar
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "features": FEATURES}, MODEL_PATH)

    meta = {
        "n_windows":     int(len(X)),
        "features":      FEATURES,
        "contamination": CONTAMINATION,
        "n_estimators":  N_ESTIMATORS,
        "offset":        float(model.offset_),
        "feature_means": dict(zip(FEATURES, scaler.mean_.tolist())),
        "feature_stds":  dict(zip(FEATURES, scaler.scale_.tolist())),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))

    print(f"\nModelo guardado en: {MODEL_PATH}")
    print("Listo. Ejecuta: uvicorn api:app --port 8001 --reload")


if __name__ == "__main__":
    main()