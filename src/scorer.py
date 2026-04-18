"""
src/scorer.py
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional

MODEL_PATH = Path(__file__).parent.parent / "models" / "ganaderIA_model.pkl"
_bundle    = None


def _load():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modelo no encontrado en {MODEL_PATH}\n"
                "Ejecuta: python scripts/02_train.py"
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def get_anomaly_score(features: dict) -> float:
    try:
        bundle   = _load()
        feat_col = bundle["features"]
        scaler   = bundle["scaler"]
        model    = bundle["model"]

        X        = pd.DataFrame([{f: features.get(f, 0.0) for f in feat_col}])
        X_scaled = scaler.transform(X)

        raw      = model.score_samples(X_scaled)[0]
        
        # Normalizar usando el rango real del training
        # score_min ≈ -0.736, score_max ≈ -0.377 (del output de 02_train.py)
        score_min = -0.736
        score_max = -0.377
        score = (score_max - raw) / (score_max - score_min)
        return float(np.clip(score, 0.0, 1.0))

    except FileNotFoundError:
        return 0.5
    except Exception:
        return 0.5


def calcular_his(
    features: dict,
    anomaly_score: float,
    clinical_risk_score: Optional[float] = None
) -> int:
    """
    Health Index Score: 100 = perfectamente sana, 0 = crítica.
    Combina anomaly_score del IF con penalizaciones por umbrales duros.
    """
    his  = 100

    # Penalización por anomalía de comportamiento
    his -= int(anomaly_score * 45)

    # Penalizaciones por umbrales veterinarios
    temp = features.get("body_temp", 38.5)
    if temp > 39.5:
        his -= 15
    if temp > 40.2:
        his -= 15   # acumulativo — fiebre alta penaliza doble

    if features.get("lying_ratio", 0) > 0.85:
        his -= 12

    if features.get("mean_accel", 1.0) < 0.08:
        his -= 8

    if features.get("temp_trend", 0) > 0.05:
        his -= 5

    # Penalización por riesgo clínico (si hay registro veterinario)
    if clinical_risk_score is not None:
        his -= int(clinical_risk_score * 20)

    return max(0, min(100, his))


def estado_desde_his(his: int) -> str:
    if his >= 70:
        return "saludable"
    if his >= 40:
        return "alerta"
    return "critico"