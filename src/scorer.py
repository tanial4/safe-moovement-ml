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
        bundle    = _load()
        feat_col  = bundle["features"]
        scaler    = bundle["scaler"]
        model     = bundle["model"]
        score_min = bundle.get("score_min", -0.8)
        score_max = bundle.get("score_max", -0.3)

        X = pd.DataFrame([{f: features.get(f, 0.0) for f in feat_col}])

        # Imputar features faltantes con la media del scaler
        for i, col in enumerate(feat_col):
            if col not in features or features[col] is None:
                X[col] = scaler.mean_[i]

        X_scaled = scaler.transform(X)
        raw      = model.score_samples(X_scaled)[0]

        # Normalizar al rango 0-1 usando score_min y score_max del training
        score = (score_max - raw) / (score_max - score_min)
        return float(np.clip(score, 0.0, 1.0))

    except FileNotFoundError:
        return 0.5
    except Exception:
        return 0.5


def evaluar_reglas_clinicas(features: dict) -> dict:
    alertas    = []
    penalizacion = 0

    # ── Temperatura corporal ──────────────────────────────────────────────
    temp = features.get("body_temp", 38.5)
    if temp < 38.0:
        alertas.append({"tipo": "hipotermia", "valor": temp, "severidad": "warning"})
        penalizacion += 15
    elif 39.4 <= temp <= 39.9:
        alertas.append({"tipo": "fiebre_leve", "valor": temp, "severidad": "warning"})
        penalizacion += 12
    elif temp > 39.9:
        alertas.append({"tipo": "fiebre_alta", "valor": temp, "severidad": "critical"})
        penalizacion += 25

    # ── Frecuencia cardíaca ───────────────────────────────────────────────
    hr_mean = features.get("heart_rate_mean")
    hr_std  = features.get("heart_rate_std")
    if hr_mean is not None:
        if hr_mean < 48:
            alertas.append({"tipo": "bradicardia", "valor": hr_mean, "severidad": "critical"})
            penalizacion += 20
        elif hr_mean > 84:
            alertas.append({"tipo": "taquicardia", "valor": hr_mean, "severidad": "warning"})
            penalizacion += 15
    if hr_std is not None and hr_std > 15:
        alertas.append({"tipo": "arritmia_posible", "valor": hr_std, "severidad": "warning"})
        penalizacion += 10

    # ── Frecuencia respiratoria ───────────────────────────────────────────
    rr = features.get("respiratory_rate")
    if rr is not None:
        if rr < 18:
            alertas.append({"tipo": "bradipnea", "valor": rr, "severidad": "warning"})
            penalizacion += 10
        elif rr > 44:
            alertas.append({"tipo": "taquipnea", "valor": rr, "severidad": "warning"})
            penalizacion += 15
        elif rr > 60:
            alertas.append({"tipo": "dificultad_respiratoria", "valor": rr, "severidad": "critical"})
            penalizacion += 10  # acumulativo

    # ── Rumia ─────────────────────────────────────────────────────────────
    rum = features.get("rumination_min")
    if rum is not None and rum < 280:
        alertas.append({"tipo": "rumia_baja", "valor": rum, "severidad": "warning"})
        penalizacion += 10
    if rum is not None and rum < 150:
        alertas.append({"tipo": "rumia_critica", "valor": rum, "severidad": "critical"})
        penalizacion += 10  # acumulativo

    # ── Hidratación ───────────────────────────────────────────────────────
    hyd = features.get("hydration_freq")
    if hyd is not None and hyd < 7:
        alertas.append({"tipo": "hidratacion_baja", "valor": hyd, "severidad": "warning"})
        penalizacion += 8
    if hyd is not None and hyd < 4:
        alertas.append({"tipo": "hidratacion_critica", "valor": hyd, "severidad": "critical"})
        penalizacion += 10  # acumulativo

    # ── Humedad ambiental ─────────────────────────────────────────────────
    hum = features.get("humidity")
    if hum is not None and hum < 40:
        alertas.append({"tipo": "humedad_baja", "valor": hum, "severidad": "warning"})
        penalizacion += 5

    # ── Estrés térmico (THI) ──────────────────────────────────────────────
    thi = features.get("thi_score")
    if thi is not None:
        if 72 <= thi < 78:
            alertas.append({"tipo": "estres_termico_leve", "valor": thi, "severidad": "info"})
            penalizacion += 5
        elif thi >= 78:
            alertas.append({"tipo": "estres_termico_alto", "valor": thi, "severidad": "warning"})
            penalizacion += 12

    return {
        "penalizacion_clinica": min(penalizacion, 40),  # cap para no dominar el HIS
        "alertas_clinicas":     alertas,
    }


def calcular_his(
    features: dict,
    anomaly_score: float,
    clinical_risk_score: Optional[float] = None
) -> tuple:
    his = 100

    # ── Penalización por anomalía de comportamiento (IF) ──────────────────
    his -= int(anomaly_score * 40)

    # ── Penalizaciones por comportamiento postural ────────────────────────
    lying = features.get("lying_ratio", 0)
    if lying > 0.85:
        his -= 8
    if features.get("mean_accel", 1.0) < 0.08:
        his -= 6
    if features.get("temp_trend", 0) > 0.05:
        his -= 4

    # ── Penalizaciones por reglas clínicas duras ──────────────────────────
    clinica = evaluar_reglas_clinicas(features)
    his    -= clinica["penalizacion_clinica"]

    # ── Penalización por riesgo clínico veterinario (si hay VetRecord) ────
    if clinical_risk_score is not None:
        his -= int(clinical_risk_score * 15)

    his_final = max(0, min(100, his))
    return his_final, clinica["alertas_clinicas"]


def estado_desde_his(his: int) -> str:
    """HIS < 86 → alerta, según especificación del proyecto."""
    if his >= 86:
        return "saludable"
    if his >= 60:
        return "alerta"
    return "critico"