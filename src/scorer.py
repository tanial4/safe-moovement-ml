import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional

MODEL_DIR   = Path(__file__).parent.parent / "models"
GLOBAL_PATH = MODEL_DIR / "ganaderIA_model.pkl"

# Cache de modelos cargados — se cargan una vez y se reusan
_bundles: dict = {}


def _model_path(cow_id: str) -> Optional[Path]:
    num = ''.join(filter(str.isdigit, cow_id))
    if not num:
        return None
    path = MODEL_DIR / f"cow_{int(num):02d}_model.pkl"
    return path if path.exists() else None


def _load(cow_id: Optional[str] = None) -> dict:
    key = cow_id or "global"

    if key not in _bundles:
        individual = _model_path(cow_id) if cow_id else None

        if individual:
            _bundles[key] = joblib.load(individual)
            _bundles[key]["_source"] = f"individual ({cow_id})"
        elif GLOBAL_PATH.exists():
            _bundles[key] = joblib.load(GLOBAL_PATH)
            _bundles[key]["_source"] = "global (fallback)"
        else:
            raise FileNotFoundError(
                "No se encontró ningún modelo.\n"
                "Ejecuta: python scripts/02_train.py"
            )

    return _bundles[key]


def get_model_source(cow_id: str) -> str:
    return "individual" if _model_path(cow_id) else "global"


def get_anomaly_score(features: dict, cow_id: Optional[str] = None) -> float:
    try:
        bundle    = _load(cow_id)
        feat_col  = bundle["features"]
        scaler    = bundle["scaler"]
        model     = bundle["model"]
        score_min = bundle.get("score_min", -0.8)
        score_max = bundle.get("score_max", -0.3)

        X = pd.DataFrame([{f: features.get(f, 0.0) for f in feat_col}])

        # Imputar features faltantes con la media del scaler
        for i, col in enumerate(feat_col):
            if col not in features or features.get(col) is None:
                X[col] = scaler.mean_[i]

        X_scaled = scaler.transform(X)
        raw      = model.score_samples(X_scaled)[0]
        offset   = model.offset_

        # raw > offset → normal → score negativo → clip a 0
        # raw < offset → anómalo → score positivo → clip a 1
        score = (offset - raw) / abs(offset)
        return float(np.clip(score, 0.0, 1.0))

    except FileNotFoundError:
        return 0.5
    except Exception:
        return 0.5


def evaluar_reglas_clinicas(features: dict) -> dict:
    alertas      = []
    penalizacion = 0

    temp = features.get("body_temp", 38.5)
    if temp < 38.0:
        alertas.append({"tipo": "hipotermia",  "valor": temp, "severidad": "warning"})
        penalizacion += 15
    elif 39.4 <= temp <= 39.9:
        alertas.append({"tipo": "fiebre_leve", "valor": temp, "severidad": "warning"})
        penalizacion += 12
    elif temp > 39.9:
        alertas.append({"tipo": "fiebre_alta", "valor": temp, "severidad": "critical"})
        penalizacion += 25

    hr_mean = features.get("heart_rate_mean")
    hr_std  = features.get("heart_rate_std")
    if hr_mean is not None:
        if hr_mean < 48:
            alertas.append({"tipo": "bradicardia",  "valor": hr_mean, "severidad": "critical"})
            penalizacion += 20
        elif hr_mean > 84:
            alertas.append({"tipo": "taquicardia",  "valor": hr_mean, "severidad": "warning"})
            penalizacion += 15
    if hr_std is not None and hr_std > 15:
        alertas.append({"tipo": "arritmia_posible", "valor": hr_std, "severidad": "warning"})
        penalizacion += 10

    rr = features.get("respiratory_rate")
    if rr is not None:
        if rr < 18:
            alertas.append({"tipo": "bradipnea",  "valor": rr, "severidad": "warning"})
            penalizacion += 10
        elif rr > 44:
            alertas.append({"tipo": "taquipnea",  "valor": rr, "severidad": "warning"})
            penalizacion += 15
        if rr > 60:
            alertas.append({"tipo": "dificultad_respiratoria", "valor": rr, "severidad": "critical"})
            penalizacion += 10

    rum = features.get("rumination_min")
    if rum is not None:
        if rum < 280:
            alertas.append({"tipo": "rumia_baja",    "valor": rum, "severidad": "warning"})
            penalizacion += 10
        if rum < 150:
            alertas.append({"tipo": "rumia_critica", "valor": rum, "severidad": "critical"})
            penalizacion += 10

    hyd = features.get("hydration_freq")
    if hyd is not None:
        if hyd < 7:
            alertas.append({"tipo": "hidratacion_baja",    "valor": hyd, "severidad": "warning"})
            penalizacion += 8
        if hyd < 4:
            alertas.append({"tipo": "hidratacion_critica", "valor": hyd, "severidad": "critical"})
            penalizacion += 10

    hum = features.get("humidity")
    if hum is not None and hum < 40:
        alertas.append({"tipo": "humedad_baja", "valor": hum, "severidad": "warning"})
        penalizacion += 5

    thi = features.get("thi_score")
    if thi is not None:
        if 72 <= thi < 78:
            alertas.append({"tipo": "estres_termico_leve", "valor": thi, "severidad": "info"})
            penalizacion += 5
        elif thi >= 78:
            alertas.append({"tipo": "estres_termico_alto", "valor": thi, "severidad": "warning"})
            penalizacion += 12

    return {
        "penalizacion_clinica": min(penalizacion, 40),
        "alertas_clinicas":     alertas,
    }


def calcular_his(
    features:            dict,
    anomaly_score:       float,
    clinical_risk_score: Optional[float] = None
) -> tuple:
    his = 100

    his -= int(anomaly_score * 40)

    if features.get("lying_ratio", 0) > 0.85:
        his -= 8
    if features.get("mean_accel", 1.0) < 0.08:
        his -= 6
    if features.get("temp_trend", 0) > 0.05:
        his -= 4

    clinica = evaluar_reglas_clinicas(features)
    his    -= clinica["penalizacion_clinica"]

    if clinical_risk_score is not None:
        his -= int(clinical_risk_score * 15)

    return max(0, min(100, his)), clinica["alertas_clinicas"]


def estado_desde_his(his: int) -> str:
    if his >= 86:
        return "saludable"
    if his >= 60:
        return "alerta"
    return "critico"