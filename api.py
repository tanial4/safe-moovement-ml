import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.scorer   import (get_anomaly_score, calcular_his, estado_desde_his,
                           get_model_source, evaluar_reglas_clinicas)
from src.features import build_features_from_readings

app = FastAPI(title="GanaderIA ML Service", version="0.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

HIS_ALERT_THRESHOLD = 86


class FeaturesIn(BaseModel):
    cow_id:           str
    window_start:     float
    window_end:       float
    mean_accel:       float
    std_accel:        float
    lying_ratio:      float
    temp_trend:       float
    body_temp:        float
    humidity:         Optional[float] = None
    ambient_temp:     Optional[float] = None
    thi_score:        Optional[float] = None
    elevation_std:    Optional[float] = None
    heart_rate_mean:  Optional[float] = None
    heart_rate_std:   Optional[float] = None
    respiratory_rate: Optional[float] = None
    rumination_min:   Optional[float] = None
    hydration_freq:   Optional[float] = None


class ScoreRequest(BaseModel):
    features: FeaturesIn


class RawReadingsRequest(BaseModel):
    cow_id:   str
    readings: list


def build_response(cow_id: str, features: dict) -> dict:
    anomaly_score          = get_anomaly_score(features, cow_id=cow_id)
    his, alertas_clinicas  = calcular_his(features, anomaly_score)
    model_source           = get_model_source(cow_id)

    return {
        "cow_id":           cow_id,
        "timestamp":        time.time(),
        "anomaly_score":    round(anomaly_score, 3),
        "his":              his,
        "estado":           estado_desde_his(his),
        "alerta":           his < HIS_ALERT_THRESHOLD,
        "alertas_clinicas": alertas_clinicas,
        "model_source":     model_source,  # "individual" o "global"
    }


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.3.0", "timestamp": time.time()}


@app.post("/score")
def score(req: ScoreRequest):
    return build_response(
        cow_id   = req.features.cow_id,
        features = req.features.model_dump(),
    )


@app.post("/score/raw")
def score_raw(req: RawReadingsRequest):
    features = build_features_from_readings(req.readings, req.cow_id)
    if features is None:
        raise HTTPException(422,
            f"Lecturas insuficientes. Mínimo 5, recibidas: {len(req.readings)}")
    return build_response(cow_id=req.cow_id, features=features)


@app.get("/thresholds")
def thresholds():
    return {
        "his_alerta":       HIS_ALERT_THRESHOLD,
        "anomaly_alerta":   0.7,
        "body_temp":        {"normal_max": 39.3, "leve": 39.9, "fiebre": 40.0},
        "heart_rate":       {"min": 48, "max": 84},
        "respiratory_rate": {"min": 18, "max": 44},
        "rumination_min":   {"alerta": 280},
        "hydration_freq":   {"min": 7, "max": 12},
        "humidity":         {"alerta": 40},
        "thi_score":        {"estres_leve": 72, "estres_alto": 78},
    }


@app.get("/models")
def list_models():
    """Lista qué vacas tienen modelo individual y cuáles usan el global."""
    from pathlib import Path
    import json
    meta_path = Path("models/model_meta.json")
    if not meta_path.exists():
        raise HTTPException(404, "Metadatos no encontrados. Ejecuta 02_train.py")
    meta = json.loads(meta_path.read_text())
    return {
        "per_cow_models": list(meta.get("per_cow_models", {}).keys()),
        "global_model":   "ganaderIA_model.pkl",
        "total_individual": len(meta.get("per_cow_models", {})),
    }