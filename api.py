"""
api.py — microservicio ML v2
Puerto 8001. Dev 2 llama este desde el orquestador.
"""
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.scorer   import get_anomaly_score, calcular_his, estado_desde_his, evaluar_reglas_clinicas
from src.features import build_features_from_readings

from src.schemas import *

app = FastAPI(title="GanaderIA ML Service", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

HIS_ALERT_THRESHOLD = 86  # según especificación del proyecto


def build_response(cow_id, features):

    anomaly_score             = get_anomaly_score(features)
    his, alertas_clinicas     = calcular_his(features, anomaly_score)
    estado                    = estado_desde_his(his)
    alerta                    = his < HIS_ALERT_THRESHOLD

    return {
        "cow_id":           cow_id,
        "timestamp":        time.time(),
        "anomaly_score":    round(anomaly_score, 3),
        "his":              his,
        "estado":           estado,
        "alerta":           alerta,
        "alertas_clinicas": alertas_clinicas,
    }


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.2.0", "timestamp": time.time()}


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
    """Devuelve los umbrales clínicos configurados — útil para el frontend."""
    return {
        "his_alerta":        HIS_ALERT_THRESHOLD,
        "anomaly_alerta":    0.7,
        "body_temp":         {"normal_max": 39.3, "leve": 39.9, "fiebre": 40.0},
        "heart_rate":        {"min": 48, "max": 84},
        "respiratory_rate":  {"min": 18, "max": 44},
        "rumination_min":    {"alerta": 280},
        "hydration_freq":    {"min": 7, "max": 12},
        "humidity":          {"alerta": 40},
        "thi_score":         {"estres_leve": 72, "estres_alto": 78},
    }