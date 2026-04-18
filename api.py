"""
api.py — microservicio ML de GanaderIA
Iniciar: uvicorn api:app --port 8001 --reload
"""
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.scorer  import get_anomaly_score, calcular_his, estado_desde_his
from src.features import build_features_from_readings

app = FastAPI(title="GanaderIA ML Service", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────────────────────

class FeaturesIn(BaseModel):
    cow_id:       str
    window_start: float
    window_end:   float
    mean_accel:   float
    std_accel:    float
    body_temp:    float
    lying_ratio:  float
    temp_trend:   float


class VetRecordIn(BaseModel):
    body_temp:  Optional[float] = None   # temperatura rectal medida por vet
    bhba:       Optional[float] = None   # cetosis
    rcs:        Optional[int]   = None   # mastitis
    bcs:        Optional[float] = None   # condición corporal


class ScoreRequest(BaseModel):
    features:   FeaturesIn
    vet_record: Optional[VetRecordIn] = None


class RawReadingsRequest(BaseModel):
    cow_id:   str
    readings: list   # lista de dicts: {timestamp, accel_x, accel_y, accel_z, body_temp}


# ── Helpers ──────────────────────────────────────────────────────────────────

def calcular_clinical_risk(vet: VetRecordIn) -> Optional[dict]:
    """
    Estimación por reglas hasta que haya datos para entrenar RandomForest.
    Devuelve None si no hay datos útiles en el registro.
    """
    if not any([vet.body_temp, vet.bhba, vet.rcs, vet.bcs]):
        return None

    risk = 0.0
    if vet.body_temp and vet.body_temp > 39.5:
        risk += 0.3
    if vet.bhba and vet.bhba > 1.2:
        risk += 0.4    # cetosis subclínica
    if vet.rcs and vet.rcs > 400:
        risk += 0.3    # mastitis subclínica
    if vet.bcs and vet.bcs < 2.5:
        risk += 0.2    # condición corporal baja

    risk  = min(1.0, risk)
    clase = "enferma" if risk > 0.6 else "sospechosa" if risk > 0.3 else "sana"

    return {
        "clinical_risk_score": round(risk, 3),
        "clase_probable":      clase,
        "confianza":           round(0.5 + risk * 0.3, 3),
    }


def build_response(cow_id: str, features: dict, vet: Optional[VetRecordIn] = None) -> dict:
    clinical_risk        = calcular_clinical_risk(vet) if vet else None
    clinical_risk_score  = clinical_risk["clinical_risk_score"] if clinical_risk else None

    anomaly_score = get_anomaly_score(features)
    his           = calcular_his(features, anomaly_score, clinical_risk_score)

    return {
        "cow_id":        cow_id,
        "timestamp":     time.time(),
        "anomaly_score": round(anomaly_score, 3),
        "his":           his,
        "estado":        estado_desde_his(his),
        "alerta":        anomaly_score > 0.7,
        "clinical_risk": clinical_risk,
    }


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.post("/score")
def score(req: ScoreRequest):
    """
    Endpoint principal — Dev 2 llama este.
    Recibe features precalculadas → devuelve anomaly_score + HIS.
    """
    return build_response(
        cow_id   = req.features.cow_id,
        features = req.features.model_dump(),
        vet      = req.vet_record,
    )


@app.post("/score/raw")
def score_raw(req: RawReadingsRequest):
    """
    Alternativa — Dev 2 manda lecturas crudas sin calcular features.
    El servicio hace el cálculo internamente.
    """
    features = build_features_from_readings(req.readings, req.cow_id)

    if features is None:
        raise HTTPException(
            status_code=422,
            detail=f"Lecturas insuficientes. Mínimo 5, recibidas: {len(req.readings)}"
        )

    return build_response(cow_id=req.cow_id, features=features)