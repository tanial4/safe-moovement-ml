"""
ai_api.py — microservicio Gemini de GanaderIA
Puerto 8002. El backend (Dev 2) llama este cuando anomaly_score > 0.7.

Iniciar:
    uvicorn ai_api:app --port 8002 --reload
"""
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.gemini_client import analyze, chat

app = FastAPI(title="GanaderIA AI Service", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Historial en memoria por vaca — para dar contexto a Gemini
# En producción esto iría en DB, para el hackathon en memoria es suficiente
_historial: dict[str, list] = {}
HISTORIAL_MAX = 20  # últimas N evaluaciones por vaca


# ── Schemas ──────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    cow_id:           str
    features:         dict
    anomaly_score:    float
    his:              int
    alertas_clinicas: list = []


class ChatRequest(BaseModel):
    mensaje:         str
    historial_chat:  list = []
    contexto_herd:   dict = {}


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "gemini", "timestamp": time.time()}


@app.post("/analyze")
def analyze_cow(req: AnalyzeRequest):
    """
    Analiza el estado de una vaca con Gemini.
    Solo llamar cuando anomaly_score > 0.7.
    """
    if req.anomaly_score <= 0.7:
        raise HTTPException(
            status_code=400,
            detail=f"anomaly_score {req.anomaly_score} no supera el umbral 0.7. "
                   "No es necesario llamar a Gemini."
        )

    historial = _historial.get(req.cow_id, [])

    result = analyze(
        cow_id           = req.cow_id,
        features         = req.features,
        anomaly_score    = req.anomaly_score,
        his              = req.his,
        alertas_clinicas = req.alertas_clinicas,
        historial        = historial,
    )

    # Guardar en historial
    _historial.setdefault(req.cow_id, []).append({
        "timestamp":     result.get("timestamp"),
        "his":           result.get("his"),
        "anomaly_score": req.anomaly_score,
        "body_temp":     req.features.get("body_temp"),
        "estado":        result.get("estado"),
    })

    # Mantener solo los últimos N
    if len(_historial[req.cow_id]) > HISTORIAL_MAX:
        _historial[req.cow_id] = _historial[req.cow_id][-HISTORIAL_MAX:]

    return result


@app.post("/chat")
def chat_with_farmer(req: ChatRequest):
    """
    Chat libre del ganadero con el sistema en lenguaje natural.
    """
    respuesta = chat(
        mensaje        = req.mensaje,
        historial_chat = req.historial_chat,
        contexto_herd  = req.contexto_herd,
    )
    return {
        "respuesta": respuesta,
        "timestamp": time.time(),
    }


@app.get("/historial/{cow_id}")
def get_historial(cow_id: str):
    """
    Devuelve el historial de evaluaciones de una vaca.
    """
    return {
        "cow_id":    cow_id,
        "historial": _historial.get(cow_id, []),
        "total":     len(_historial.get(cow_id, [])),
    }


@app.delete("/historial/{cow_id}")
def clear_historial(cow_id: str):
    """Limpia el historial de una vaca — útil para resetear la demo."""
    _historial.pop(cow_id, None)
    return {"cow_id": cow_id, "cleared": True}