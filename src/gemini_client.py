"""
src/gemini_client.py
Cliente del agente Gemini. Solo se llama cuando anomaly_score > 0.7.
"""
import os
import json
import time
from pathlib import Path
from typing import Optional

import google.generativeai as genai

from src.prompts import SYSTEM_PROMPT, build_analysis_prompt, build_chat_prompt

from dotenv import load_dotenv
load_dotenv()

_model = None


def _get_model():
    global _model
    if _model is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY no encontrada.\n"
                "Agrégala al archivo .env: GEMINI_API_KEY=tu_key"
            )
        genai.configure(api_key=api_key)
        _model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=SYSTEM_PROMPT,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=5000,
            )
        )
    return _model


def analyze(
    cow_id:          str,
    features:        dict,
    anomaly_score:   float,
    his:             int,
    alertas_clinicas: list,
    historial:       list = [],
) -> dict:
    """
    Llama a Gemini para analizar el estado de una vaca anómala.
    Solo llamar cuando anomaly_score > 0.7.

    Devuelve dict con:
        his, estado, condicion_probable, confianza,
        accion_recomendada, justificacion
    O un fallback si Gemini falla.
    """
    try:
        model  = _get_model()
        prompt = build_analysis_prompt(
            features, anomaly_score, his, alertas_clinicas, historial
        )

        response = model.generate_content(prompt)
        text     = response.text.strip()

        # Limpiar markdown si Gemini lo añade
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        print("=== RESPUESTA GEMINI RAW ===")
        print(repr(text))
        print("===========================")

        result         = json.loads(text)
        result["cow_id"]   = cow_id
        result["timestamp"] = time.time()
        result["source"]    = "gemini"
        return result

    except json.JSONDecodeError:
        return _fallback(cow_id, anomaly_score, his,
                         "Error parseando respuesta de Gemini")
    except Exception as e:
        return _fallback(cow_id, anomaly_score, his, str(e))


def chat(mensaje: str, historial_chat: list, contexto_herd: dict) -> str:
    try:
        model    = _get_model()
        prompt   = build_chat_prompt(mensaje, contexto_herd)
        messages = historial_chat + [{"role": "user", "parts": [prompt]}]

        response = model.generate_content(messages)
        text     = response.text.strip()

        # Limpiar markdown si Gemini lo añade
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        return text

    except Exception as e:
        return f"No pude procesar tu consulta en este momento. Error: {str(e)}"


def _fallback(cow_id: str, anomaly_score: float, his: int, error: str) -> dict:
    """
    Respuesta de emergencia si Gemini no está disponible.
    El sistema no se cae — usa el HIS y el score del IF.
    """
    if his < 40:
        estado   = "critico"
        accion   = "Revisión veterinaria inmediata"
        condicion = "Anomalía crítica detectada por sensores"
    elif his < 86:
        estado   = "alerta"
        accion   = "Monitorear de cerca en las próximas horas"
        condicion = "Comportamiento anómalo detectado"
    else:
        estado   = "saludable"
        accion   = "Sin acción requerida"
        condicion = "Sin anomalías"

    return {
        "cow_id":             cow_id,
        "timestamp":          time.time(),
        "his":                his,
        "estado":             estado,
        "condicion_probable": condicion,
        "confianza":          "baja",
        "accion_recomendada": accion,
        "justificacion":      f"Diagnóstico basado en reglas (Gemini no disponible: {error})",
        "source":             "fallback",
    }