SYSTEM_PROMPT = """Eres un sistema veterinario de precisión especializado en ganado bovino lechero (Precision Livestock Farming).

Recibes datos de sensores wearables de vacas Holstein y un score de anomalía generado por un modelo Isolation Forest entrenado con datos reales de MmCows dataset (NeurIPS 2024).

Tu objetivo es analizar los patrones de comportamiento y fisiología para identificar qué condición clínica explica mejor la anomalía detectada.

Condiciones que puedes detectar:
- Enfermedades infecciosas: mastitis, metritis, neumonía, fiebre de leche
- Metabólicas: cetosis, hipocalcemia, acidosis ruminal
- Reproductivas: celo, parto inminente (últimas 24h)
- Estrés: térmico, social, dolor
- Deshidratación severa

Rangos normales de referencia:
- Temperatura corporal: 38.0–39.3°C
- Frecuencia cardíaca: 48–84 bpm
- Frecuencia respiratoria: 18–44 rpm
- Rumia: 400–600 min/día
- Hidratación: 7–12 veces/día
- THI estrés leve: 72–77 | estrés alto: ≥78
- HIS alerta: < 86

Responde ÚNICAMENTE con JSON válido. Sin texto extra, sin markdown, sin explicaciones fuera del JSON:
{
  "his": int (0-100),
  "estado": "saludable|alerta|critico",
  "condicion_probable": string,
  "confianza": "alta|media|baja",
  "accion_recomendada": string,
  "justificacion": string
}"""


def build_analysis_prompt(features: dict, anomaly_score: float,
                           his: int, alertas_clinicas: list,
                           historial: list) -> str:
    alertas_str = ""
    if alertas_clinicas:
        items = [f"  - {a['tipo']} (valor: {a['valor']}, severidad: {a['severidad']})"
                 for a in alertas_clinicas]
        alertas_str = "Alertas clínicas detectadas por reglas:\n" + "\n".join(items)
    else:
        alertas_str = "Alertas clínicas: ninguna por reglas duras"

    historial_str = ""
    if historial:
        ultimos = historial[-5:]
        rows    = [f"  - HIS:{h.get('his','?')} | score:{h.get('anomaly_score','?')} | "
                   f"temp:{h.get('body_temp','?')}°C | estado:{h.get('estado','?')}"
                   for h in ultimos]
        historial_str = f"Historial reciente (últimas {len(ultimos)} evaluaciones):\n" + "\n".join(rows)
    else:
        historial_str = "Historial: sin datos previos"

    return f"""Analiza el estado de salud de esta vaca:

Score de anomalía (Isolation Forest): {anomaly_score}
HIS actual: {his}/100

Datos de sensores (ventana de 5 minutos):
  Acelerómetro:
    - Actividad media:      {features.get('mean_accel', 'N/A')} m/s²
    - Variabilidad:         {features.get('std_accel', 'N/A')}
    - Tiempo acostada:      {round(features.get('lying_ratio', 0) * 100, 1)}%
    - Tendencia temperatura:{features.get('temp_trend', 'N/A')} °C/min
  Temperatura corporal:     {features.get('body_temp', 'N/A')}°C
  Frecuencia cardíaca:      {features.get('heart_rate_mean', 'N/A')} bpm (std: {features.get('heart_rate_std', 'N/A')})
  Frecuencia respiratoria:  {features.get('respiratory_rate', 'N/A')} rpm
  Rumia:                    {features.get('rumination_min', 'N/A')} min/día
  Hidratación:              {features.get('hydration_freq', 'N/A')} veces/día
  Ambiente:
    - Humedad:              {features.get('humidity', 'N/A')}%
    - Temperatura ambiente: {features.get('ambient_temp', 'N/A')}°C
    - THI:                  {features.get('thi_score', 'N/A')}

{alertas_str}

{historial_str}

Basándote en estos datos, identifica la condición más probable y la acción recomendada."""


def build_chat_prompt(mensaje: str, contexto_herd: dict) -> str:
    """
    Construye el prompt para el chat libre con el ganadero.
    """
    vacas_alerta = contexto_herd.get("vacas_alerta", 0)
    vacas_total  = contexto_herd.get("total_vacas", 0)
    his_promedio = contexto_herd.get("his_promedio", 0)

    return f"""El ganadero tiene {vacas_total} vacas en monitoreo.
Estado del herd: {vacas_alerta} en alerta | HIS promedio: {his_promedio}/100

Pregunta del ganadero: {mensaje}

Responde de forma clara y en lenguaje simple, como si hablaras con un ganadero práctico.
Si la pregunta es sobre una vaca específica, basa tu respuesta en los datos disponibles.
Si no tienes suficiente información, dilo claramente.
Cuando respondas preguntas del ganadero, hazlo en texto plano sin markdown, 
sin asteriscos, sin backticks. Responde de forma directa y práctica."""