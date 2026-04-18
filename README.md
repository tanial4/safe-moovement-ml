# Safe Moovement — ML & AI Service

Sistema de monitoreo inteligente de ganado bovino. Este repo contiene el módulo de Machine Learning y el agente de IA conectado a Gemini.

## Servicios

| Servicio | Puerto | Descripción |
|---|---|---|
| ML Service | 8001 | Isolation Forest + reglas clínicas |
| AI Service | 8002 | Agente Gemini — diagnóstico clínico |
| Backend | 8000 | FastAPI (repo separado, Dev 2) |
| Frontend | 5173 | React + Leaflet (repo separado, Dev 1) |

---

## Requisitos

- Python 3.11+
- Cuenta en Kaggle (para descargar el dataset)
- API Key de Gemini → https://aistudio.google.com/app/apikey

---

## Setup inicial

```bash
# 1. Clonar el repo y crear entorno virtual
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus keys
```

### .env
```
KAGGLE_API_TOKEN=tu_token_kaggle
GEMINI_API_KEY=tu_key_gemini
```

---

## Dataset — MmCows

Dataset real de 10 vacas Holstein lecheras con 14 días de sensores wearables.
Paper: "MmCows: A Multimodal Dataset for Dairy Cattle Monitoring" (NeurIPS 2024)

### Descargar

```bash
# Configurar Kaggle API token en .env primero, luego:
export $(cat .env | xargs)
kaggle datasets download hienvuvg/mmcows -p data/mmcows_raw --unzip
```

O descarga manualmente desde https://www.kaggle.com/datasets/hienvuvg/mmcows
y extrae en `data/mmcows_raw/`.

### Estructura esperada después de descargar

```
data/mmcows_raw/sensor_data/sensor_data/main_data/
    immu/T01/T01_0721.csv ...   ← acelerómetro (por día)
    cbt/C01.csv ...              ← temperatura corporal
    ankle/C01/...                ← comportamiento (lying)
    pressure/T01/...             ← elevación
    thi/average.csv              ← temperatura y humedad ambiente
```

---

## Pipeline ML

### 1. Limpiar datos

```bash
python scripts/01_clean_mmcows.py
```

Genera `data/clean/mmcows_clean.csv` con 39,581 ventanas de 5 minutos.

### 2. Entrenar modelo

```bash
python scripts/02_train.py
```

Genera `models/ganaderIA_model.pkl` — Isolation Forest con 14 features.

---

## Levantar servicios

```bash
# ML Service (terminal 1)
uvicorn api:app --port 8001 --reload

# AI Service (terminal 2)
uvicorn ai_api:app --port 8002 --reload

# Simulador (terminal 3)
python scripts/03_simulator.py --direct-ml --backend http://localhost:8001 --inject-sick 3 --n-cows 8
```

---

## Endpoints

### ML Service — puerto 8001

#### GET /health
```bash
curl http://localhost:8001/health
```

#### POST /score/raw
Recibe lecturas crudas del sensor y devuelve HIS + alertas.

```json
// Request
{
  "cow_id": "C01",
  "readings": [
    {
      "timestamp":        1700000000.0,
      "accel_x":          0.12,
      "accel_y":         -0.05,
      "accel_z":          9.78,
      "body_temp":        38.5,
      "heart_rate":       64.0,
      "respiratory_rate": 28.0,
      "rumination_min":   420.0,
      "hydration_freq":   9.5,
      "humidity":         55.0,
      "ambient_temp":     22.0,
      "thi_score":        70.0,
      "elevation":        276.12,
      "lying_bin":        0
    }
    // mínimo 5 lecturas
  ]
}

// Response
{
  "cow_id":        "C01",
  "timestamp":     1700000305.12,
  "anomaly_score": 0.173,
  "his":           93,
  "estado":        "saludable",
  "alerta":        false,
  "alertas_clinicas": []
}
```

#### GET /thresholds
Devuelve todos los umbrales clínicos configurados.

---

### AI Service — puerto 8002

#### POST /analyze
Solo llamar cuando `alerta: true` del ML service.

```json
// Request
{
  "cow_id":           "C03",
  "features":         { /* features calculadas por el ML */ },
  "anomaly_score":    0.84,
  "his":              28,
  "alertas_clinicas": [
    {"tipo": "fiebre_alta", "valor": 40.2, "severidad": "critical"}
  ]
}

// Response
{
  "cow_id":             "C03",
  "his":                28,
  "estado":             "critico",
  "condicion_probable": "Mastitis aguda o metritis postparto",
  "confianza":          "alta",
  "accion_recomendada": "Revisión veterinaria inmediata",
  "justificacion":      "Fiebre sostenida + inactividad extrema...",
  "source":             "gemini"
}
```

#### POST /chat
Chat libre del ganadero en lenguaje natural.

```json
// Request
{
  "mensaje":        "La vaca C03 lleva dos días con fiebre, qué hago?",
  "historial_chat": [],
  "contexto_herd":  {"total_vacas": 8, "vacas_alerta": 1, "his_promedio": 87}
}

// Response
{
  "respuesta":  "Ganadero, dado que la vaca C03...",
  "timestamp":  1700000305.12
}
```

#### GET /historial/{cow_id}
Historial de evaluaciones de una vaca (en memoria, se limpia al reiniciar).

#### DELETE /historial/{cow_id}
Limpia el historial — útil para resetear la demo.

---

## Features del modelo

### Reales (de MmCows)

| Feature | Fuente | Descripción |
|---|---|---|
| mean_accel | immu | Media de aceleración dinámica (gravity removida) |
| std_accel | immu | Variabilidad del movimiento |
| lying_ratio | ankle | Proporción del tiempo acostada (0–1) |
| temp_trend | cbt | Tendencia de temperatura °C/min |
| body_temp | cbt | Temperatura corporal media °C |
| humidity | thi | Humedad ambiental % |
| ambient_temp | thi | Temperatura ambiental °C |
| thi_score | thi | Índice de estrés térmico |
| elevation_std | pressure | Variabilidad de elevación (movimiento vertical) |

### Simuladas (distribuciones de literatura veterinaria)

| Feature | Normal | Alerta |
|---|---|---|
| heart_rate_mean | 48–84 bpm | < 48 o > 84 |
| heart_rate_std | < 15 | > 15 → arritmia posible |
| respiratory_rate | 18–44 rpm | < 18 o > 44 |
| rumination_min | 400–600 min/día | < 280 |
| hydration_freq | 7–12 veces/día | < 7 |

### Umbrales de temperatura

| Rango | Estado |
|---|---|
| < 38.0°C | Hipotermia |
| 38.0–39.3°C | Normal |
| 39.4–39.9°C | Fiebre leve |
| > 39.9°C | Fiebre alta |

---

## HIS — Health Index Score

Score de 0 a 100. Alerta cuando baja de 86.

```
HIS >= 86  → saludable
HIS 60–85  → alerta
HIS < 60   → crítico
```

Calculado combinando:
- Penalización por anomaly_score del Isolation Forest (hasta -40 pts)
- Penalizaciones por umbrales clínicos duros (fiebre, taquicardia, etc.)
- Penalización por comportamiento postural (lying_ratio, inactividad)

---

## Simulador

```bash
# Demo completa con vaca enferma y escape
python scripts/03_simulator.py \
    --backend http://localhost:8000 \
    --inject-sick 3 \
    --escape-at 7 \
    --n-cows 8

# Probar directo contra ML sin backend
python scripts/03_simulator.py \
    --direct-ml \
    --backend http://localhost:8001 \
    --inject-sick 3 \
    --n-cows 5

# Opciones disponibles
--n-cows       Número de vacas (default: 8)
--interval     Segundos entre lecturas (default: 5.0)
--buffer-size  Lecturas antes de evaluar (default: 5)
--backend      URL del servicio destino
--inject-sick  Número de vaca enferma desde el inicio
--escape-at    Número de vaca que escapa a los 60s
--direct-ml    Conectar directo al ML service
```

---

## Estructura del proyecto

```
ganaderIA-ml/
├── api.py                  ← ML Service (puerto 8001)
├── ai_api.py               ← AI Service Gemini (puerto 8002)
├── requirements.txt
├── .env                    ← no subir a git
├── .env.example
├── .gitignore
├── data/
│   ├── mmcows_raw/         ← no subir a git
│   └── clean/              ← no subir a git
├── models/
│   ├── ganaderIA_model.pkl ← no subir a git
│   └── model_meta.json     ← no subir a git
├── scripts/
│   ├── 01_clean_mmcows.py
│   ├── 02_train.py
│   └── 03_simulator.py
└── src/
    ├── __init__.py
    ├── features.py         ← build_features_from_readings()
    ├── scorer.py           ← get_anomaly_score() + calcular_his()
    ├── gemini_client.py    ← analyze() + chat()
    └── prompts.py          ← system prompt + builders de contexto
```

---

## Tecnologías

- **scikit-learn** — Isolation Forest + StandardScaler
- **pandas / numpy** — limpieza y transformación de datos
- **FastAPI + uvicorn** — microservicios REST
- **google-generativeai** — SDK oficial de Gemini
- **joblib** — serialización del modelo
- **httpx** — cliente HTTP del simulador
- **python-dotenv** — carga de variables de entorno
