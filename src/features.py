"""
src/features.py — v2
Construye SensorFeatures a partir de lecturas crudas.
"""
import numpy as np
from typing import Optional

LYING_THRESH = 0.3
GRAVITY      = 9.81


def accel_magnitude(ax, ay, az):
    return float(np.abs(np.sqrt(ax**2 + ay**2 + az**2) - GRAVITY))


def build_features_from_readings(readings: list, cow_id: str) -> Optional[dict]:
    """
    readings: lista de dicts con keys:
        timestamp, accel_x, accel_y, accel_z, body_temp
        + opcionales: heart_rate, respiratory_rate,
                      rumination_min, hydration_freq,
                      humidity, ambient_temp, thi_score,
                      elevation, lying_bin
    """
    if len(readings) < 5:
        return None

    readings = sorted(readings, key=lambda r: r["timestamp"])

    mags  = [accel_magnitude(r["accel_x"], r["accel_y"], r["accel_z"]) for r in readings]
    temps = [r["body_temp"] for r in readings]
    diffs = [temps[i] - temps[i - 1] for i in range(1, len(temps))]

    # lying_ratio: usar lying_bin si disponible
    lying_bins = [r.get("lying_bin") for r in readings if r.get("lying_bin") is not None]
    if len(lying_bins) > 3:
        lying_ratio = float(np.mean(lying_bins))
    else:
        lying_ratio = float(np.mean([1 if m < LYING_THRESH else 0 for m in mags]))

    # elevation_std
    elevations = [r.get("elevation") for r in readings if r.get("elevation") is not None]
    elev_std   = float(np.std(elevations)) if len(elevations) > 3 else None

    def avg(key):
        vals = [r.get(key) for r in readings if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "cow_id":       cow_id,
        "window_start": readings[0]["timestamp"],
        "window_end":   readings[-1]["timestamp"],

        # Acelerómetro
        "mean_accel":   float(np.mean(mags)),
        "std_accel":    float(np.std(mags)),
        "lying_ratio":  lying_ratio,
        "temp_trend":   float(np.mean(diffs)) if diffs else 0.0,

        # Temperatura corporal
        "body_temp":    float(np.mean(temps)),

        # Ambiente
        "humidity":     avg("humidity"),
        "ambient_temp": avg("ambient_temp"),
        "thi_score":    avg("thi_score"),

        # Elevación
        "elevation_std": elev_std,

        # Fisiológicos (simulados o de sensor real futuro)
        "heart_rate_mean":  avg("heart_rate"),
        "heart_rate_std":   float(np.std([r.get("heart_rate") for r in readings
                                          if r.get("heart_rate") is not None]))
                            if len([r for r in readings if r.get("heart_rate")]) > 1 else None,
        "respiratory_rate": avg("respiratory_rate"),
        "rumination_min":   avg("rumination_min"),
        "hydration_freq":   avg("hydration_freq"),
    }