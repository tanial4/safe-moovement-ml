"""
src/features.py
"""
import numpy as np
from typing import Optional

LYING_THRESH   = 0.3
WINDOW_SECONDS = 300


def accel_magnitude(ax: float, ay: float, az: float) -> float:
    return float(np.sqrt(ax**2 + ay**2 + az**2))


def build_features_from_readings(readings: list, cow_id: str) -> Optional[dict]:
    """
    readings: lista de dicts con keys:
        timestamp, accel_x, accel_y, accel_z, body_temp
    Devuelve dict con SensorFeatures o None si no hay suficientes datos.
    """
    if len(readings) < 5:
        return None

    readings = sorted(readings, key=lambda r: r["timestamp"])

    mags  = [accel_magnitude(r["accel_x"], r["accel_y"], r["accel_z"]) for r in readings]
    temps = [r["body_temp"] for r in readings]
    diffs = [temps[i] - temps[i - 1] for i in range(1, len(temps))]

    return {
        "cow_id":       cow_id,
        "window_start": readings[0]["timestamp"],
        "window_end":   readings[-1]["timestamp"],
        "mean_accel":   float(np.mean(mags)),
        "std_accel":    float(np.std(mags)),
        "body_temp":    float(np.mean(temps)),
        "lying_ratio":  float(np.mean([1 if m < LYING_THRESH else 0 for m in mags])),
        "temp_trend":   float(np.mean(diffs)) if diffs else 0.0,
    }