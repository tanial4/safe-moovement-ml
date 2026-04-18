from typing import Optional

from pydantic import BaseModel


class FeaturesIn(BaseModel):
    cow_id:        str
    window_start:  float
    window_end:    float
    # Acelerómetro
    mean_accel:    float
    std_accel:     float
    lying_ratio:   float
    temp_trend:    float
    # Temperatura corporal
    body_temp:     float
    # Ambiente — opcionales
    humidity:      Optional[float] = None
    ambient_temp:  Optional[float] = None
    thi_score:     Optional[float] = None
    elevation_std: Optional[float] = None
    # Fisiológicos — opcionales
    heart_rate_mean:  Optional[float] = None
    heart_rate_std:   Optional[float] = None
    respiratory_rate: Optional[float] = None
    rumination_min:   Optional[float] = None
    hydration_freq:   Optional[float] = None


class ScoreRequest(BaseModel):
    features:   FeaturesIn


class RawReadingsRequest(BaseModel):
    cow_id:   str
    readings: list