import pandas as pd
import numpy as np
from pathlib import Path
import sys

RAW            = Path("data/mmcows_raw/sensor_data/sensor_data/main_data")
OUT            = Path("data/clean")
WINDOW_SECONDS = 300   # ventana de 5 minutos
MIN_READINGS   = 10
LYING_THRESH   = 0.3
GRAVITY        = 9.81


# ── Distribuciones de literatura veterinaria para features simuladas ─────────
# Se usan para generar datos realistas en el CSV de entrenamiento.
# El simulador (03_simulator.py) usa las mismas distribuciones.

SIM_NORMAL = {
    "heart_rate":      {"mean": 65,   "std": 8,    "min": 48,  "max": 84},
    "respiratory_rate":{"mean": 28,   "std": 5,    "min": 18,  "max": 44},
    "rumination_min":  {"mean": 420,  "std": 60,   "min": 200, "max": 600},
    "hydration_freq":  {"mean": 9.5,  "std": 1.5,  "min": 7,   "max": 12},
}

SIM_SICK = {
    "heart_rate":      {"mean": 90,   "std": 10,   "min": 75,  "max": 120},
    "respiratory_rate":{"mean": 52,   "std": 8,    "min": 36,  "max": 80},
    "rumination_min":  {"mean": 150,  "std": 60,   "min": 0,   "max": 280},
    "hydration_freq":  {"mean": 4,    "std": 1.5,  "min": 0,   "max": 6},
}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def sim_feature(dist, n=1):
    vals = [clamp(np.random.normal(dist["mean"], dist["std"]),
                  dist["min"], dist["max"]) for _ in range(n)]
    return vals[0] if n == 1 else vals


def find_col(df, candidates):
    cols_lower = [c.lower().strip() for c in df.columns]
    for cand in candidates:
        for i, col in enumerate(cols_lower):
            if cand in col:
                return df.columns[i]
    raise ValueError(
        f"No se encontró columna para {candidates}.\n"
        f"Columnas disponibles: {list(df.columns)}"
    )


# ── Loaders por fuente ───────────────────────────────────────────────────────

def load_accel(cow_id):
    tag     = f"T{cow_id:02d}"
    cow_dir = RAW / "immu" / tag
    files   = sorted(cow_dir.glob("*.csv")) if cow_dir.exists() else []
    if not files:
        return None
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, low_memory=False))
        except Exception:
            continue
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.lower().str.strip()
    return df


def load_temp(cow_id):
    tag  = f"C{cow_id:02d}"
    path = RAW / "cbt" / f"{tag}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.lower().str.strip()
    return df


def load_ankle(cow_id):
    tag     = f"C{cow_id:02d}"
    cow_dir = RAW / "ankle" / tag
    files   = sorted(cow_dir.glob("*.csv")) if cow_dir.exists() else []
    if not files:
        return None
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, low_memory=False))
        except Exception:
            continue
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.lower().str.strip()
    return df


def load_pressure(cow_id):
    tag     = f"T{cow_id:02d}"
    cow_dir = RAW / "pressure" / tag
    files   = sorted(cow_dir.glob("*.csv")) if cow_dir.exists() else []
    if not files:
        return None
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, low_memory=False))
        except Exception:
            continue
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.lower().str.strip()
    return df


def load_thi():
    path = RAW / "thi" / "average.csv"
    if not path.exists():
        # intentar S01
        path = RAW / "thi" / "S01.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.lower().str.strip()
    return df


# ── Merge y limpieza por vaca ────────────────────────────────────────────────

def load_cow(cow_id, thi_df):
    # Acelerómetro — obligatorio
    accel = load_accel(cow_id)
    if accel is None:
        return None

    try:
        ts_a = find_col(accel, ["time", "ts", "timestamp"])
        ax   = find_col(accel, ["ax", "accel_x", "_x", " x"])
        ay   = find_col(accel, ["ay", "accel_y", "_y", " y"])
        az   = find_col(accel, ["az", "accel_z", "_z", " z"])
    except ValueError as e:
        print(f"  T{cow_id:02d} accel cols: {e}")
        return None

    accel = (accel
             .rename(columns={ts_a: "ts", ax: "ax", ay: "ay", az: "az"})
             [["ts", "ax", "ay", "az"]]
             .dropna().sort_values("ts").reset_index(drop=True))
    accel["ts"] = pd.to_numeric(accel["ts"], errors="coerce")
    accel = accel.dropna(subset=["ts"])
    accel["accel_mag"] = np.abs(
        np.sqrt(accel["ax"]**2 + accel["ay"]**2 + accel["az"]**2) - GRAVITY
    )

    # Temperatura — obligatorio
    temp = load_temp(cow_id)
    if temp is None:
        return None
    try:
        ts_t = find_col(temp, ["time", "ts", "timestamp"])
        tc   = find_col(temp, ["temp", "cbt", "body", "temperature"])
    except ValueError as e:
        print(f"  C{cow_id:02d} temp cols: {e}")
        return None
    temp = (temp
            .rename(columns={ts_t: "ts", tc: "temp"})
            [["ts", "temp"]]
            .dropna().sort_values("ts").reset_index(drop=True))
    temp["ts"] = pd.to_numeric(temp["ts"], errors="coerce")
    temp = temp.dropna(subset=["ts"])

    # Merge accel + temp
    merged = pd.merge_asof(accel, temp, on="ts", tolerance=30,
                           direction="nearest").dropna()
    if len(merged) < MIN_READINGS:
        return None

    # Ankle (lying binario) — opcional pero mejor que lying_ratio calculado
    ankle = load_ankle(cow_id)
    if ankle is not None:
        try:
            ts_ank  = find_col(ankle, ["time", "ts", "timestamp"])
            ly      = find_col(ankle, ["lying"])
            ankle   = (ankle
                       .rename(columns={ts_ank: "ts", ly: "lying_bin"})
                       [["ts", "lying_bin"]]
                       .dropna().sort_values("ts").reset_index(drop=True))
            ankle["ts"] = pd.to_numeric(ankle["ts"], errors="coerce")
            ankle = ankle.dropna(subset=["ts"])
            merged = pd.merge_asof(merged, ankle, on="ts", tolerance=60,
                                   direction="nearest")
        except ValueError:
            merged["lying_bin"] = np.nan

    # Pressure (elevación) — opcional
    pressure = load_pressure(cow_id)
    if pressure is not None:
        try:
            ts_p  = find_col(pressure, ["time", "ts", "timestamp"])
            elev  = find_col(pressure, ["elevation", "elev"])
            pressure = (pressure
                        .rename(columns={ts_p: "ts", elev: "elevation"})
                        [["ts", "elevation"]]
                        .dropna().sort_values("ts").reset_index(drop=True))
            pressure["ts"] = pd.to_numeric(pressure["ts"], errors="coerce")
            pressure = pressure.dropna(subset=["ts"])
            merged = pd.merge_asof(merged, pressure, on="ts", tolerance=30,
                                   direction="nearest")
        except ValueError:
            merged["elevation"] = np.nan

    # THI ambiental — opcional
    if thi_df is not None:
        try:
            ts_thi  = find_col(thi_df, ["time", "ts", "timestamp"])
            hum     = find_col(thi_df, ["humidity", "hum"])
            atmp    = find_col(thi_df, ["temperature", "temp"])
            thi_col = find_col(thi_df, ["thi"])
            thi_sub = (thi_df
                       .rename(columns={ts_thi: "ts", hum: "humidity",
                                        atmp: "ambient_temp", thi_col: "thi"})
                       [["ts", "humidity", "ambient_temp", "thi"]]
                       .dropna().sort_values("ts").reset_index(drop=True))
            thi_sub["ts"] = pd.to_numeric(thi_sub["ts"], errors="coerce")
            thi_sub = thi_sub.dropna(subset=["ts"])
            merged = pd.merge_asof(merged, thi_sub, on="ts", tolerance=300,
                                   direction="nearest")
        except ValueError:
            merged["humidity"]     = np.nan
            merged["ambient_temp"] = np.nan
            merged["thi"]          = np.nan

    merged["cow_id"] = cow_id
    return merged


# ── Build windows ────────────────────────────────────────────────────────────

def build_windows(df, cow_id):
    windows = []
    df      = df.copy()
    df["wid"] = (df["ts"] // WINDOW_SECONDS).astype(int)

    for wid, w in df.groupby("wid"):
        if len(w) < MIN_READINGS:
            continue

        temp_diffs = w["temp"].diff().dropna()

        # lying_ratio: usar ankle si disponible, sino accel
        if "lying_bin" in w.columns and w["lying_bin"].notna().sum() > 3:
            lying_ratio = float(w["lying_bin"].mean())
        else:
            lying_ratio = float((w["accel_mag"] < LYING_THRESH).mean())

        # elevation_std: variabilidad vertical (movimiento activo)
        elev_std = float(w["elevation"].std()) if "elevation" in w.columns and w["elevation"].notna().sum() > 3 else np.nan

        # THI y ambiente
        humidity     = float(w["humidity"].mean())     if "humidity"     in w.columns and w["humidity"].notna().sum() > 0     else np.nan
        ambient_temp = float(w["ambient_temp"].mean()) if "ambient_temp" in w.columns and w["ambient_temp"].notna().sum() > 0 else np.nan
        thi_score    = float(w["thi"].mean())          if "thi"          in w.columns and w["thi"].notna().sum() > 0          else np.nan

        # Features simuladas — distribuciones de literatura veterinaria
        heart_rate_mean  = sim_feature(SIM_NORMAL["heart_rate"])
        heart_rate_std   = abs(np.random.normal(5, 2))
        respiratory_rate = sim_feature(SIM_NORMAL["respiratory_rate"])
        rumination_min   = sim_feature(SIM_NORMAL["rumination_min"])
        hydration_freq   = sim_feature(SIM_NORMAL["hydration_freq"])

        windows.append({
            # Identificadores
            "cow_id":       cow_id,
            "window_id":    int(wid),
            "window_start": float(w["ts"].min()),
            "window_end":   float(w["ts"].max()),

            # Features reales — acelerómetro
            "mean_accel":   float(w["accel_mag"].mean()),
            "std_accel":    float(w["accel_mag"].std()),
            "lying_ratio":  lying_ratio,
            "temp_trend":   float(temp_diffs.mean()) if len(temp_diffs) > 0 else 0.0,

            # Features reales — temperatura corporal
            "body_temp":    float(w["temp"].mean()),

            # Features reales — ambiente
            "humidity":     humidity,
            "ambient_temp": ambient_temp,
            "thi_score":    thi_score,

            # Feature real — elevación
            "elevation_std": elev_std,

            # Features simuladas — literatura veterinaria
            "heart_rate_mean":   round(heart_rate_mean, 1),
            "heart_rate_std":    round(heart_rate_std, 2),
            "respiratory_rate":  round(respiratory_rate, 1),
            "rumination_min":    round(rumination_min, 0),
            "hydration_freq":    round(hydration_freq, 1),
        })

    return windows


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    if not RAW.exists():
        print(f"ERROR: No se encontró {RAW}")
        sys.exit(1)

    print("=== 01_clean_mmcows.py (v2 — features expandidas) ===\n")

    print("Cargando THI ambiental...")
    thi_df = load_thi()
    print(f"  THI: {'OK' if thi_df is not None else 'no disponible'}\n")

    all_windows = []

    for cow_id in range(1, 11):
        try:
            df = load_cow(cow_id, thi_df)
        except Exception as e:
            print(f"  T{cow_id:02d}: ERROR — {e}")
            continue

        if df is None:
            print(f"  T{cow_id:02d}: no disponible, skip")
            continue

        windows = build_windows(df, cow_id)
        all_windows.extend(windows)

        has_ankle    = "lying_bin"  in df.columns and df["lying_bin"].notna().sum() > 0
        has_pressure = "elevation"  in df.columns and df["elevation"].notna().sum() > 0
        has_thi      = "humidity"   in df.columns and df["humidity"].notna().sum() > 0

        extras = []
        if has_ankle:    extras.append("ankle")
        if has_pressure: extras.append("pressure")
        if has_thi:      extras.append("thi")

        print(f"  T{cow_id:02d}: {len(df):>7} lecturas → {len(windows):>4} ventanas"
              f"  [{', '.join(extras) if extras else 'solo accel+temp'}]")

    if not all_windows:
        print("\nERROR: No se generaron ventanas.")
        sys.exit(1)

    result   = pd.DataFrame(all_windows)
    out_path = OUT / "mmcows_clean.csv"
    result.to_csv(out_path, index=False)

    real_features = ["mean_accel","std_accel","body_temp","lying_ratio",
                     "temp_trend","humidity","ambient_temp","thi_score","elevation_std"]
    sim_features  = ["heart_rate_mean","heart_rate_std",
                     "respiratory_rate","rumination_min","hydration_freq"]

    print(f"\nTotal: {len(result)} ventanas · {result['cow_id'].nunique()} vacas")
    print(f"Guardado en: {out_path}")
    print(f"\nFeatures reales ({len(real_features)}):")
    print(result[real_features].describe().round(3))
    print(f"\nFeatures simuladas ({len(sim_features)}):")
    print(result[sim_features].describe().round(3))
    print("\nListo. Ejecuta: python scripts/02_train.py")


if __name__ == "__main__":
    np.random.seed(42)
    main()