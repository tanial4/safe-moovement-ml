"""
scripts/01_clean_mmcows.py
Adaptado a la estructura real de MmCows:
  immu/T01/T01_0721.csv, T01_0722.csv ...  (un CSV por día)
  cbt/C01.csv, C02.csv ...                 (un CSV por vaca)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

RAW            = Path("data/mmcows_raw/sensor_data/sensor_data/main_data")
OUT            = Path("data/clean")
WINDOW_SECONDS = 300
MIN_READINGS   = 10
LYING_THRESH   = 0.3


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


def load_accel(cow_id):
    """
    Carga y concatena todos los CSVs diarios de acelerómetro de una vaca.
    Ruta: immu/T{cow_id}/*.csv
    """
    tag      = f"T{cow_id:02d}"
    cow_dir  = RAW / "immu" / tag
    csv_files = sorted(cow_dir.glob("*.csv")) if cow_dir.exists() else []

    if not csv_files:
        return None

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return None

    merged = pd.concat(dfs, ignore_index=True)
    merged.columns = merged.columns.str.lower().str.strip()
    return merged


def load_temp(cow_id):
    """
    Carga el CSV de temperatura corporal.
    Ruta: cbt/C{cow_id}.csv  (usa C no T)
    """
    tag  = f"C{cow_id:02d}"
    path = RAW / "cbt" / f"{tag}.csv"

    if not path.exists():
        return None

    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.lower().str.strip()
    return df


def load_cow(cow_id):
    accel = load_accel(cow_id)
    temp  = load_temp(cow_id)

    if accel is None or temp is None:
        return None

    try:
        ts_a = find_col(accel, ["time", "ts", "timestamp"])
        ax   = find_col(accel, ["ax", "accel_x", "_x", " x"])
        ay   = find_col(accel, ["ay", "accel_y", "_y", " y"])
        az   = find_col(accel, ["az", "accel_z", "_z", " z"])
    except ValueError as e:
        print(f"  T{cow_id:02d} accel: {e}")
        return None

    try:
        ts_t = find_col(temp, ["time", "ts", "timestamp"])
        tc   = find_col(temp, ["temp", "cbt", "body", "temperature"])
    except ValueError as e:
        print(f"  C{cow_id:02d} temp: {e}")
        return None

    accel = (accel
             .rename(columns={ts_a: "ts", ax: "ax", ay: "ay", az: "az"})
             [["ts", "ax", "ay", "az"]]
             .dropna()
             .sort_values("ts")
             .reset_index(drop=True))

    temp = (temp
            .rename(columns={ts_t: "ts", tc: "temp"})
            [["ts", "temp"]]
            .dropna()
            .sort_values("ts")
            .reset_index(drop=True))

    # Convertir timestamps a numérico por si vienen como string
    accel["ts"] = pd.to_numeric(accel["ts"], errors="coerce")
    temp["ts"]  = pd.to_numeric(temp["ts"],  errors="coerce")
    accel = accel.dropna(subset=["ts"])
    temp  = temp.dropna(subset=["ts"])

    merged = pd.merge_asof(
        accel, temp,
        on="ts",
        tolerance=30,
        direction="nearest"
    ).dropna()

    if len(merged) < MIN_READINGS:
        return None

    merged["accel_mag"] = np.sqrt(
        merged["ax"]**2 + merged["ay"]**2 + merged["az"]**2
    )
    merged["accel_mag"] = np.abs(merged["accel_mag"] - 9.81)
    merged["cow_id"] = cow_id
    return merged


def build_windows(df, cow_id):
    windows = []
    df      = df.copy()
    df["wid"] = (df["ts"] // WINDOW_SECONDS).astype(int)

    for wid, w in df.groupby("wid"):
        if len(w) < MIN_READINGS:
            continue

        diffs = w["temp"].diff().dropna()

        windows.append({
            "cow_id":       cow_id,
            "window_id":    int(wid),
            "window_start": float(w["ts"].min()),
            "window_end":   float(w["ts"].max()),
            "mean_accel":   float(w["accel_mag"].mean()),
            "std_accel":    float(w["accel_mag"].std()),
            "body_temp":    float(w["temp"].mean()),
            "lying_ratio": float((w["accel_mag"] < 0.3).mean()),
            "temp_trend":   float(diffs.mean()) if len(diffs) > 0 else 0.0,
        })

    return windows


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    if not RAW.exists():
        print(f"ERROR: No se encontró {RAW}")
        sys.exit(1)

    print("=== 01_clean_mmcows.py ===\n")

    # Mostrar estructura detectada
    immu_dir = RAW / "immu"
    cbt_dir  = RAW / "cbt"
    print(f"Acelerómetro: {immu_dir}")
    print(f"Temperatura:  {cbt_dir}\n")

    all_windows = []

    for cow_id in range(1, 11):
        try:
            df = load_cow(cow_id)
        except Exception as e:
            print(f"  T{cow_id:02d}: ERROR — {e}")
            continue

        if df is None:
            print(f"  T{cow_id:02d}: no disponible, skip")
            continue

        windows = build_windows(df, cow_id)
        all_windows.extend(windows)
        print(f"  T{cow_id:02d}: {len(df):>7} lecturas → {len(windows):>4} ventanas")

    if not all_windows:
        print("\nERROR: No se generaron ventanas.")
        print("\nRevisa las columnas del CSV con:")
        print("  python -c \"import pandas as pd; df=pd.read_csv('data/mmcows_raw/sensor_data/sensor_data/main_data/immu/T01/T01_0721.csv'); print(df.columns.tolist()); print(df.head(2))\"")
        sys.exit(1)

    result   = pd.DataFrame(all_windows).dropna()
    out_path = OUT / "mmcows_clean.csv"
    result.to_csv(out_path, index=False)

    print(f"\nTotal: {len(result)} ventanas · {result['cow_id'].nunique()} vacas")
    print(f"Guardado en: {out_path}\n")
    print(result[["mean_accel","std_accel","body_temp","lying_ratio","temp_trend"]].describe().round(3))
    print("\nListo. Ejecuta: python scripts/02_train.py")


if __name__ == "__main__":
    main()