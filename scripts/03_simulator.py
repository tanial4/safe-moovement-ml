"""
scripts/03_simulator.py
Simula lecturas de sensores y las envía al backend cada N segundos.

Uso básico:
    python scripts/03_simulator.py

Demo controlada (para la presentación):
    python scripts/03_simulator.py --inject-sick 3 --escape-at 7
"""
import argparse
import time
import random
import math
import json
from datetime import datetime

try:
    import httpx
    _client_class = httpx.Client
except ImportError:
    import urllib.request
    _client_class = None

# ── Distribuciones basadas en MmCows ────────────────────────────────────────

NORMAL = {
    "body_temp":   (38.6, 0.3,  37.5, 39.4),
    "accel_mean":  (0.45, 0.15, 0.05, 1.20),
}

SICK = {
    "body_temp":   (40.1, 0.2,  39.6, 41.5),
    "accel_mean":  (0.10, 0.05, 0.02, 0.25),
}

BASE_LAT      = 4.7110
BASE_LNG      = -74.0721
FENCE_RADIUS  = 0.002


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def sample(mean, std, lo, hi):
    return clamp(random.gauss(mean, std), lo, hi)


def accel_components(magnitude):
    a1  = random.uniform(0, 2 * math.pi)
    a2  = random.uniform(0, math.pi)
    mag = abs(magnitude + random.gauss(0, 0.02))
    return (
        mag * math.sin(a2) * math.cos(a1),
        mag * math.sin(a2) * math.sin(a1),
        mag * math.cos(a2),
    )


class Cow:
    def __init__(self, cow_id, sick=False):
        self.id       = cow_id
        self.sick     = sick
        self.escaped  = False
        angle         = random.uniform(0, 2 * math.pi)
        r             = random.uniform(0, FENCE_RADIUS * 0.7)
        self.lat      = BASE_LAT + r * math.cos(angle)
        self.lng      = BASE_LNG + r * math.sin(angle)

    def reading(self):
        dist      = SICK if self.sick else NORMAL
        self.lat += random.gauss(0, 0.00003)
        self.lng += random.gauss(0, 0.00003)
        mean_mag  = sample(*dist["accel_mean"])
        ax, ay, az = accel_components(mean_mag)
        return {
            "cow_id":    self.id,
            "timestamp": time.time(),
            "lat":       round(self.lat, 6),
            "lng":       round(self.lng, 6),
            "accel_x":   round(ax, 4),
            "accel_y":   round(ay, 4),
            "accel_z":   round(az, 4),
            "body_temp": round(sample(*dist["body_temp"]), 2),
        }

    def escape(self):
        angle    = random.uniform(0, 2 * math.pi)
        dist     = FENCE_RADIUS * random.uniform(1.5, 2.5)
        self.lat = BASE_LAT + dist * math.cos(angle)
        self.lng = BASE_LNG + dist * math.sin(angle)
        self.escaped = True
        print(f"\n  [{self.id}] ESCAPE → ({self.lat:.5f}, {self.lng:.5f})")


def send(reading, backend):
    data = json.dumps(reading).encode()
    try:
        if _client_class:
            with httpx.Client() as c:
                r = c.post(f"{backend}/ingest", content=data,
                           headers={"Content-Type": "application/json"}, timeout=3)
                return r.status_code == 200
        else:
            req = urllib.request.Request(
                f"{backend}/ingest", data=data,
                headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=3):
                return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cows",      type=int,   default=8)
    parser.add_argument("--interval",    type=float, default=5.0)
    parser.add_argument("--backend",     type=str,   default="http://localhost:8000")
    parser.add_argument("--inject-sick", type=int,   default=None,
                        help="Número de vaca enferma desde el inicio (ej: 3 → C03)")
    parser.add_argument("--escape-at",   type=int,   default=None,
                        help="Número de vaca que escapa a los 60s (ej: 7 → C07)")
    args = parser.parse_args()

    cows = []
    for i in range(1, args.n_cows + 1):
        sick = (i == args.inject_sick)
        cows.append(Cow(f"C{i:02d}", sick=sick))
        if sick:
            print(f"  [C{i:02d}] arranca ENFERMA")

    print(f"\nSimulador: {args.n_cows} vacas → {args.backend}")
    print(f"Intervalo: {args.interval}s  |  Ctrl+C para detener\n")

    start = time.time()
    i     = 0

    while True:
        i      += 1
        elapsed = time.time() - start
        ok      = 0

        if args.escape_at and elapsed > 60 and not cows[args.escape_at - 1].escaped:
            cows[args.escape_at - 1].escape()

        for cow in cows:
            if send(cow.reading(), args.backend):
                ok += 1

        ts     = datetime.now().strftime("%H:%M:%S")
        status = "OK" if ok == len(cows) else f"WARN {ok}/{len(cows)}"
        print(f"[{ts}] iter {i:04d} | {status}        ", end="\r")

        if ok == 0 and i == 1:
            print(f"\nAVISO: el backend en {args.backend} no responde.")
            print("Inicia el backend de Dev 2 primero.\n")

        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSimulador detenido.")