"""
scripts/03_simulator.py — v3
Acumula lecturas en buffer por vaca antes de mandar al modelo.
Buffer de 5 lecturas → equivale a 25s de datos antes de evaluar.

Uso:
    python scripts/03_simulator.py --direct-ml --backend http://localhost:8001 --inject-sick 3
    python scripts/03_simulator.py --backend http://localhost:8000 --inject-sick 3 --escape-at 7
"""
import argparse
import time
import random
import math
import json
from datetime import datetime

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    import urllib.request
    HAS_HTTPX = False

NORMAL = {
    "body_temp":        (38.6,  0.30,  37.5, 39.3),
    "accel_mean":       (0.45,  0.15,  0.05, 1.20),
    "heart_rate":       (65.0,  8.00,  48.0, 84.0),
    "respiratory_rate": (28.0,  5.00,  18.0, 44.0),
    "rumination_min":   (420.0, 60.0, 200.0, 600.0),
    "hydration_freq":   (9.5,   1.50,   7.0, 12.0),
    "humidity":         (55.0,  8.00,  30.0, 80.0),
    "ambient_temp":     (22.0,  3.00,  10.0, 35.0),
    "thi_score":        (70.0,  4.00,  55.0, 80.0),
    "elevation":        (276.0, 0.50, 274.0, 278.0),
}

SICK = {
    "body_temp":        (40.2,  0.30,  39.6, 41.5),
    "accel_mean":       (0.10,  0.05,  0.02, 0.25),
    "heart_rate":       (92.0, 10.00,  75.0, 120.0),
    "respiratory_rate": (52.0,  8.00,  36.0, 80.0),
    "rumination_min":   (120.0, 60.0,   0.0, 280.0),
    "hydration_freq":   (3.5,   1.50,   0.0,  6.0),
    "humidity":         (55.0,  8.00,  30.0, 80.0),
    "ambient_temp":     (22.0,  3.00,  10.0, 35.0),
    "thi_score":        (70.0,  4.00,  55.0, 80.0),
    "elevation":        (276.0, 0.50, 274.0, 278.0),
}

BASE_LAT     = 4.7110
BASE_LNG     = -74.0721
FENCE_RADIUS = 0.002
BUFFER_SIZE  = 5   # lecturas antes de evaluar


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
        self.id      = cow_id
        self.sick    = sick
        self.escaped = False
        angle        = random.uniform(0, 2 * math.pi)
        r            = random.uniform(0, FENCE_RADIUS * 0.7)
        self.lat     = BASE_LAT + r * math.cos(angle)
        self.lng     = BASE_LNG + r * math.sin(angle)
        self.elev    = sample(*NORMAL["elevation"])

    def reading(self):
        dist           = SICK if self.sick else NORMAL
        self.lat      += random.gauss(0, 0.00003)
        self.lng      += random.gauss(0, 0.00003)
        self.elev     += random.gauss(0, 0.05)
        mean_mag       = sample(*dist["accel_mean"])
        ax, ay, az     = accel_components(mean_mag)
        accel_mag      = abs(math.sqrt(ax**2 + ay**2 + az**2) - 9.81)

        return {
            "timestamp":        time.time(),
            "accel_x":          round(ax, 4),
            "accel_y":          round(ay, 4),
            "accel_z":          round(az, 4),
            "body_temp":        round(sample(*dist["body_temp"]), 2),
            "lying_bin":        1 if accel_mag < 0.3 else 0,
            "heart_rate":       round(sample(*dist["heart_rate"]), 1),
            "respiratory_rate": round(sample(*dist["respiratory_rate"]), 1),
            "rumination_min":   round(sample(*dist["rumination_min"]), 0),
            "hydration_freq":   round(sample(*dist["hydration_freq"]), 1),
            "humidity":         round(sample(*dist["humidity"]), 1),
            "ambient_temp":     round(sample(*dist["ambient_temp"]), 1),
            "thi_score":        round(sample(*dist["thi_score"]), 2),
            "elevation":        round(self.elev, 2),
            "lat":              round(self.lat, 6),
            "lng":              round(self.lng, 6),
        }

    def escape(self):
        angle    = random.uniform(0, 2 * math.pi)
        dist     = FENCE_RADIUS * random.uniform(1.5, 2.5)
        self.lat = BASE_LAT + dist * math.cos(angle)
        self.lng = BASE_LNG + dist * math.sin(angle)
        self.escaped = True
        print(f"\n  [{self.id}] ESCAPE → ({self.lat:.5f}, {self.lng:.5f})")


def send_payload(payload, backend, direct_ml):
    url  = f"{backend}/score/raw" if direct_ml else f"{backend}/ingest"
    data = json.dumps(payload).encode()
    try:
        if HAS_HTTPX:
            with httpx.Client() as c:
                r = c.post(url, content=data,
                           headers={"Content-Type": "application/json"},
                           timeout=5)
                if direct_ml and r.status_code == 200:
                    return True, r.json()
                return r.status_code == 200, None
        else:
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if direct_ml:
                    return True, json.loads(resp.read())
                return True, None
    except Exception:
        return False, None


def print_alert(response, cow_id):
    if not response:
        return
    his     = response.get("his", "?")
    estado  = response.get("estado", "?")
    alerta  = response.get("alerta", False)
    alertas = response.get("alertas_clinicas", [])
    score   = response.get("anomaly_score", "?")

    if alerta:
        tipos = [a["tipo"] for a in alertas]
        print(f"\n  [{cow_id}] ALERTA — HIS:{his} | score:{score} | {estado.upper()}"
              f"{' | ' + ', '.join(tipos) if tipos else ''}")
    else:
        pass  # vacas sanas no imprimen nada


def main():
    parser = argparse.ArgumentParser(description="GanaderIA sensor simulator v3")
    parser.add_argument("--n-cows",      type=int,   default=8)
    parser.add_argument("--interval",    type=float, default=5.0,
                        help="Segundos entre lecturas por vaca")
    parser.add_argument("--buffer-size", type=int,   default=BUFFER_SIZE,
                        help="Lecturas a acumular antes de evaluar (default: 5)")
    parser.add_argument("--backend",     type=str,   default="http://localhost:8000")
    parser.add_argument("--inject-sick", type=int,   default=None,
                        help="Número de vaca enferma desde el inicio (ej: 3 → C03)")
    parser.add_argument("--escape-at",   type=int,   default=None,
                        help="Número de vaca que escapa a los 60s (ej: 7 → C07)")
    parser.add_argument("--direct-ml",   action="store_true",
                        help="Conectar directo al ML service /score/raw (sin backend)")
    args = parser.parse_args()

    cows    = []
    buffers = {}

    for i in range(1, args.n_cows + 1):
        cow_id = f"C{i:02d}"
        sick   = (i == args.inject_sick)
        cows.append(Cow(cow_id, sick=sick))
        buffers[cow_id] = []
        if sick:
            print(f"  [{cow_id}] arranca ENFERMA")

    mode = "ML directo (/score/raw)" if args.direct_ml else "Backend (/ingest)"
    print(f"\nSimulador v3: {args.n_cows} vacas → {args.backend} [{mode}]")
    print(f"Buffer: {args.buffer_size} lecturas → evalúa cada {args.buffer_size * args.interval:.0f}s por vaca")
    print(f"Ctrl+C para detener\n")

    start      = time.time()
    iteration  = 0
    evaluations = 0

    while True:
        iteration += 1
        elapsed    = time.time() - start
        ok         = 0
        evaluated  = 0

        if args.escape_at and elapsed > 60 and not cows[args.escape_at - 1].escaped:
            cows[args.escape_at - 1].escape()

        for cow in cows:
            reading = cow.reading()

            if args.direct_ml:
                # Acumular en buffer
                buffers[cow.id].append(reading)

                if len(buffers[cow.id]) >= args.buffer_size:
                    payload = {
                        "cow_id":   cow.id,
                        "readings": buffers[cow.id]
                    }
                    success, response = send_payload(payload, args.backend, True)
                    buffers[cow.id]   = []  # resetear buffer

                    if success:
                        ok        += 1
                        evaluated += 1
                        evaluations += 1
                        print_alert(response, cow.id)
                    else:
                        ok += 0
                else:
                    ok += 1  # acumulando, no es error
            else:
                # Modo backend — manda lectura cruda a /ingest
                reading["cow_id"] = cow.id
                success, _        = send_payload(reading, args.backend, False)
                if success:
                    ok += 1

        ts     = datetime.now().strftime("%H:%M:%S")
        status = "OK" if ok == len(cows) else f"WARN {ok}/{len(cows)}"
        buf_status = f"buf:{len(buffers[cows[0].id])}/{args.buffer_size}" if args.direct_ml else ""
        print(f"[{ts}] iter {iteration:04d} | evals:{evaluations:04d} | {buf_status} | {status}        ", end="\r")

        if ok == 0 and iteration == 1:
            print(f"\nAVISO: no responde {args.backend}")
            if args.direct_ml:
                print("Asegúrate de que uvicorn api:app --port 8001 está corriendo.\n")
            else:
                print("Asegúrate de que el backend de Dev 2 está corriendo.\n")

        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSimulador detenido.")