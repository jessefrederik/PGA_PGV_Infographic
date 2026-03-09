#!/usr/bin/env python3
"""
Fetch real velocity waveforms for 4 stations per distance bin (20 total)
from KNMI FDSN. Bins match the box plot: 0–5, 5–10, 10–20, 20–30, 30–50 km.
Outputs data/mini_waveforms.json with ~80-point sparklines.
"""

import json
import pathlib
import numpy as np
from obspy import UTCDateTime, Stream
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth
from scipy.signal import detrend
from scipy.integrate import cumulative_trapezoid

# ── Config ───────────────────────────────────────────────────────────────────
ORIGIN = UTCDateTime("2018-01-08T14:00:52")
EV_LAT, EV_LON = 53.3633, 6.7862
PRE_SEC = 2
POST_SEC = 25
FMIN = 0.5
FMAX = 40.0
TARGET_PTS = 200
PER_BIN = 4
BINS = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 50)]
BIN_LABELS = ["0–5", "5–10", "10–20", "20–30", "30–50"]

DATA = pathlib.Path("data")
OUTPUT = DATA / "mini_waveforms.json"

client = Client("http://rdsa.knmi.nl")
t_start = ORIGIN - PRE_SEC
t_end = ORIGIN + POST_SEC

# ── 1. Discover all available stations ───────────────────────────────────────
print("Querying available stations...")
inv_all = client.get_stations(
    network="NL", starttime=t_start, endtime=t_end,
    channel="HG?,HN?,BH?", level="station",
    minlatitude=52.5, maxlatitude=54.0,
    minlongitude=5.5, maxlongitude=7.5,
)

candidates = []
for net in inv_all:
    for sta in net:
        d_m, _, _ = gps2dist_azimuth(EV_LAT, EV_LON, sta.latitude, sta.longitude)
        d_km = d_m / 1000
        if 0.3 <= d_km <= 52:
            candidates.append((net.code, sta.code, round(d_km, 2)))

candidates.sort(key=lambda x: x[2])
print(f"Found {len(candidates)} candidates total")

# ── 2. Pick candidates per bin (more than needed, in case some fail) ─────────
def bin_index(dist):
    for i, (lo, hi) in enumerate(BINS):
        if lo <= dist < hi:
            return i
    return None

EXCLUDE_STATIONS = {"BOWW"}  # replaced by BGAR
FORCE_STATIONS = [("NL", "BGAR", 2.68, 0)]  # (net, code, dist_km, bin_idx)

binned = [[] for _ in BINS]
for c in candidates:
    if c[1] in EXCLUDE_STATIONS:
        continue
    bi = bin_index(c[2])
    if bi is not None:
        binned[bi].append(c)

# Inject forced stations into targets
for net, code, dist, bi in FORCE_STATIONS:
    binned[bi].insert(0, (net, code, dist))  # prioritize at front

# Pick evenly spaced within each bin, extra candidates for fallback
targets = []
for bi, pool in enumerate(binned):
    n = len(pool)
    pick_n = min(PER_BIN + 4, n)  # extra fallback
    if n == 0:
        print(f"  WARNING: no candidates for bin {BIN_LABELS[bi]}")
        continue
    indices = [int(i * (n - 1) / max(1, pick_n - 1)) for i in range(pick_n)]
    seen = set()
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            targets.append((*pool[idx], bi))
    print(f"  Bin {BIN_LABELS[bi]}: {n} available, targeting {pick_n}")


# ── 3. Fetch waveforms ──────────────────────────────────────────────────────
results_per_bin = [[] for _ in BINS]

for net, code, dist, bi in targets:
    if len(results_per_bin[bi]) >= PER_BIN:
        continue

    print(f"  {code:6s} ({dist:5.1f} km, bin {BIN_LABELS[bi]})... ", end="", flush=True)

    try:
        st_raw = Stream()
        for chan in ["HG?", "HN?", "BH?"]:
            try:
                st_raw = client.get_waveforms(net, code, "*", chan, t_start, t_end)
                if len(st_raw) > 0:
                    break
            except Exception:
                continue

        if len(st_raw) == 0:
            print("no waveforms")
            continue

        inv = client.get_stations(
            network=net, station=code,
            starttime=t_start, endtime=t_end, level="response",
        )

        best_tr, best_peak = None, 0
        for tr in st_raw:
            try:
                tr_copy = tr.copy()
                tr_copy.remove_response(inventory=inv, output="ACC")
                peak = np.max(np.abs(tr_copy.data))
                if peak > best_peak:
                    best_peak = peak
                    best_tr = tr
            except Exception:
                continue

        if best_tr is None:
            print("response failed")
            continue

        st_proc = Stream([best_tr.copy()])
        st_proc.remove_response(inventory=inv, output="ACC")
        st_proc.detrend("demean")
        st_proc.detrend("linear")
        st_proc.filter("bandpass", freqmin=FMIN, freqmax=FMAX, corners=4, zerophase=True)

        tr_acc = st_proc[0]
        dt = 1.0 / tr_acc.stats.sampling_rate
        vel_raw = cumulative_trapezoid(tr_acc.data, dx=dt, initial=0)
        vel_raw = detrend(vel_raw, type="linear")
        vel_mms = vel_raw * 1000.0

        n_pts = len(vel_mms)
        step = max(1, n_pts // TARGET_PTS)
        idx = np.arange(0, n_pts, step)[:TARGET_PTS]

        times = np.arange(n_pts) * dt - PRE_SEC
        pgv_actual = float(np.max(np.abs(vel_mms)))

        results_per_bin[bi].append({
            "station_id": code,
            "distance_km": dist,
            "pgv_mms": round(pgv_actual, 2),
            "bin": BIN_LABELS[bi],
            "bin_index": bi,
            "times": [round(float(t), 3) for t in times[idx]],
            "velocity": [round(float(v), 4) for v in vel_mms[idx]],
        })

        print(f"OK  PGV={pgv_actual:.1f} mm/s")

    except Exception as e:
        print(f"error: {e}")

# Flatten and sort
results = []
for bi, bin_results in enumerate(results_per_bin):
    bin_results.sort(key=lambda s: s["distance_km"])
    results.extend(bin_results)

with open(OUTPUT, "w") as f:
    json.dump(results, f, separators=(",", ":"))

size_kb = OUTPUT.stat().st_size / 1024
print(f"\n✓ Wrote {OUTPUT} ({size_kb:.0f} KB)")
for bi, label in enumerate(BIN_LABELS):
    n = len(results_per_bin[bi])
    codes = [s["station_id"] for s in results_per_bin[bi]]
    print(f"  {label} km: {n} stations — {codes}")
