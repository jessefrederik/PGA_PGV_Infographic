"""
prepare_data.py — Reads CSVs + waveform JSON → outputs data/scroll_data.json
for the scrollytelling PGV infographic.

International comparison uses m34_pgv_with_vs30.csv from the ML comparison
pipeline (03_generate_m34_csv.R), which already has:
  - Borehole/downhole station removal (surface only)
  - ML/JMA magnitude filtering (M3.35–3.45)
  - Depth ≤15 km
  - Geometric mean PGV
"""

import csv
import json
import math
import random
import re
from pathlib import Path

random.seed(42)

DATA = Path("data")

DIST_MAX_KM = 50

# ── Helpers ──────────────────────────────────────────────────────────────────

def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def quantile(sorted_vals, q):
    n = len(sorted_vals)
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (pos - lo) * (sorted_vals[hi] - sorted_vals[lo])

def compute_box(values):
    """Compute box plot stats: q1, median, q3, whiskers, outliers."""
    if not values:
        return None
    s = sorted(values)
    q1 = quantile(s, 0.25)
    med = quantile(s, 0.50)
    q3 = quantile(s, 0.75)
    iqr = q3 - q1
    wlo = q1 - 1.5 * iqr
    whi = q3 + 1.5 * iqr
    whisker_lo = min(v for v in s if v >= wlo)
    whisker_hi = max(v for v in s if v <= whi)
    outliers = [v for v in s if v < wlo or v > whi]
    return {
        "q1": round(q1, 4), "median": round(med, 4), "q3": round(q3, 4),
        "whisker_lo": round(whisker_lo, 4), "whisker_hi": round(whisker_hi, 4),
        "outliers": [round(v, 4) for v in outliers], "n": len(s)
    }

BINS = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 50)]
BIN_LABELS = ["0–5", "5–10", "10–20", "20–30", "30–50"]

def bin_index(dist):
    for i, (lo, hi) in enumerate(BINS):
        if lo <= dist < hi:
            return i
    return None

def is_surface_station(station_id):
    """Groningen borehole filter: G-stations ending in 1-4 are deep sensors."""
    if re.match(r'^G\d{2}[1-9]$', station_id):
        return False
    return True

# ── 1. Waveform data (Step 0) ───────────────────────────────────────────────

with open(DATA / "waveform_data.json") as f:
    waveform_raw = json.load(f)

# Use first station (G140, closest)
st0 = waveform_raw["stations"][0]
waveform = {
    "times": st0["velocity"]["times"],
    "velocity": st0["velocity"]["data"],
    "pgv": st0["velocity"]["pgv"],
    "acceleration": st0["acceleration"]["data"],
    "pga": st0["acceleration"]["pga"],
    "acc_unit": st0["acceleration"]["unit"],
    "station_id": st0["station"]["code"],
    "distance_km": st0["station"]["distance_km"],
    "unit": st0["velocity"]["unit"],
    "event": waveform_raw["event"],
}

# ── 2. Groningen event data (Steps 1-3) ──────────────────────────────────────

gron_rows = read_csv(DATA / "groningen_pgv.csv")

# Parse into event groups, filtering borehole stations
gron_events = {}
n_borehole_removed = 0
for r in gron_rows:
    sid = r["station_id"]
    if not is_surface_station(sid):
        n_borehole_removed += 1
        continue
    eid = r["event_id"]
    if eid not in gron_events:
        gron_events[eid] = {
            "event_id": eid,
            "date": r["event_date"],
            "magnitude": float(r["magnitude"]),
            "stations": [],
        }
    dist = float(r["epicentral_dist_km"])
    pgv = float(r["pgv_mms"])
    if 0 < dist <= DIST_MAX_KM and pgv > 0:
        gron_events[eid]["stations"].append({
            "dist": round(dist, 3),
            "pgv": round(pgv, 4),
            "station_id": sid,
        })

print(f"Groningen: {len(gron_rows)} rows, {n_borehole_removed} borehole removed")

# Name mapping for the three target events
EVENT_NAMES = {
    "GRN_24": "Zeerijp 2018",
    "GRN_26": "Westerwijtwerd 2019",
    "GRN_35": "Zeerijp 2025",
}

# Step 1: Mini-stations — pick 20 from GRN_24 spanning distance range
zeerijp = gron_events["GRN_24"]
zeerijp["name"] = EVENT_NAMES["GRN_24"]

# Pick 20 representative stations (stratified across distance)
zeerijp_sorted = sorted(zeerijp["stations"], key=lambda s: s["dist"])
n = len(zeerijp_sorted)
indices = [int(i * (n - 1) / 19) for i in range(20)]
mini_stations = [
    {"station_id": zeerijp_sorted[i]["station_id"],
     "distance_km": zeerijp_sorted[i]["dist"],
     "pgv_mms": zeerijp_sorted[i]["pgv"]}
    for i in indices
]

# Enrich mini_stations with lat/lon from mini_waveforms.json (KNMI FDSN coordinates)
mini_wf_path = DATA / "mini_waveforms.json"
if mini_wf_path.exists():
    with open(mini_wf_path) as f:
        mini_wf = json.load(f)
    coord_map = {s["station_id"]: (s.get("lat"), s.get("lon")) for s in mini_wf}
    for ms in mini_stations:
        lat, lon = coord_map.get(ms["station_id"], (None, None))
        if lat is not None:
            ms["lat"] = lat
            ms["lon"] = lon

# Step 2-3: Scatter events
three_events = []
for eid in ["GRN_24", "GRN_26", "GRN_35"]:
    ev = gron_events[eid]
    ev["name"] = EVENT_NAMES[eid]
    three_events.append({
        "event_id": ev["event_id"],
        "name": ev["name"],
        "date": ev["date"],
        "magnitude": ev["magnitude"],
        "stations": ev["stations"],
    })

# ── 3. International data from combined CSV ──────────────────────────────────
# Uses m34_pgv_with_vs30.csv produced by ml_comparison_pipeline/03_generate_m34_csv.R
# Already filtered: surface-only, ML/JMA mag 3.35-3.45, geometric mean PGV

REGION_MAP = {
    "Groningen": "Netherlands",
    "Japan": "Japan",
    "Italy": "Italy",
    "California": "California",
}

combined_rows = read_csv(DATA / "m34_pgv_with_vs30.csv")
intl_data = {name: [] for name in REGION_MAP.values()}

for r in combined_rows:
    region = r["region"]
    if region not in REGION_MAP:
        continue
    country = REGION_MAP[region]
    dist = float(r["epicentral_dist_km"])
    pgv = float(r["pgv_mms"])
    if 0 < dist <= DIST_MAX_KM and pgv > 0:
        intl_data[country].append({"dist": round(dist, 3), "pgv": round(pgv, 4)})

for country, stations in intl_data.items():
    print(f"  {country}: {len(stations)} records")

# ── 4. Box plots (Step 4-5) ─────────────────────────────────────────────────

def compute_country_boxes(stations):
    binned = [[] for _ in BINS]
    for s in stations:
        bi = bin_index(s["dist"])
        if bi is not None:
            binned[bi].append(s["pgv"])
    return [compute_box(b) for b in binned]

box_plots = {
    "bins": BIN_LABELS,
    "countries": {
        country: {"bins": compute_country_boxes(stations)}
        for country, stations in intl_data.items()
    }
}

# ── 5. Scatter for international (sampled) ──────────────────────────────────

def sample_stratified(stations, target=800):
    """Downsample large datasets, stratified by distance bin."""
    if len(stations) <= target:
        return stations
    binned = [[] for _ in BINS]
    for s in stations:
        bi = bin_index(s["dist"])
        if bi is not None:
            binned[bi].append(s)
    per_bin = target // len(BINS)
    result = []
    for b in binned:
        if len(b) <= per_bin:
            result.extend(b)
        else:
            result.extend(random.sample(b, per_bin))
    return result

scatter = {
    country: [{"dist": s["dist"], "pgv": s["pgv"]}
              for s in sample_stratified(stations)]
    for country, stations in intl_data.items()
}

# ── 6. Assemble and write ───────────────────────────────────────────────────

output = {
    "waveform": waveform,
    "miniStations": mini_stations,
    "zeerijp": three_events[0],  # GRN_24
    "groningenEvents": three_events,
    "boxPlots": box_plots,
    "scatter": scatter,
}

out_path = DATA / "scroll_data.json"
with open(out_path, "w") as f:
    json.dump(output, f, separators=(",", ":"))

size_kb = out_path.stat().st_size / 1024
print(f"\n✓ Wrote {out_path} ({size_kb:.0f} KB)")
print(f"  Waveform samples: {len(waveform['times'])}")
print(f"  Mini stations: {len(mini_stations)}")
print(f"  Zeerijp stations (surface): {len(zeerijp['stations'])}")
ev_summary = [e["name"] + " (" + str(len(e["stations"])) + ")" for e in three_events]
print(f"  Three events: {ev_summary}")
print(f"  Scatter: " + ", ".join(f"{k}: {len(v)}" for k, v in scatter.items()))
for country, data in box_plots["countries"].items():
    meds = [b["median"] if b else None for b in data["bins"]]
    print(f"  Box {country}: medians = {meds}")
