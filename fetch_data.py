#!/usr/bin/env python3
"""
Fetch seismic waveform data for the Zeerijp M3.4 earthquake (2018-01-08)
from the KNMI FDSN webservice and export processed data as JSON.

Fetches three stations at increasing distances to show PGV attenuation.
"""

import json
import pathlib
import numpy as np
from obspy import UTCDateTime, Stream
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth
from scipy.signal import detrend
from scipy.integrate import cumulative_trapezoid


# ── Event parameters ──────────────────────────────────────────────────────────
EVENT = {
    "name": "Zeerijp",
    "date": "2018-01-08",
    "time": "14:00:52",
    "latitude": 53.3633,
    "longitude": 6.7862,
    "depth_km": 3.0,
    "magnitude": 3.4,
    "magnitude_type": "ML",
}

# Three stations at increasing distances
STATIONS = ["G140", "G190", "G230"]
NETWORK = "NL"

# Time window around the event
PRE_EVENT_SEC = 5
POST_EVENT_SEC = 30

# Bandpass filter corners (Hz)
FMIN = 0.5
FMAX = 80.0

OUTPUT_PATH = pathlib.Path(__file__).parent / "data" / "waveform_data.json"


def downsample(times, data, target_points=2000):
    """Reduce array to ~target_points for JSON size."""
    n = len(data)
    if n <= target_points:
        return times.tolist(), data.tolist()
    step = max(1, n // target_points)
    idx = np.arange(0, n, step)
    return times[idx].tolist(), data[idx].tolist()


def process_station(client, station_code, origin_time, t_start, t_end):
    """Fetch and process waveforms for a single station. Returns a dict."""
    print(f"\n{'='*60}")
    print(f"Processing station {NETWORK}.{station_code}...")

    # Fetch metadata with response info
    inventory = client.get_stations(
        network=NETWORK, station=station_code,
        starttime=t_start, endtime=t_end, level="response",
    )
    sta_meta = inventory[0][0]
    sta_lat, sta_lon = sta_meta.latitude, sta_meta.longitude

    dist_m, _, _ = gps2dist_azimuth(
        EVENT["latitude"], EVENT["longitude"], sta_lat, sta_lon
    )
    dist_km = dist_m / 1000.0
    print(f"  Distance: {dist_km:.2f} km")

    # Fetch all HG? components, pick the one with highest PGA
    st_all = client.get_waveforms(NETWORK, station_code, "*", "HG?", t_start, t_end)
    print(f"  Traces: {[tr.stats.channel for tr in st_all]}")

    best_tr, best_peak = None, 0
    for tr in st_all:
        tr_copy = tr.copy()
        tr_copy.remove_response(inventory=inventory, output="ACC")
        peak = np.max(np.abs(tr_copy.data))
        if peak > best_peak:
            best_peak = peak
            best_tr = tr
    print(f"  Using channel {best_tr.stats.channel}")

    # Process: remove response → detrend → filter
    st_acc = Stream([best_tr.copy()])
    st_acc.remove_response(inventory=inventory, output="ACC")
    st_acc.detrend("demean")
    st_acc.detrend("linear")
    st_acc.filter("bandpass", freqmin=FMIN, freqmax=FMAX, corners=4, zerophase=True)

    tr_acc = st_acc[0]
    acc_data = tr_acc.data * 100.0  # m/s² → cm/s²

    # Integrate to velocity
    dt = 1.0 / tr_acc.stats.sampling_rate
    vel_raw = cumulative_trapezoid(tr_acc.data, dx=dt, initial=0)
    vel_raw = detrend(vel_raw, type='linear')
    vel_data = vel_raw * 1000.0  # m/s → mm/s

    # Time axis relative to origin
    acc_times = np.arange(len(acc_data)) / tr_acc.stats.sampling_rate - PRE_EVENT_SEC
    vel_times = acc_times.copy()

    # PGA
    pga_idx = int(np.argmax(np.abs(acc_data)))
    pga_value = float(np.abs(acc_data[pga_idx]))
    pga_signed = float(acc_data[pga_idx])
    pga_time = float(acc_times[pga_idx])

    # PGV
    pgv_idx = int(np.argmax(np.abs(vel_data)))
    pgv_value = float(np.abs(vel_data[pgv_idx]))
    pgv_signed = float(vel_data[pgv_idx])
    pgv_time = float(vel_times[pgv_idx])

    print(f"  PGA = {pga_value:.2f} cm/s² at t = {pga_time:.3f} s")
    print(f"  PGV = {pgv_value:.2f} mm/s at t = {pgv_time:.3f} s")

    # Downsample
    acc_t_ds, acc_d_ds = downsample(acc_times, acc_data)
    vel_t_ds, vel_d_ds = downsample(vel_times, vel_data)

    return {
        "station": {
            "network": NETWORK,
            "code": station_code,
            "latitude": float(sta_lat),
            "longitude": float(sta_lon),
            "distance_km": round(dist_km, 2),
        },
        "acceleration": {
            "times": [round(t, 4) for t in acc_t_ds],
            "data": [round(d, 4) for d in acc_d_ds],
            "unit": "cm/s²",
            "sampling_rate": float(tr_acc.stats.sampling_rate),
            "channel": tr_acc.stats.channel,
            "pga": {
                "value": round(pga_value, 4),
                "value_g": round(pga_value / 981.0, 6),
                "signed_value": round(pga_signed, 4),
                "time": round(pga_time, 4),
            },
        },
        "velocity": {
            "times": [round(t, 4) for t in vel_t_ds],
            "data": [round(d, 4) for d in vel_d_ds],
            "unit": "mm/s",
            "sampling_rate": float(tr_acc.stats.sampling_rate),
            "channel": tr_acc.stats.channel + " (integrated)",
            "pgv": {
                "value": round(pgv_value, 4),
                "signed_value": round(pgv_signed, 4),
                "time": round(pgv_time, 4),
            },
        },
    }


def fetch_and_process():
    """Main pipeline: fetch waveforms for 3 stations, export JSON."""

    origin_time = UTCDateTime(f"{EVENT['date']}T{EVENT['time']}")
    t_start = origin_time - PRE_EVENT_SEC
    t_end = origin_time + POST_EVENT_SEC

    client = Client("http://rdsa.knmi.nl")

    stations_data = []
    for code in STATIONS:
        result = process_station(client, code, origin_time, t_start, t_end)
        stations_data.append(result)

    # Sort by distance (should already be, but be explicit)
    stations_data.sort(key=lambda s: s["station"]["distance_km"])

    output = {
        "event": {
            "name": EVENT["name"],
            "date": EVENT["date"],
            "time": EVENT["time"],
            "latitude": EVENT["latitude"],
            "longitude": EVENT["longitude"],
            "depth_km": EVENT["depth_km"],
            "magnitude": EVENT["magnitude"],
            "magnitude_type": EVENT["magnitude_type"],
        },
        "stations": stations_data,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nData written to {OUTPUT_PATH}")
    for s in stations_data:
        print(f"  {s['station']['code']}: {s['station']['distance_km']} km, "
              f"PGV = {s['velocity']['pgv']['value']:.1f} mm/s")


if __name__ == "__main__":
    fetch_and_process()
