#!/usr/bin/env python3
"""
Verify whether CESMD's reported PGV is geometric mean or max-horizontal.

Downloads per-channel waveforms from SCEDC FDSN for a CESMD event,
computes per-channel PGV, then compares geometric mean vs max-horizontal
against the CESMD-reported value.

Uses: simplemseed, numpy, scipy, urllib
"""

import io
import math
import urllib.request
import numpy as np
from scipy import signal, integrate
from simplemseed import readMiniseed2Records

# ── Event: ci40590799 (Brawley M3.4, 2024-05-18) ──
# Station IMP (Imperial) — CESMD reports pgv = 3.66 cm/s, epidist = 3.7 km
EVENT_TIME = "2024-05-18T19:46:17"
CESMD_PGV_CMS = 3.66  # cm/s as reported by CESMD API

# SCEDC FDSN parameters
NET = "CI"
STA = "IMP"
LOC = "--"
CHANNELS = ["HNE", "HNN", "HNZ"]  # East, North, Vertical (acceleration)
SAMP_RATE = 100  # Hz (expected)

# Time window: 10s before origin, 90s after
START = "2024-05-18T19:46:07"
END = "2024-05-18T19:47:47"

FDSN_BASE = "https://service.scedc.caltech.edu/fdsnws/dataselect/1/query"


def fetch_channel(cha):
    """Download miniSEED for one channel from SCEDC FDSN."""
    url = (
        f"{FDSN_BASE}?net={NET}&sta={STA}&loc={LOC}&cha={cha}"
        f"&starttime={START}&endtime={END}"
    )
    print(f"  Fetching {cha}: {url}")
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        data = resp.read()
    print(f"  Got {len(data)} bytes")
    return data


def mseed_to_array(raw_bytes):
    """Parse miniSEED bytes into a numpy array of samples."""
    records = readMiniseed2Records(io.BytesIO(raw_bytes))
    all_samples = []
    sample_rate = None
    for rec in records:
        all_samples.extend(rec.decompress())
        if sample_rate is None:
            sample_rate = rec.header.sampleRate
    return np.array(all_samples, dtype=float), sample_rate


def remove_mean(data):
    """Remove DC offset."""
    return data - np.mean(data)


def counts_to_acceleration(counts, sensitivity=None):
    """
    Convert raw counts to m/s².
    For CI.IMP HN channels: typical sensitivity ~1e9 counts/(m/s²).
    We'll try fetching from FDSN station service.
    """
    # We'll determine sensitivity from station metadata
    return counts / sensitivity if sensitivity else counts


def get_instrument_sensitivity(cha):
    """Fetch overall sensitivity from SCEDC FDSN station service."""
    url = (
        f"https://service.scedc.caltech.edu/fdsnws/station/1/query"
        f"?net={NET}&sta={STA}&loc=--&cha={cha}"
        f"&starttime={START}&level=response&format=xml"
    )
    print(f"  Fetching response for {cha}...")
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        xml = resp.read().decode()

    # Parse overall sensitivity (InstrumentSensitivity > Value)
    import re
    # Look for InstrumentSensitivity block
    m = re.search(
        r'<InstrumentSensitivity>\s*<Value>([\d.eE+\-]+)</Value>\s*'
        r'<Frequency>([\d.eE+\-]+)</Frequency>\s*'
        r'<InputUnits>\s*<Name>(\w+)',
        xml
    )
    if m:
        sens = float(m.group(1))
        freq = float(m.group(2))
        unit = m.group(3)
        print(f"    Sensitivity: {sens:.4e} counts/({unit}) at {freq} Hz")
        return sens, unit
    else:
        raise ValueError(f"Could not parse sensitivity from FDSN response for {cha}")


def integrate_acc_to_vel(acc, dt):
    """
    Integrate acceleration to velocity with highpass filter
    to remove baseline drift.
    """
    # Highpass filter at 0.1 Hz to remove drift
    sos = signal.butter(4, 0.1, btype='high', fs=1/dt, output='sos')
    acc_filt = signal.sosfilt(sos, acc)

    # Cumulative trapezoidal integration
    vel = integrate.cumulative_trapezoid(acc_filt, dx=dt, initial=0)

    # Filter again to clean up integration artifacts
    vel = signal.sosfilt(sos, vel)
    return vel


def main():
    print("=" * 60)
    print("CESMD PGV Verification: ci40590799, Station IMP")
    print(f"CESMD reported PGV: {CESMD_PGV_CMS} cm/s")
    print("=" * 60)

    pgv_per_channel = {}

    for cha in CHANNELS:
        print(f"\n── Channel {cha} ──")

        # 1. Get instrument response
        sensitivity, input_unit = get_instrument_sensitivity(cha)

        # 2. Download waveform
        raw = fetch_channel(cha)
        counts, sr = mseed_to_array(raw)
        dt = 1.0 / sr
        print(f"  Samples: {len(counts)}, Sample rate: {sr} Hz")

        # 3. Remove mean, convert to physical units (m/s²)
        counts = remove_mean(counts)
        acc_ms2 = counts / sensitivity  # counts / (counts/(m/s²)) = m/s²
        print(f"  Peak acceleration: {np.max(np.abs(acc_ms2)):.6f} m/s² "
              f"({np.max(np.abs(acc_ms2))/9.81:.6f} g)")

        # 4. Integrate to velocity (m/s)
        vel_ms = integrate_acc_to_vel(acc_ms2, dt)
        pgv_ms = np.max(np.abs(vel_ms))
        pgv_cms = pgv_ms * 100  # convert to cm/s
        pgv_mms = pgv_ms * 1000  # convert to mm/s

        pgv_per_channel[cha] = pgv_cms
        print(f"  PGV: {pgv_cms:.4f} cm/s  ({pgv_mms:.4f} mm/s)")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    h_channels = {k: v for k, v in pgv_per_channel.items() if k != "HNZ"}
    h_values = list(h_channels.values())

    geomean = math.sqrt(h_values[0] * h_values[1])
    max_h = max(h_values)
    arith_mean = sum(h_values) / 2

    print(f"\nPer-channel PGV (cm/s):")
    for cha, pgv in pgv_per_channel.items():
        label = " (horizontal)" if cha != "HNZ" else " (vertical)"
        print(f"  {cha}: {pgv:.4f}{label}")

    print(f"\nCombinations of horizontal components:")
    print(f"  Geometric mean:  {geomean:.4f} cm/s")
    print(f"  Max horizontal:  {max_h:.4f} cm/s")
    print(f"  Arithmetic mean: {arith_mean:.4f} cm/s")

    print(f"\nCESMD reported:    {CESMD_PGV_CMS:.4f} cm/s")

    print(f"\nRatios (computed / CESMD):")
    print(f"  Geom mean / CESMD:  {geomean / CESMD_PGV_CMS:.4f}")
    print(f"  Max horiz / CESMD:  {max_h / CESMD_PGV_CMS:.4f}")
    print(f"  Arith mean / CESMD: {arith_mean / CESMD_PGV_CMS:.4f}")

    closest = min(
        [("geometric mean", geomean), ("max horizontal", max_h), ("arithmetic mean", arith_mean)],
        key=lambda x: abs(x[1] - CESMD_PGV_CMS)
    )
    print(f"\n→ Closest match: {closest[0]} ({closest[1]:.4f} cm/s)")
    print(f"  Difference from CESMD: {abs(closest[1] - CESMD_PGV_CMS):.4f} cm/s "
          f"({abs(closest[1] - CESMD_PGV_CMS) / CESMD_PGV_CMS * 100:.1f}%)")


if __name__ == "__main__":
    main()
