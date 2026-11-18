"""The list of stations: scan the band once, keep it in a file, reuse it.

stations.json (next to the scripts unless --stations says otherwise):
  {"scanned": <unix time>, "stations": [{"freq": 100.7, "name": "SKY", "snr": 31.2, "rt": "..."}]}
"""

from __future__ import annotations

import json
import os
import time

from .scanner import scan

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FILE = os.path.join(ROOT, "stations.json")


def load(path=DEFAULT_FILE):
    """-> {"scanned", "stations"} or None when there is no (readable) file."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data.get("stations"), list):
            return data
    except (OSError, ValueError, AttributeError):
        pass
    return None


def save(data, path=DEFAULT_FILE):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def scan_band(source, threshold_db=10.0, rds_seconds=3.0, rds_min_snr=12.0, progress=None):
    """Sweep 87.5-108 MHz, then read each station's RDS name.

    progress(phase, done, total, data): phase "sweep" (data: MHz being swept),
    "found" (data: the list so far, names still empty) and "rds" (data: the list,
    names filled in as they come). -> {"scanned", "stations"}
    """
    def sweep_progress(i, n, fc):
        if progress:
            progress("sweep", i, n, round(fc / 1e6, 2))

    result = scan(source, threshold_db=threshold_db, progress=sweep_progress)
    found = sorted(result.stations, key=lambda s: s.freq_hz)
    out = [{"freq": round(s.freq_mhz, 1), "name": "", "snr": round(float(s.snr_db), 1), "rt": ""} for s in found]
    if progress:
        progress("found", len(out), len(out), out)
    try:
        from .rds import read_station_rds
    except ImportError:
        read_station_rds = None
    if read_station_rds and rds_seconds > 0:
        targets = [i for i, s in enumerate(found) if s.snr_db >= rds_min_snr]
        for k, i in enumerate(targets):
            try:
                info = read_station_rds(source, found[i].freq_hz, max_seconds=rds_seconds)
                out[i]["name"] = (info.ps or "").strip()
                out[i]["rt"] = (info.rt or "").strip()
            except Exception:
                pass
            if progress:
                progress("rds", k + 1, len(targets), out)
    return {"scanned": time.time(), "stations": out}


def decode_all_rds(source, stations, seconds: float, min_snr: float) -> dict:
    """Listen to each station in turn and decode its RDS. Returns {freq_hz: RdsInfo}."""
    from .rds import read_station_rds

    targets = [s for s in stations if s.snr_db >= min_snr]
    if not targets:
        return {}
    print(f"\nReading RDS from {len(targets)} station(s), up to {seconds:g} s each "
          "(Ctrl+C to skip) ...")
    results = {}
    try:
        for s in targets:
            print(f"  {s.freq_mhz:6.1f} MHz ... ", end="", flush=True)
            t0 = time.time()
            info = read_station_rds(source, s.freq_hz, max_seconds=seconds)
            results[s.freq_hz] = info
            label = info.ps or ("no RDS" if info.groups == 0 else "incomplete")
            print(f"{label:<10} ({info.groups} groups, {time.time() - t0:.1f} s)")
    except KeyboardInterrupt:
        print("\n  RDS skipped.")
    return results
