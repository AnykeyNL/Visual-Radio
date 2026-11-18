"""FM -> DAB+ -> FM, the way visual_radio.py hands the Airspy around.

Stop visual_radio.py first (it uses the Airspy), then:

    conda activate sdr
    python tools\\airspy_handover_test.py            (DAB+ channel 12C; another one: --channel 11C)
    python tools\\airspy_handover_test.py --reception     (only: why does welle-cli find nothing?)

1. FM: the Airspy in its helper process, half a second of samples, then let go.
2. DAB+: welle-cli opens the Airspy and looks for the ensemble on the channel.
3. FM again: a new helper opens the Airspy.
Twice round. When DAB+ finds nothing, it then tries a few gains on two channels
and shows what welle-cli reports (signal to noise, its own messages).
Paste what it prints.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faulthandler  # noqa: E402

import numpy as np  # noqa: E402

from visualradio import dab  # noqa: E402
from visualradio.airspy_proc import AirspyProcess  # noqa: E402


def fm(step):
    t0 = time.time()
    src = AirspyProcess(sample_rate=6e6, gain=18)
    src.tune(100.0e6)
    a = src.read(3_000_000)
    level = 10 * np.log10(float(np.mean(np.abs(a) ** 2)) + 1e-12)
    src.close()
    print(f"{step} FM: opened, {len(a)} samples ({level:.0f} dB), let go - {time.time() - t0:.1f} s", flush=True)


def dab_step(step, exe, channel):
    t0 = time.time()
    w = dab.WelleCli(exe, channel, log=lambda m: print("   welle-cli:", m))
    try:
        ens = dab.scan_channel(w, channel, settle=6, max_wait=20)
        if ens:
            names = ", ".join(s["name"] for s in ens["services"][:6])
            print(f"{step} DAB+ {channel}: {ens['label']} ({len(ens['services'])} stations: {names}) "
                  f"- {time.time() - t0:.1f} s", flush=True)
            return True
        else:
            print(f"{step} DAB+ {channel}: welle-cli runs, but found no ensemble (signal? other channel?) "
                  f"- {time.time() - t0:.1f} s", flush=True)
        return False
    finally:
        w.close()


def main():
    faulthandler.enable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="12C")
    ap.add_argument("--reception", action="store_true", help="only the DAB+ reception check")
    ap.add_argument("--welle")
    a = ap.parse_args()
    exe = dab.find_welle(a.welle)
    if not exe:
        sys.exit(dab.WELLE_HELP)
    found = False
    try:
        if not a.reception:
            for r in (1, 2):
                fm(f"[{r}.1]")
                found = dab_step(f"[{r}.2]", exe, a.channel) or found
            fm("[3.1]")
            print("\nThe Airspy goes from FM to DAB+ and back.")
        if not found:
            reception(exe, [a.channel] + [c for c in ("12C", "11C") if c != a.channel][:1])
    except Exception as exc:
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        sys.exit(1)


def reception(exe, channels):
    """Several gains x channels: what does welle-cli see?"""
    print("\nDAB+ reception check (takes about two minutes):", flush=True)
    shown = False
    for gain in (-1, 6, 12, 18):
        w = dab.WelleCli(exe, channels[0], gain=gain, log=lambda m: None)
        try:
            for ch in channels:
                if ch != w.channel:
                    w.set_channel(ch)
                best, label, n, t0, m = None, "", 0, time.time(), {}
                while time.time() - t0 < 12 and not (label and n):
                    time.sleep(1.0)
                    try:
                        m = w.mux()
                    except Exception:
                        continue
                    snr = (m.get("demodulator") or {}).get("snr")
                    if isinstance(snr, (int, float)):
                        best = snr if best is None else max(best, snr)
                    label, n = dab.ensemble_label(m), len(dab.audio_services(m))
                hw = (m.get("receiver") or {}).get("hardware") or {}
                print(f"  gain {'auto' if gain < 0 else gain:>4}  {ch:>3}: "
                      + (f"{label!r}, {n} stations" if label or n else "nothing")
                      + (f", SNR {best:.1f} dB" if best is not None else "")
                      + (f"  [{hw.get('name', '')} gain {hw.get('gain', '?')}]" if hw else ""), flush=True)
            if not shown:
                shown = True
                tail = [ln for ln in w.lines if ln.strip()][-15:]
                print("  welle-cli says:\n    " + "\n    ".join(tail or ["(nothing)"]), flush=True)
        finally:
            w.close()
        time.sleep(0.5)


if __name__ == "__main__":
    main()
