"""Which way of letting go of the Airspy works on this PC?

For DAB+ the program hands the Airspy Mini to welle-cli and later takes it back.
On some PCs SoapySDR's Airspy driver crashes (access violation) while it closes
the device. This tries a few ways to close it, each in its own process, so a
crash only ends that one try. Stop visual_radio.py first (it uses the Airspy).

    conda activate sdr
    python tools\\airspy_release_test.py

Takes about a minute; paste the summary at the end.
"""

from __future__ import annotations

import faulthandler
import os
import subprocess
import sys
import time

WAYS = {
    "A": "stop stream, close stream, close device   (what the program does now)",
    "B": "stop stream, close device                 (stream left to the driver)",
    "C": "close stream, close device                (closing stops it)",
    "D": "close device only                         (driver stops the stream)",
    "E": "stop, wait, close stream, wait, close device",
    "F": "stop stream, close stream, keep device open",
}


def open_airspy():
    import numpy as np
    import SoapySDR

    rx = SoapySDR.SOAPY_SDR_RX
    dev = SoapySDR.Device("driver=airspy")
    dev.setSampleRate(rx, 0, 6e6)
    dev.setFrequency(rx, 0, 100e6)
    st = dev.setupStream(rx, SoapySDR.SOAPY_SDR_CF32)
    dev.activateStream(st)
    buf = np.empty(65536, np.complex64)
    got, t0 = 0, time.time()
    while got < 3_000_000 and time.time() - t0 < 5:      # half a second of samples
        sr = dev.readStream(st, [buf], len(buf), timeoutUs=1_000_000)
        got += max(sr.ret, 0)
    return SoapySDR, dev, st, got


def child(way):
    faulthandler.enable()
    SoapySDR, dev, st, got = open_airspy()
    print(f"  opened, read {got} samples", flush=True)
    if way in "ABEF":
        dev.deactivateStream(st)
    if way == "E":
        time.sleep(0.5)
    if way in "ACEF":
        dev.closeStream(st)
    if way == "E":
        time.sleep(0.5)
    if way != "F":
        SoapySDR.Device.unmake(dev) if not hasattr(dev, "close") else dev.close()
        del dev
    print("  released without a crash", flush=True)
    # can another program (like welle-cli) open it now?
    r = subprocess.run([sys.executable, __file__, "--open"], capture_output=True, text=True, timeout=60)
    print("  another program can open it: " + ("yes" if r.returncode == 0 else "NO"), flush=True)
    # can we open it again ourselves (back to FM)?
    try:
        _, dev2, st2, got2 = open_airspy()
        print(f"  open again in the same program: yes ({got2} samples)", flush=True)
    except Exception as exc:
        print(f"  open again in the same program: NO ({exc})", flush=True)
    os._exit(0)            # skip clean-up at exit: we test only the step above


def main():
    if "--open" in sys.argv:
        try:
            _, dev, st, got = open_airspy()
        except Exception:
            os._exit(1)
        os._exit(0 if got > 0 else 1)
    if "--way" in sys.argv:
        child(sys.argv[sys.argv.index("--way") + 1])
        return
    results = {}
    for way, what in WAYS.items():
        print(f"\n{way}: {what}", flush=True)
        r = subprocess.run([sys.executable, __file__, "--way", way], capture_output=True, text=True, timeout=120)
        out = r.stdout.rstrip()
        print(out)
        crashed = r.returncode != 0
        if crashed:
            print(f"  CRASHED (exit code {r.returncode & 0xFFFFFFFF:#x})")
            if r.stderr:
                print("  " + "\n  ".join(r.stderr.strip().splitlines()[:6]))
        results[way] = ("crash" if crashed else "ok") + ("" if crashed else
                        (", other program " + ("yes" if "open it: yes" in out else "no"))
                        + (", reopen " + ("yes" if "same program: yes" in out else "no")))
        time.sleep(1.0)
    print("\nSummary (paste this):")
    for way, res in results.items():
        print(f"  {way}: {res}")


if __name__ == "__main__":
    main()
