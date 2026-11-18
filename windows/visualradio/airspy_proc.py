"""The Airspy Mini in a helper process, so it can be let go of safely.

On some PCs SoapySDR's Airspy driver crashes (access violation) whenever it
closes the device, which takes the whole program down. For DAB+ the Airspy must
be handed to welle-cli and later taken back, so here the Airspy lives in a small
child process: closing it ends that process, Windows frees the USB device, and
welle-cli (or a new helper) can open it straight away.

AirspyProcess has the same interface as sources.AirspySource:
    src.sample_rate, src.gain, src.info, src.describe(), src.tune(hz), src.read(n), src.close()

Between the processes: the child writes frames [4-byte tag][uint32 length][data]
to its stdout ("IQ  " complex64 samples, "ACK " after a retune, "RDY " with JSON
facts once the device is open, "ERR " with a message); it reads commands
("tune <hz>") from its stdin and stops when stdin closes.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import threading
import time

import numpy as np

from .sources import DeviceError

HEAD = struct.Struct("<4sI")


class AirspyProcess:
    def __init__(self, sample_rate=6e6, gain=18.0, serial=None, settle_s=0.03, start_timeout=20.0):
        cmd = [sys.executable, "-u", "-m", "visualradio.airspy_proc", "--rate", repr(float(sample_rate)),
               "--gain", repr(float(gain)), "--settle", repr(float(settle_s))]
        if serial:
            cmd += ["--serial", serial]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     cwd=root, bufsize=0, creationflags=flags)
        self.out = self.proc.stdout
        self.err_lines = []
        threading.Thread(target=self._pump_stderr, daemon=True).start()
        self._seq = 0
        self._left = 0                       # samples of the current IQ frame not read yet
        ready = threading.Event()
        proc = self.proc

        def watchdog():                      # a helper that hangs while opening the device
            if not ready.wait(start_timeout) and proc.poll() is None:
                proc.kill()
        threading.Thread(target=watchdog, daemon=True).start()
        try:
            while True:
                tag, body = self._frame()
                if tag == b"RDY ":
                    facts = json.loads(body.decode())
                    break
                if tag == b"ERR ":
                    raise DeviceError(body.decode(errors="replace"))
        except DeviceError:
            self.close()
            raise
        finally:
            ready.set()
        self.sample_rate = float(facts["rate"])
        self.gain = float(facts["gain"])
        self.info = facts.get("info") or {}

    def _pump_stderr(self):
        for line in iter(self.proc.stderr.readline, b""):
            self.err_lines = (self.err_lines + [line.decode(errors="replace").rstrip()])[-20:]

    def _exact(self, n):
        buf = bytearray(n)
        view, got = memoryview(buf), 0
        while got < n:
            k = self.out.readinto(view[got:])
            if not k:
                raise DeviceError("The Airspy helper stopped" + (": " + self.err_lines[-1] if self.err_lines else ""))
            got += k
        return buf

    def _frame(self):
        """The next whole non-IQ frame, or an IQ frame's header (its samples are left in the pipe)."""
        tag, n = HEAD.unpack(self._exact(HEAD.size))
        if tag == b"IQ  ":
            self._left = n // 8
            return tag, None
        return tag, bytes(self._exact(n)) if n else b""

    def _skip_iq(self):
        while self._left:
            k = min(self._left, 1 << 16)
            self._exact(8 * k)
            self._left -= k

    def describe(self):
        label = self.info.get("label") or self.info.get("product") or "Airspy"
        return (f"{label} (serial {self.info.get('serial', '?')}), {self.sample_rate / 1e6:g} MSPS, "
                f"gain {self.gain:g} dB")

    def tune(self, freq_hz):
        self._seq += 1
        self._send(f"tune {float(freq_hz)!r} {self._seq}\n")
        while True:                          # samples from before the retune are thrown away
            self._skip_iq()
            tag, body = self._frame()
            if tag == b"ACK " and body.decode() == str(self._seq):
                return
            if tag == b"ERR ":
                raise DeviceError(body.decode(errors="replace"))

    def read(self, n):
        out = np.empty(n, dtype=np.complex64)
        got = 0
        while got < n:
            if not self._left:
                tag, body = self._frame()
                if tag == b"ERR ":
                    raise DeviceError(body.decode(errors="replace"))
                continue
            k = min(self._left, n - got)
            out[got:got + k] = np.frombuffer(self._exact(8 * k), dtype=np.complex64)
            got += k
            self._left -= k
        return out

    def _send(self, line):
        try:
            self.proc.stdin.write(line.encode())
            self.proc.stdin.flush()
        except OSError as exc:
            raise DeviceError(f"The Airspy helper stopped ({exc})") from exc

    def close(self):
        """Let go of the Airspy: end the helper (Windows then frees the USB device)."""
        p = getattr(self, "proc", None)
        if p is None:
            return
        self.proc = None
        for f in (p.stdin, p.stdout):        # the helper stops by itself when it sees this
            try:
                f.close()
            except OSError:
                pass
        try:
            p.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            p.kill()                         # Windows frees the USB device all the same
            p.wait(timeout=5.0)
        try:
            p.stderr.close()
        except OSError:
            pass
        time.sleep(0.3)                      # USB needs a moment before the next one opens it


# ---------------------------------------------------------------- the child process

def _child(argv):
    import argparse
    import queue

    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=6e6)
    ap.add_argument("--gain", type=float, default=18.0)
    ap.add_argument("--serial")
    ap.add_argument("--settle", type=float, default=0.03)
    a = ap.parse_args(argv)
    out = sys.stdout.buffer
    lock = threading.Lock()

    def send(tag, data=b""):
        with lock:
            out.write(HEAD.pack(tag, len(data)))
            if len(data):
                out.write(data)
            out.flush()

    try:
        from .sources import AirspySource
        src = AirspySource(sample_rate=a.rate, gain=a.gain, serial=a.serial, settle_s=a.settle)
    except Exception as exc:
        send(b"ERR ", str(exc).encode())
        os._exit(1)
    send(b"RDY ", json.dumps({"rate": src.sample_rate, "gain": src.gain, "info": src.info}).encode())

    cmds = queue.Queue()

    def stdin_reader():
        for line in sys.stdin.buffer:
            cmds.put(line.decode().split())
        cmds.put(None)                       # the program closed us (or ended)
    threading.Thread(target=stdin_reader, daemon=True).start()

    chunk = 65536
    try:
        while True:
            try:
                while True:
                    c = cmds.get_nowait()
                    if c is None:
                        os._exit(0)          # no clean-up: closing the device is what crashes
                    if c and c[0] == "tune":
                        src.tune(float(c[1]))
                        send(b"ACK ", c[2].encode())
            except queue.Empty:
                pass
            send(b"IQ  ", src.read(chunk).tobytes())
    except (BrokenPipeError, OSError):
        os._exit(0)                          # the program is gone
    except Exception as exc:
        try:
            send(b"ERR ", str(exc).encode())
        except OSError:
            pass
        os._exit(1)


if __name__ == "__main__":
    _child(sys.argv[1:])
