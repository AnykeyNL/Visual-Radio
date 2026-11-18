"""DAB+ digital radio through welle-cli (the command line receiver of welle.io).

welle-cli runs in the background with the Airspy and a small web server on a
local port (only this PC can reach it). This module asks it for:

  /mux.json        the ensemble on the current channel: its services (stations),
                   their DLS text (DAB's radio text), slideshow and the signal quality
  POST /channel    another channel (5A ... 13F)
  /pcm/<sid>       a station's decoded audio as raw PCM: frames of
                   b"PCM1" + uint32 rate + uint32 nbytes + nbytes of int16 stereo
                   (added to welle-cli for this program; the normal build only has MP3)
  /slide/<sid>     the latest slideshow picture

The Airspy can only be used by one program at a time, so visual_radio.py
lets go of it before welle-cli starts, and takes it back for FM.

stations_dab.json keeps the scanned ensembles:
  {"scanned": <unix time>, "ensembles": [{"channel": "12C", "mhz": 227.36, "label": "NPO",
   "id": "0x8001", "snr": 14.0, "services": [{"sid": "0x8daa", "name": "NPO Radio 1"}]}]}
"""

from __future__ import annotations

import http.client
import json
import math
import os
import queue
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
import zlib
from collections import deque

import numpy as np
from scipy import signal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FILE = os.path.join(ROOT, "stations_dab.json")
AUDIO_RATE = 48000

# Band III, EN 300 401 annex. The "N" channels are hardly used; a scan skips them.
CHANNELS = {
    "5A": 174.928, "5B": 176.640, "5C": 178.352, "5D": 180.064,
    "6A": 181.936, "6B": 183.648, "6C": 185.360, "6D": 187.072,
    "7A": 188.928, "7B": 190.640, "7C": 192.352, "7D": 194.064,
    "8A": 195.936, "8B": 197.648, "8C": 199.360, "8D": 201.072,
    "9A": 202.928, "9B": 204.640, "9C": 206.352, "9D": 208.064,
    "10A": 209.936, "10N": 210.096, "10B": 211.648, "10C": 213.360, "10D": 215.072,
    "11A": 216.928, "11N": 217.088, "11B": 218.640, "11C": 220.352, "11D": 222.064,
    "12A": 223.936, "12N": 224.096, "12B": 225.648, "12C": 227.360, "12D": 229.072,
    "13A": 230.784, "13B": 232.496, "13C": 234.208, "13D": 235.776, "13E": 237.488, "13F": 239.200,
}
SCAN_CHANNELS = [c for c in CHANNELS if not c.endswith("N")]
SETTLE = 3.0          # seconds to wait for an ensemble before a channel counts as empty

WELLE_HELP = ("DAB+ needs welle-cli: put welle-cli.exe in the tools folder "
              "(see README, 'DAB+ digital radio').")


class DabError(RuntimeError):
    pass


def find_welle(path=None):
    """The welle-cli program: --welle, tools\\welle-cli.exe, or on the PATH. None if missing."""
    for c in ([path] if path else []) + [os.path.join(ROOT, "tools", "welle-cli.exe"),
                                         os.path.join(ROOT, "tools", "welle-cli", "welle-cli.exe"),
                                         os.path.join(ROOT, "tools", "welle-cli")]:
        if c and os.path.isfile(c):
            return c
    return shutil.which("welle-cli")


def clean(label):
    """DAB labels are padded to 16 characters."""
    return " ".join(str(label or "").split())


def norm_sid(sid):
    """0x8DAA / 8daa / 36266 -> "0x8daa" (welle-cli's spelling)."""
    s = str(sid).strip().lower()
    try:
        n = int(s, 16) if s.startswith("0x") or any(c in "abcdef" for c in s) else int(s)
    except ValueError:
        return s
    return f"0x{n:04x}"


def load(path=DEFAULT_FILE):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data.get("ensembles"), list):
            return data
    except (OSError, ValueError, AttributeError):
        pass
    return None


def save(data, path=DEFAULT_FILE):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def find_service(data, name_or_sid, channel=None):
    """Look a station up in the saved list by name (or part of it) or service id.
    -> (channel, sid, name) or None."""
    import re
    want = " ".join(str(name_or_sid or "").lower().split())
    if not want:
        return None
    word = re.compile(r"(?<![\w%])" + re.escape(want) + r"(?![\w%])")
    hits = []
    for ens in (data or {}).get("ensembles", []):
        if channel and ens["channel"] != channel:
            continue
        for s in ens.get("services", []):
            name = s.get("name", "")
            low = " ".join(name.lower().split())
            if norm_sid(want) == s["sid"] or want == low:
                return ens["channel"], s["sid"], name
            if want in low:      # "radio 1": NPO Radio 1 before Radio 10
                hits.append((0 if word.search(low) else 1, len(low), (ens["channel"], s["sid"], name)))
    return min(hits)[2] if hits else None


def audio_services(mux):
    """[{sid, name}] of the stations (audio services) in a mux.json answer."""
    out = []
    for s in (mux or {}).get("services") or []:
        if not s.get("url_mp3"):
            continue
        lab = s.get("label") or {}
        out.append({"sid": norm_sid(s.get("sid")), "name": clean(lab.get("label")),
                    "short": clean(lab.get("shortlabel"))})
    return out


def ensemble_label(mux):
    return clean(((mux or {}).get("ensemble") or {}).get("label", {}).get("label"))


# ------------------------------------------------------------------ audio

class Resampler:
    """Streaming rate change to 48 kHz for DAB's 16 / 24 / 32 / 48 kHz audio."""

    def __init__(self, rate_in, rate_out=AUDIO_RATE):
        g = math.gcd(int(rate_in), int(rate_out))
        self.up, self.down = rate_out // g, rate_in // g
        self.h = signal.firwin(24 * max(self.up, self.down) + 1, 0.9 / max(self.up, self.down)) * self.up
        self.zi = np.zeros(len(self.h) - 1)
        self.phase = 0

    def __call__(self, x):
        if self.up == self.down == 1:
            return x.astype(np.float32)
        y = np.zeros(len(x) * self.up)
        y[::self.up] = x
        y, self.zi = signal.lfilter(self.h, 1.0, y, zi=self.zi)
        out = y[self.phase::self.down]
        self.phase = (self.phase - len(y)) % self.down
        return out.astype(np.float32)


def read_frames(sock, stop=lambda: False):
    """PCM1 frames from a /pcm/<sid> answer (headers included) -> (rate, int16 stereo)."""
    buf = b""

    def need(n):
        nonlocal buf
        while len(buf) < n:
            if stop():
                raise EOFError
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                raise EOFError
            buf += chunk
        out, buf = buf[:n], buf[n:]
        return out

    while b"\r\n\r\n" not in buf:
        if stop():
            return
        try:
            chunk = sock.recv(65536)
        except socket.timeout:
            continue
        if not chunk:
            return
        buf += chunk
    head, buf = buf.split(b"\r\n\r\n", 1)
    if b" 200" not in head.split(b"\r\n", 1)[0]:
        raise DabError(head.split(b"\r\n", 1)[0].decode(errors="replace"))
    try:
        while True:
            magic = need(4)
            if magic != b"PCM1":
                raise DabError("unexpected data from welle-cli (is it the version with /pcm?)")
            rate, n = struct.unpack("<II", need(8))
            yield rate, np.frombuffer(need(n), dtype="<i2")
    except EOFError:
        return


# ------------------------------------------------------------------ welle-cli

class WelleApi:
    """welle-cli's web interface on 127.0.0.1:port."""

    port = 0

    def _http(self, method, path, body=None, timeout=4.0):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            c.request(method, path, body=body)
            r = c.getresponse()
            return r.status, r.read(), r.getheader("Content-Type") or ""
        finally:
            c.close()

    def mux(self):
        status, body, _ = self._http("GET", "/mux.json")
        if status != 200:
            raise DabError(f"mux.json: HTTP {status}")
        return json.loads(body.decode("utf-8", errors="replace"))

    def set_channel(self, channel):
        self._http("POST", "/channel", body=channel.encode(), timeout=15.0)
        self.channel = channel

    def slide(self, sid):
        try:
            status, body, ctype = self._http("GET", f"/slide/{sid}")
        except OSError:
            return None
        return (body, ctype) if status == 200 and body else None

    def open_pcm(self, sid):
        s = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        s.sendall(f"GET /pcm/{sid} HTTP/1.0\r\nHost: localhost\r\n\r\n".encode())
        s.settimeout(1.0)
        return s


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class WelleCli(WelleApi):
    """welle-cli.exe in the background on one channel."""

    def __init__(self, exe, channel, device="airspy", gain=-1, iq_file=None, log=print, start_timeout=20.0):
        self.exe, self.log = exe, log
        self.channel = channel            # the channel welle-cli is on now
        self.port = _free_port()
        cmd = [exe, "-c", channel, "-w", str(self.port), "-T"]       # -T: no transmitter ids (saves CPU)
        cmd += ["-f", iq_file] if iq_file else ["-F", device]
        if gain is not None and not iq_file:
            cmd += ["-g", str(int(gain))]                            # -1: automatic gain
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        self.lines = deque(maxlen=60)
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         cwd=os.path.dirname(os.path.abspath(exe)), creationflags=flags)
        except OSError as exc:
            raise DabError(f"could not start {exe} ({exc})") from exc
        threading.Thread(target=self._pump, daemon=True).start()
        t0 = time.time()
        while True:
            if self.proc.poll() is not None:
                time.sleep(0.2)
                raise DabError(self._why())
            try:
                self.mux()
                return
            except (OSError, DabError, ValueError):
                pass
            if time.time() - t0 > start_timeout:
                self.close()
                raise DabError("welle-cli did not answer. " + self._why())
            time.sleep(0.25)

    def _pump(self):
        for raw in iter(self.proc.stdout.readline, b""):
            self.lines.append(raw.decode(errors="replace").rstrip())

    def _why(self):
        text = "\n".join(self.lines)
        if "airspy_set_samplerate() failed" in text:
            return ("the Airspy refused 4.096 MSPS, the rate welle-cli uses for DAB+. Older Airspy "
                    "firmware only knows its fixed rates: update it (airspy_spiflash, see README).")
        if "Could not start device" in text or "No valid device found" in text or "airspy_open" in text:
            return ("welle-cli could not open the Airspy. Is it plugged in and not used by another "
                    "program (SDR#, another copy of this script)?")
        tail = [ln for ln in self.lines if ln.strip()][-4:]
        return "welle-cli stopped: " + (" | ".join(tail) or f"exit code {self.proc.returncode}")

    def alive(self):
        return self.proc.poll() is None

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(5)


# ------------------------------------------------------------------ the receiver

class DabRadio:
    """One DAB+ station at a time: audio (48 kHz mono, like FmAudio gives), its DLS
    text and slideshow. Switching stations within the ensemble is instant; another
    channel takes a second or two while welle-cli finds the new ensemble.

    on_info(dict) is called when what is known about the station changes:
      {"channel", "mhz", "ensemble", "name", "dls", "snr", "slide": version or 0}"""

    def __init__(self, welle, channel, sid, log=print, on_info=None):
        self.welle = welle
        self.log = log
        self.on_info = on_info
        self.lock = threading.Lock()
        self.target = (channel, norm_sid(sid))
        self.gen = 0
        # the channel welle-cli is on now (after a scan that is not where it started)
        self.welle_channel = getattr(welle, "channel", None) or channel
        self.sock = None
        self.q = queue.Queue(maxsize=400)
        self.info = {}
        self.slide = None                 # (bytes, content type) of the newest picture
        self.slide_ver = 0
        self._mot = None
        self.scanning = False
        self.stop = threading.Event()
        self.last_audio = time.time()
        self._rest = None                 # (time, samples) left over from the last read
        self.services = []                # stations seen in the current ensemble
        threading.Thread(target=self._audio_loop, daemon=True).start()
        threading.Thread(target=self._meta_loop, daemon=True).start()

    # -------------------------------------------------- control

    def tune(self, channel, sid):
        with self.lock:
            self.target = (channel, norm_sid(sid))
            self.gen += 1
            self._drop_stream()
            self.info, self.slide, self._mot = {}, None, None
        self._clear()
        self._rest = None

    def _drop_stream(self):
        s, self.sock = self.sock, None
        if s:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            s.close()

    def _clear(self):
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                return

    def read(self, timeout=0.3):
        """Audio that arrived since the last call (48 kHz mono float32) and the time
        its first sample was received. Waits up to `timeout` s for some."""
        parts, first = [], None
        try:
            t, a = self.q.get(timeout=timeout)
            parts.append(a)
            first = t
            while True:
                t, a = self.q.get_nowait()
                parts.append(a)
        except queue.Empty:
            pass
        if self._rest is not None:            # keep blocks a multiple of 6 samples (the 24 / 16 kHz
            parts.insert(0, self._rest[1])      # decimators downstream want that)
            first = self._rest[0]
            self._rest = None
        if not parts:
            return np.zeros(0, np.float32), time.time()
        a = np.concatenate(parts)
        cut = len(a) - len(a) % 6
        if cut < len(a):
            self._rest = (first + cut / AUDIO_RATE, a[cut:])
        return a[:cut], first

    def close(self):
        self.stop.set()
        with self.lock:
            self._drop_stream()

    # -------------------------------------------------- workers

    def _wait_service(self, sid, gen, timeout=20.0):
        t0 = time.time()
        while time.time() - t0 < timeout and gen == self.gen and not self.stop.is_set():
            try:
                if any(s["sid"] == sid for s in audio_services(self.welle.mux())):
                    return True
            except (OSError, DabError, ValueError):
                pass
            time.sleep(0.3)
        return False

    def _audio_loop(self):
        missing = None
        while not self.stop.is_set():
            if self.scanning:
                time.sleep(0.2)
                continue
            with self.lock:
                (channel, sid), gen = self.target, self.gen
            try:
                if self.welle_channel != channel:
                    self.welle.set_channel(channel)
                    self.welle_channel = channel
                if not self._wait_service(sid, gen):
                    if gen == self.gen and missing != (channel, sid):
                        missing = (channel, sid)
                        self.log(f"DAB+: station {sid} not (yet) found on channel {channel}, still trying")
                    continue
                missing = None
                sock = self.welle.open_pcm(sid)
                with self.lock:
                    if gen != self.gen:
                        sock.close()
                        continue
                    self.sock = sock
                rs, rate = None, None
                for r, pcm in read_frames(sock, stop=lambda: gen != self.gen or self.stop.is_set()):
                    if gen != self.gen:
                        break
                    if r != rate:
                        rate, rs = r, Resampler(r)
                    mono = pcm.astype(np.float32).reshape(-1, 2).mean(axis=1) / 32768.0
                    a = rs(mono)
                    now = time.time()
                    self.last_audio = now
                    try:
                        self.q.put_nowait((now - len(a) / AUDIO_RATE, a))
                    except queue.Full:
                        pass
            except (OSError, DabError, ValueError) as exc:
                if gen == self.gen and not self.stop.is_set() and " 503 " not in str(exc):
                    self.log(f"DAB+ audio: {exc}")
            finally:
                with self.lock:
                    if gen == self.gen:
                        self._drop_stream()
            if not self.stop.is_set():
                time.sleep(0.5)

    def _meta_loop(self):
        while not self.stop.wait(1.0):
            if self.scanning:
                continue
            with self.lock:
                (channel, sid), gen = self.target, self.gen
            try:
                m = self.welle.mux()
            except (OSError, DabError, ValueError):
                continue
            if self.welle_channel != channel:
                continue
            self.services = audio_services(m)
            svc = next((s for s in m.get("services") or [] if norm_sid(s.get("sid")) == sid), None)
            info = {"channel": channel, "mhz": CHANNELS.get(channel), "ensemble": ensemble_label(m),
                    "snr": (m.get("demodulator") or {}).get("snr"), "sid": sid}
            if svc:
                info["name"] = clean((svc.get("label") or {}).get("label"))
                info["dls"] = clean((svc.get("dls") or {}).get("label"))
                mot = (svc.get("mot") or {}).get("lastchange") or 0
                if mot and mot != self._mot:
                    pic = self.welle.slide(sid)
                    if pic and gen == self.gen:
                        self._mot = mot
                        self.slide = pic
                        self.slide_ver += 1
            info["slide"] = self.slide_ver if self.slide else 0
            if gen != self.gen:
                continue
            if info != self.info:
                self.info = info
                if self.on_info:
                    try:
                        self.on_info(dict(info))
                    except Exception as exc:
                        self.log(f"DAB+ info: {exc}")

    # -------------------------------------------------- scan

    def scan(self, channels=SCAN_CHANNELS, progress=None, settle=None, max_wait=12.0):
        """Visit every channel, note the ensembles and their stations.
        progress(done, total, channel, ensembles_so_far). -> stations_dab.json data."""
        self.scanning = True
        with self.lock:
            self.gen += 1
            self._drop_stream()
        found = []
        try:
            for i, ch in enumerate(channels):
                if self.stop.is_set():
                    break
                if progress:
                    progress(i, len(channels), ch, found)
                ens = scan_channel(self.welle, ch, settle, max_wait)
                self.welle_channel = ch
                if ens:
                    found.append(ens)
                    self.log(f"DAB+ {ch}: {ens['label'] or '(no name)'}, "
                             f"{len(ens['services'])} station(s)")
            if progress:
                progress(len(channels), len(channels), None, found)
        finally:
            self.scanning = False
            with self.lock:
                self.gen += 1                       # the audio loop tunes back to the target
        return {"scanned": time.time(), "ensembles": found}


def scan_channel(welle, ch, settle=None, max_wait=12.0):
    """-> {channel, mhz, label, id, snr, services} or None when nothing is on ch."""
    settle = SETTLE if settle is None else settle
    welle.set_channel(ch)
    t0, last, steady, m = time.time(), None, time.time(), {}
    while time.time() - t0 < max_wait:
        time.sleep(0.4)
        try:
            m = welle.mux()
        except (OSError, DabError, ValueError):
            continue
        label, svcs = ensemble_label(m), audio_services(m)
        if not label and not svcs:
            if time.time() - t0 > settle:
                return None
            continue
        sig = (label, tuple((s["sid"], s["name"]) for s in svcs))
        if sig != last:
            last, steady = sig, time.time()
        elif svcs and all(s["name"] for s in svcs) and time.time() - steady >= 2.0:
            break
    svcs = [s for s in audio_services(m) if s["name"]]
    if not svcs:
        return None
    return {"channel": ch, "mhz": CHANNELS.get(ch), "label": ensemble_label(m),
            "id": ((m.get("ensemble") or {}).get("id") or ""), "snr": (m.get("demodulator") or {}).get("snr"),
            "services": sorted(svcs, key=lambda s: s["name"].lower())}


# ------------------------------------------------------------------ simulator

SIM_ENSEMBLES = {
    "8A": ("Lokaal Mux", 11.0, [("0x8701", "Radio Rijnmond"), ("0x8702", "Sublime"), ("0x8703", "Arrow Classic Rock")]),
    "11C": ("Commercial NL", 17.5, [("0x8501", "Radio 538"), ("0x8502", "Qmusic"), ("0x8503", "Sky Radio"),
                                    ("0x8504", "Radio 10"), ("0x8505", "100%NL")]),
    "12C": ("NPO", 21.0, [("0x8201", "NPO Radio 1"), ("0x8202", "NPO Radio 2"), ("0x8203", "NPO 3FM"),
                          ("0x8204", "NPO Klassiek"), ("0x8205", "NPO Soul & Jazz")]),
}
SIM_DLS = ["Coldplay - Viva La Vida", "Nu: Bruno Mars - Uptown Funk", "Het nieuws van 12:00",
           "Adele - Rolling In The Deep", "Luister ook via de app", "Golden Earring - Radar Love"]


def _png(w, h, rgb_fn):
    rows = b"".join(b"\x00" + bytes(np.clip(rgb_fn(y), 0, 255).astype(np.uint8).tobytes()) for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows, 6)) + chunk(b"IEND", b""))


def sim_slide(sid, version):
    """A 320x240 picture in the station's colours (a stand-in for a real slideshow)."""
    n = int(sid, 16) + version * 7
    a = np.array([(n * 97) % 255, (n * 57) % 255, (n * 23) % 255], float)
    b = np.array([255 - a[1], a[2], 255 - a[0]], float)
    xs = np.linspace(0, 1, 320)[:, None]

    def row(y):
        t = (xs + y / 240) / 2
        glow = np.exp(-(((xs - .7) ** 2) + ((y / 240 - .35) ** 2)) * 9)[:, :1] * 120
        return a * (1 - t) + b * t + glow
    return _png(320, 240, row)


class SimWelle(WelleApi):
    """A fake welle-cli (same web interface) for --simulate and the tests: a few
    ensembles, stations playing tones and noisy 'talk', DLS that changes, slides."""

    def __init__(self, ensembles=None, lock_time=0.5):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        self.ensembles = ensembles or SIM_ENSEMBLES
        self.channel = None
        self.tuned_at = time.time()
        self.lock_time = lock_time
        self.started = time.time()
        sim = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *a):
                pass

            def _send(self, body, ctype):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/mux.json":
                    self._send(json.dumps(sim._mux()).encode(), "application/json")
                elif self.path == "/channel":
                    self._send((sim.channel or "").encode(), "text/plain")
                elif self.path.startswith("/slide/"):
                    sid = self.path[7:]
                    if sim._svc(sid):
                        self._send(sim_slide(sid, sim._mot(sid)), "image/png")
                    else:
                        self.send_error(404)
                elif self.path.startswith("/pcm/"):
                    sim._stream(self, self.path[5:])
                else:
                    self.send_error(404)

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                ch = self.rfile.read(n).decode().strip()
                if ch in CHANNELS:
                    sim.channel, sim.tuned_at = ch, time.time()
                self._send(b"Retuning...", "text/plain")

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def start(self, channel):
        self.channel, self.tuned_at = channel, time.time()
        return self

    def _locked(self):
        return self.channel in self.ensembles and time.time() - self.tuned_at >= self.lock_time

    def _svc(self, sid):
        if not self._locked():
            return None
        return next((s for s in self.ensembles[self.channel][2] if s[0] == norm_sid(sid)), None)

    def _mot(self, sid):
        return int((time.time() - self.started) // 40)

    def _mux(self):
        out = {"ensemble": {"label": {"label": "", "shortlabel": ""}, "id": "0x0000"}, "services": [],
               "demodulator": {"snr": 0.0}}
        if not self._locked():
            return out
        label, snr, svcs = self.ensembles[self.channel]
        out["ensemble"] = {"label": {"label": label.ljust(16), "shortlabel": label[:8]}, "id": "0x8fff"}
        out["demodulator"]["snr"] = snr
        k = int((time.time() - self.started) // 20)
        for i, (sid, name) in enumerate(svcs):
            out["services"].append({
                "sid": sid, "label": {"label": name.ljust(16), "shortlabel": name[:8]}, "url_mp3": f"/mp3/{sid}",
                "dls": {"label": SIM_DLS[(k + i) % len(SIM_DLS)], "lastchange": k},
                "mot": {"lastchange": self._mot(sid) + 1}})
        return out

    def _stream(self, h, sid):
        svc = self._svc(sid)
        if not svc:
            h.send_error(404)
            return
        h.send_response(200)
        h.send_header("Content-Type", "application/octet-stream")
        h.end_headers()
        n = int(norm_sid(sid), 16)
        rng = np.random.default_rng(n)
        rate, frame = (48000, 2304) if n % 2 else (32000, 1536)       # both rates DAB+ uses
        notes = 220 * 2 ** (np.array([0, 4, 7, 12]) / 12) * (1 + (n % 5) / 10)
        t0, sent, clock = time.time(), 0, 0
        try:
            while self.channel and self._svc(sid):
                t = (clock + np.arange(frame)) / rate
                clock += frame
                if int(t[0] // 8) % 2:                      # 8 s "music", 8 s "talk"
                    x = sum(np.sin(2 * np.pi * f * t) for f in notes) * 0.12
                else:
                    env = (np.sin(2 * np.pi * 3.1 * t) > -.2) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t))
                    x = signal.lfilter([1], [1, -1.6, .8], rng.standard_normal(frame)) * 0.05 * env
                pcm = (np.clip(np.repeat(x[:, None], 2, axis=1), -1, 1) * 32000).astype("<i2").tobytes()
                h.wfile.write(b"PCM1" + struct.pack("<II", rate, len(pcm)) + pcm)
                h.wfile.flush()
                sent += frame
                ahead = t0 + sent / rate - time.time()
                if ahead > 0:
                    time.sleep(ahead)
        except OSError:
            pass

    def alive(self):
        return True

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
