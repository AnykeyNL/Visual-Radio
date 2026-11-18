"""Live web page for visual_radio.py --web.

Standard library only: a threaded HTTP server that serves one self-contained
page (visualradio/web_page.html, no internet needed) and pushes updates to every
open browser with Server-Sent Events:

  station     {band, key, freq, name, rt, since}   FM: frequency, RDS name, RDS radio text;
              DAB+ also {channel, sid, mhz, ensemble, snr, slide}: rt is the DLS text,
              slide the URL of the slideshow picture (/slide?v=n)
  level       {db, bands[], pending, now}   ~10x per second: audio level + spectrum
  transcript  {id, time, kind, lang, text, duration, words: [[word, start, end]], speakers?: [voice per word], instrumental?: true}
  song        {artist, title, album, cover, source, score, time}   (cover: /cover/<file>)
  stations    {scanned, stations: [{freq, name, snr, rt}], control,
               dab: {scanned, ensembles: [{channel, mhz, label, snr, services: [{sid, name}]}], ready}}
  scan        {phase, done, total, mhz, band, channel}   while a band is being scanned (phase "done" at the end)
  notice      {text, error}   something to tell whoever is looking (e.g. DAB+ could not start)
  detect      {talk: sound|rds, songs: rds|acoustid|both|off, language: auto|nl|..., acoustid, song_lookup,
               isolate: off|talk|songs|all, isolate_ready, control}
  sys         {cpu, ram, proc, gpu: {name, util, mem_used, mem_total, temp, power}, piece, steps, backlog}

With control enabled (visual_radio.py --web-control) the page may also
  POST /tune  {"freq": 100.7}     change the station
              {"band": "dab", "channel": "12C", "sid": "0x8201"}   a DAB+ station
  POST /scan  {"band": "fm"|"dab"}   scan that band again (and save the list)
  POST /detect {"talk", "songs", "language", "isolate"}  talk/music, song source, spoken language,
               voice isolation
Without it those answer 403.

All times are the PC's clock (seconds) at the moment the audio was received.
With a playback delay the page shows each word `delay` seconds after that,
which is when it comes out of the speakers. The page can also play the
(delayed) audio itself: /audio streams it as binary frames
  [float64 time][uint32 n | kind << 24][n x int16 @ 24 kHz]
and the browser schedules each frame at time + delay, so every device plays
in step with the PC and with the text. Kind 0 is the radio itself; /audio?stems=1
streams kind 1 (the voice only) and kind 2 (the music only) from Demucs, with the
same times, for as long as a page asks for them (see visualradio.stems).
"""

from __future__ import annotations

import collections
import json
import re
import os
import queue
import socket
import struct
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from scipy import signal

from .rds import _Decimator

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_page.html")
COVERS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "covers")
N_BANDS = 48


def lan_ip() -> str:
    """This PC's address on the local network (no packets are actually sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.168.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class Spectrum:
    """Audio block -> N_BANDS log-spaced band levels 0..1 (60 Hz - 14 kHz)."""

    def __init__(self, rate=48000, nfft=4096):
        self.nfft = nfft
        self.win = np.hanning(nfft).astype(np.float32)
        f = np.fft.rfftfreq(nfft, 1 / rate)
        edges = np.geomspace(60, 14000, N_BANDS + 1)
        self.idx = [np.flatnonzero((f >= lo) & (f < hi)) for lo, hi in zip(edges[:-1], edges[1:])]
        self.idx = [i if len(i) else np.array([np.searchsorted(f, lo)])
                    for i, lo in zip(self.idx, edges[:-1])]

    def __call__(self, audio):
        a = audio[-self.nfft:]
        if len(a) < self.nfft:
            a = np.pad(a, (self.nfft - len(a), 0))
        p = np.abs(np.fft.rfft(a * self.win)) ** 2
        db = np.array([10 * np.log10(p[i].mean() + 1e-12) for i in self.idx])
        # tilt: music falls off ~ 3-4 dB/octave, lift the top so the bars look even
        db += np.linspace(0, 24, N_BANDS)
        return np.clip((db + 5) / 45, 0, 1).round(3).tolist()


class LiveWeb:
    def __init__(self, port=8080, history=150, delay=0.0, cover_dir=COVERS):
        self.delay = float(delay)
        self.cover_dir = cover_dir
        self.song = None
        self.detection = None        # see set_detection
        self.sys = None              # see set_sys
        self.sys_hist = collections.deque(maxlen=120)   # [cpu, proc, gpu, vram %] per sample
        self.control = None          # fn(command dict) -> (ok, message); None: read-only page
        self.stations = {"scanned": None, "stations": []}
        self.dab = {"scanned": None, "ensembles": [], "ready": False}
        self.slide = None            # (bytes, content type): the DAB+ slideshow picture
        self.scan_state = None
        self.clients: set[queue.Queue] = set()
        self.audio_clients: set[queue.Queue] = set()
        # recent audio (as frames) so a new listener starts right away, not after `delay` s
        self.audio_backlog = deque(maxlen=int((max(self.delay, 2.0) + 2) * 10))
        self.to24k = _Decimator(signal.firwin(63, 10500, fs=48000), 2)
        self.stem_clients: set[queue.Queue] = set()
        # voice + music frames (4 + 4 a second) still to be played: a second page joins in at once
        self.stem_backlog = deque(maxlen=int((max(self.delay, 5.0) + 10) * 8))
        self.stem_dec = {}                           # kind -> _Decimator
        self.lock = threading.Lock()
        self.history = deque(maxlen=history)
        self.station = {"band": "fm", "key": "", "freq": None, "name": "", "rt": "", "since": time.time()}
        self.info = {}
        self.next_id = 1
        self.started = time.time()
        self.spectrum = Spectrum()
        self._last_level = 0.0
        web = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):   # keep the console clean
                pass

            def do_GET(self):
                path = self.path.split("?")[0]
                if path in ("/", "/index.html"):
                    with open(PAGE, "rb") as f:
                        body = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(body)
                elif path == "/events":
                    web._serve_events(self)
                elif path == "/audio":
                    web._serve_audio(self, stems="stems=1" in self.path)
                elif path == "/slide":
                    web._serve_slide(self)
                elif path.startswith("/cover/"):
                    web._serve_cover(self, path[7:])
                elif path == "/state":
                    body = json.dumps(web.snapshot()).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif path == "/stations":
                    web._json(self, 200, web.stations_info())
                else:
                    self.send_error(404)

            def do_POST(self):
                path = self.path.split("?")[0]
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                    body = json.loads(self.rfile.read(min(n, 10000)) or b"{}") if n else {}
                except (ValueError, OSError):
                    return web._json(self, 400, {"ok": False, "message": "bad request"})
                if web.control is None:
                    return web._json(self, 403, {"ok": False, "message":
                                                 "Changing stations is off (start with --web-control)"})
                if path == "/tune" and body.get("band") == "dab":
                    from .dab import CHANNELS
                    ch, sid = str(body.get("channel") or "").upper(), str(body.get("sid") or "").lower()
                    if ch not in CHANNELS or not re.fullmatch(r"0x[0-9a-f]{4,8}", sid):
                        return web._json(self, 400, {"ok": False, "message": "channel (5A-13F) and sid (0x....) needed"})
                    ok, msg = web.control({"cmd": "tune", "band": "dab", "channel": ch, "sid": sid})
                elif path == "/tune":
                    try:
                        mhz = round(float(body.get("freq")), 1)
                    except (TypeError, ValueError):
                        return web._json(self, 400, {"ok": False, "message": "freq missing"})
                    if not 87.5 <= mhz <= 108.0:
                        return web._json(self, 400, {"ok": False, "message": "87.5 - 108 MHz only"})
                    ok, msg = web.control({"cmd": "tune", "freq": mhz})
                elif path == "/scan":
                    ok, msg = web.control({"cmd": "scan", "band": "dab" if body.get("band") == "dab" else "fm"})
                elif path == "/detect":
                    talk, songs, lang = body.get("talk"), body.get("songs"), body.get("language")
                    iso = body.get("isolate")
                    if (talk not in (None, "sound", "rds") or songs not in (None, "rds", "acoustid", "both")
                            or iso not in (None, "off", "talk", "songs", "all")
                            or not (lang is None or lang == "auto" or re.fullmatch(r"[a-z]{2,3}", str(lang)))):
                        return web._json(self, 400, {"ok": False, "message":
                                                     "talk: sound|rds, songs: rds|acoustid|both, language: auto|nl|en|..., "
                                                     "isolate: off|talk|songs|all"})
                    if iso not in (None, "off") and not (web.detection or {}).get("isolate_ready"):
                        return web._json(self, 409, {"ok": False, "message":
                                                     "Voice isolation needs Demucs: pip install demucs"})
                    if songs == "acoustid" or songs == "both":
                        if not (web.detection or {}).get("acoustid"):
                            return web._json(self, 409, {"ok": False, "message":
                                                         "AcoustID needs --acoustid-key (free at acoustid.org)"})
                    ok, msg = web.control({"cmd": "detect", "talk": talk, "songs": songs, "language": lang,
                                           "isolate": iso})
                else:
                    return web._json(self, 404, {"ok": False, "message": "unknown"})
                web._json(self, 200 if ok else 409, {"ok": ok, "message": msg})

        self.httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    # ------------------------------------------------------------ publishing

    def urls(self):
        return [f"http://localhost:{self.port}", f"http://{lan_ip()}:{self.port}"]

    def snapshot(self):
        return {"station": self.station, "info": self.info, "history": list(self.history),
                "song": self.song, "stations": self.stations_info(), "scan": self.scan_state,
                "detect": self.detection_info(), "sys": self.sys,
                "sys_hist": list(self.sys_hist),
                "started": self.started, "delay": self.delay, "now": time.time()}

    def _send(self, event, data):
        msg = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()
        with self.lock:
            for q in list(self.clients):
                if q.qsize() < 200:          # a stuck browser doesn't grow memory forever
                    q.put(msg)

    def set_info(self, **info):
        """Static facts for the footer: model, language, content mode ..."""
        self.info.update(info)
        self._send("info", self.info)

    def set_sys(self, sample):
        """CPU / GPU load and time per step (visualradio.sysload), about once a second."""
        self.sys = sample
        g = sample.get("gpu") or {}
        vram = 100 * g["mem_used"] / g["mem_total"] if g.get("mem_total") else None
        self.sys_hist.append([sample.get("cpu"), sample.get("proc"), g.get("util"),
                              None if vram is None else round(vram, 1)])
        self._send("sys", sample)

    def set_detection(self, info):
        """What decides talk/music and where songs come from:
        {"talk": "sound"|"rds", "songs": "rds"|"acoustid"|"both"|"off", "acoustid": bool, "song_lookup": bool}"""
        self.detection = dict(info)
        self._send("detect", self.detection_info())

    def detection_info(self):
        return dict(self.detection or {}, control=self.control is not None) if self.detection else None

    def stations_info(self):
        return dict(self.stations, control=self.control is not None, dab=self.dab)

    def set_dab_stations(self, data, ready=True):
        """The saved DAB+ list (visualradio.dab format); ready: DAB+ can be used (welle-cli found)."""
        self.dab = {"scanned": data.get("scanned"), "ensembles": list(data.get("ensembles") or []),
                    "ready": bool(ready)}
        self._send("stations", self.stations_info())

    def set_slide(self, body, ctype, version):
        """A new DAB+ slideshow picture for the station."""
        self.slide = (body, ctype or "image/jpeg")
        self.set_station(slide=f"/slide?v={version}")

    def notice(self, text, error=False):
        self._send("notice", {"text": text, "error": bool(error)})

    def set_stations(self, data):
        """The saved station list (visualradio.stations format)."""
        self.stations = {"scanned": data.get("scanned"), "stations": list(data.get("stations") or [])}
        self._send("stations", self.stations_info())

    def scan_progress(self, phase, done=0, total=0, mhz=None, band="fm", channel=None):
        st = {"phase": phase, "done": done, "total": total, "mhz": mhz, "band": band, "channel": channel}
        self.scan_state = None if phase == "done" else st
        self._send("scan", st)

    @staticmethod
    def _json(h, code, data):
        body = json.dumps(data).encode()
        h.send_response(code)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(body)))
        h.send_header("Cache-Control", "no-cache")
        h.end_headers()
        h.wfile.write(body)

    def set_station(self, freq_hz=None, name=None, rt=None, retuned=False, band=None, **dab):
        """retuned: the receiver moved to another station just now; the page switches
        its header when that moment is played (time + delay).
        band "dab" with channel, sid, mhz (and later ensemble, snr, slide)."""
        changed = False
        if retuned:
            self.station = {"band": band or self.station.get("band", "fm"), "key": "", "freq": None,
                            "name": "", "rt": "", "since": time.time()}
            self.song = None             # a page that opens now must not show the old station's song
            self.slide = None
            changed = True
        elif band and band != self.station.get("band"):
            self.station["band"] = band
            changed = True
        for k, v in (("freq", None if freq_hz is None else round(freq_hz / 1e6, 2)),
                     ("name", name), ("rt", rt)) + tuple(dab.items()):
            if v is not None and v != self.station.get(k):
                self.station[k] = v
                changed = True
        st = self.station
        key = f"dab:{st.get('channel')}:{st.get('sid')}" if st.get("band") == "dab" else f"fm:{st.get('freq')}"
        if key != st.get("key"):
            st["key"] = key
            changed = True
        if changed:
            self._send("station", self.station)

    def set_song(self, song):
        """A recognised song: {artist, title, album, cover (file in cover_dir), source, time, language}."""
        from .style import itunes_scene
        item = {k: song.get(k) for k in ("artist", "title", "album", "source", "score", "time", "genre", "year",
                                          "language")}
        item["scene"] = itunes_scene(song.get("genre"))
        item["cover"] = f"/cover/{song['cover']}" if song.get("cover") else None
        self.song = item
        self._send("song", item)

    def audio(self, block, pending=0, captured=None):
        """Call with every 48 kHz audio block (captured = time its first sample was
        received). Streams it to listening browsers, and sends level + spectrum
        about 10x a second."""
        now = time.time()
        if captured is None:
            captured = now - len(block) / 48000
        if len(block) % 2 == 0:
            pcm = np.clip(self.to24k(block.astype(np.float32)) * 32767, -32768, 32767).astype("<i2")
            frame = struct.pack("<dI", captured, len(pcm)) + pcm.tobytes()
            with self.lock:
                self.audio_backlog.append(frame)
                for q in list(self.audio_clients):
                    if q.qsize() < 600:
                        q.put(frame)
        if now - self._last_level < 0.09:
            return
        self._last_level = now
        db = 20 * np.log10(float(np.sqrt(np.mean(block ** 2))) + 1e-9)
        self._send("level", {"db": round(db, 1), "bands": self.spectrum(block), "pending": pending,
                             "now": round(now, 3), "captured": round(captured, 3)})

    def stem_audio(self, kind, times, blocks):
        """The voice (kind 1) or the music (kind 2) of the radio, from the stem
        splitter: 48 kHz blocks, each with the time its first sample was received."""
        dec = self.stem_dec.get(kind)
        if dec is None:
            dec = self.stem_dec[kind] = _Decimator(signal.firwin(63, 10500, fs=48000), 2)
        frames = []
        for t, block in zip(times, blocks):
            if len(block) % 2:
                block = block[:-1]
            pcm = np.clip(dec(np.asarray(block, dtype=np.float32)) * 32767, -32768, 32767).astype("<i2")
            frames.append(struct.pack("<dI", t, len(pcm) | (kind << 24)) + pcm.tobytes())
        data = b"".join(frames)
        with self.lock:
            self.stem_backlog.extend(frames)
            for q in list(self.stem_clients):
                if q.qsize() < 200:
                    q.put(data)

    def stems_wanted(self):
        """True while some page listens to the voice or the music only."""
        return bool(self.stem_clients)

    def stems_reset(self):
        """The stems start afresh (another station): drop what is buffered."""
        with self.lock:
            self.stem_backlog.clear()

    def transcript(self, t):
        item = {"id": self.next_id, "time": t.start, "kind": t.kind or "speech",
                "lang": t.language, "text": t.text, "duration": round(t.duration_s, 1),
                "words": [[w[0], round(t.start + w[1], 2), round(t.start + (w[2] if len(w) > 2 else w[1] + 0.3), 2)]
                          for w in (getattr(t, "words", None) or [])] or None,
                "style": getattr(t, "style", None)}
        if getattr(t, "instrumental", False):
            item["instrumental"] = True          # music without words: the page shows a visual
        spk = getattr(t, "speakers", None)
        if spk and item["words"] and len(spk) == len(item["words"]):
            item["speakers"] = list(spk)          # voice id per word (1, 2 ...; null: unknown)
        self.next_id += 1
        self.history.append(item)
        self._send("transcript", item)

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    # ------------------------------------------------------------ SSE

    def _serve_slide(self, h):
        pic = self.slide
        if not pic:
            h.send_error(404)
            return
        h.send_response(200)
        h.send_header("Content-Type", pic[1])
        h.send_header("Content-Length", str(len(pic[0])))
        h.send_header("Cache-Control", "max-age=3600")
        h.end_headers()
        h.wfile.write(pic[0])

    def _serve_cover(self, h, name):
        name = os.path.basename(name)
        path = os.path.join(self.cover_dir, name)
        if not name.lower().endswith((".jpg", ".png")) or not os.path.isfile(path):
            h.send_error(404)
            return
        with open(path, "rb") as f:
            body = f.read()
        h.send_response(200)
        h.send_header("Content-Type", "image/png" if name.lower().endswith(".png") else "image/jpeg")
        h.send_header("Content-Length", str(len(body)))
        h.send_header("Cache-Control", "max-age=86400")
        h.end_headers()
        h.wfile.write(body)

    def _serve_audio(self, h, stems=False):
        q: queue.Queue = queue.Queue()
        clients, backlog_src = (self.stem_clients, self.stem_backlog) if stems else \
            (self.audio_clients, self.audio_backlog)
        h.send_response(200)
        h.send_header("Content-Type", "application/octet-stream")
        h.send_header("Cache-Control", "no-cache")
        h.send_header("Connection", "close")
        h.end_headers()
        with self.lock:
            backlog = list(backlog_src)
            clients.add(q)
        if stems:            # only what is still to be played
            old = time.time() - self.delay - 3
            backlog = [f for f in backlog if struct.unpack_from("<d", f)[0] >= old]
        try:
            h.wfile.write(b"".join(backlog))
            h.wfile.flush()
            while True:
                try:
                    h.wfile.write(q.get(timeout=3 if stems else 5))
                    h.wfile.flush()
                except queue.Empty:
                    if stems:        # an empty frame: notices a page that left while Demucs warms up
                        h.wfile.write(struct.pack("<dI", 0.0, 3 << 24))
                        h.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        finally:
            with self.lock:
                clients.discard(q)

    def _serve_events(self, h):
        q: queue.Queue = queue.Queue()
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.send_header("Cache-Control", "no-cache")
        h.send_header("Connection", "keep-alive")
        h.send_header("X-Accel-Buffering", "no")
        h.end_headers()
        try:
            h.wfile.write(f"retry: 2000\nevent: hello\ndata: "
                          f"{json.dumps(self.snapshot(), ensure_ascii=False)}\n\n".encode())
            h.wfile.flush()
            with self.lock:
                self.clients.add(q)
            while True:
                try:
                    msg = q.get(timeout=15)
                except queue.Empty:
                    msg = b": ping\n\n"
                h.wfile.write(msg)
                h.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        finally:
            with self.lock:
                self.clients.discard(q)
