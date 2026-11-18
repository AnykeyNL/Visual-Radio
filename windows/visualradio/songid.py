"""Which song is playing? RDS radio text first, AcoustID fingerprint second.

1. RDS: many stations send "Artist - Title" in their radio text. That costs
   nothing and needs no key, so it is tried first.
2. AcoustID: Chromaprint's `fpcalc` makes an audio fingerprint, and the free
   AcoustID web service (https://acoustid.org) looks it up in MusicBrainz.
   AcoustID matches whole recordings from their start, not random excerpts,
   so the fingerprint is taken from the moment a song begins (the speech /
   music classifier says when talk turns into music), first after 40 s and,
   if that finds nothing, again after 100 s. AcoustID also only matches songs
   whose length is close to the length sent, which is unknown for a song on
   the radio, so one batched request asks for 2:00 - 6:00 in four steps.
3. Album art: Cover Art Archive (by MusicBrainz release group), otherwise the
   iTunes search API. Images and results are cached in cache/ next to the scripts.

Needs an AcoustID *application* API key (free) for step 2: --acoustid-key or
the ACOUSTID_KEY environment variable. Without a key, fpcalc or internet the
rest keeps working with RDS only.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from collections import deque

import numpy as np
from scipy import signal

from .rds import _Decimator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
UA = "VisualRadio/1.0 (personal SDR radio tool)"
RATE = 16000
# AcoustID compares the given length with the song's length (+- maxdurationdiff,
# at most 30 s). Together these cover songs of 2:00 to 6:00.
DURATION_GUESSES = (150, 210, 270, 330)

RT_PREFIX = re.compile(r"^\s*(now playing|now on air|on air|you're listening to|nu|nu op [^:]*|"
                       r"nu te horen|jetzt läuft|en ce moment|np|playing)\s*[:\-–]\s*", re.I)


def parse_radiotext(rt: str):
    """'Now playing: Artist - Title' -> ('Artist', 'Title'), else None."""
    if not rt:
        return None
    s = RT_PREFIX.sub("", rt.strip())
    s = re.split(r"\s+[|•·]\s+|\s{3,}", s)[0]          # drop "· call the studio ..." tails
    m = re.match(r"^(.{2,60}?)\s+[-–—/]\s+(.{2,80}?)\s*$", s)
    if m:
        return m.group(1).strip(" ."), m.group(2).strip(" .")
    m = re.match(r"^(.{2,80}?)\s+by\s+(.{2,60})$", s, re.I)   # not "van"/"de": too common in Dutch
    if m:
        return m.group(2).strip(" ."), m.group(1).strip(" .")
    return None


def rds_song(rt, station=""):
    """(artist, title) when the radio text names a song, else None. Text that names
    the station itself ("Radio 2 - the best hits") is a slogan, not a song."""
    parsed = parse_radiotext(rt)
    st = re.sub(r"\W", "", station or "").lower()
    if parsed and st and any(st in re.sub(r"\W", "", p).lower() for p in parsed):
        return None
    return parsed


class RdsKind:
    """Music or talk, as RDS says: music while the radio text names a song."""

    def __init__(self):
        self.events = []         # (time, "music" | "speech"), only changes
        self.lock = threading.Lock()

    def reset(self):
        with self.lock:
            self.events = []

    def update(self, rt, when, station=""):
        if not rt:                # radio text not received this time: no news
            return
        kind = "music" if rds_song(rt, station) else "speech"
        with self.lock:
            if not self.events or self.events[-1][1] != kind:
                self.events.append((when, kind))
            while len(self.events) > 2 and self.events[1][0] < when - 900:
                self.events.pop(0)

    def regions(self, start, duration, regions):
        """Transcriber.region_override: replace the classifier's split by the RDS one.
        Before RDS has said anything, the classifier's regions are kept."""
        with self.lock:
            events = list(self.events)
        before = [k for t, k in events if t <= start]
        if not before:
            return regions
        out, cur, t0 = [], before[-1], 0.0
        for t, k in events:
            if start < t < start + duration and k != cur:
                out.append((cur, t0, t - start))
                cur, t0 = k, t - start
        out.append((cur, t0, duration))
        return [r for r in out if r[2] - r[1] > 0.3]


def find_fpcalc(explicit=None):
    for cand in (explicit, os.environ.get("FPCALC"),
                 os.path.join(ROOT, "tools", "fpcalc.exe"), os.path.join(ROOT, "tools", "fpcalc"),
                 shutil.which("fpcalc")):
        if cand and os.path.isfile(cand):
            return cand
    return None


class AcoustIdError(RuntimeError):
    def __init__(self, code, message):
        self.code = code
        hint = ""
        if code in (4, 5, 6):
            hint = (" - use the *application* API key from https://acoustid.org/new-application "
                    "(not the user API key on your profile page)")
        super().__init__(f"AcoustID: {message}{hint}")


# ISO 639-3 (as MusicBrainz stores it) -> Whisper's language codes
LANG3 = {
    "eng": "en", "nld": "nl", "deu": "de", "fra": "fr", "spa": "es", "ita": "it", "por": "pt", "swe": "sv",
    "nor": "no", "nob": "no", "nno": "nn", "dan": "da", "fin": "fi", "isl": "is", "pol": "pl", "ces": "cs",
    "slk": "sk", "slv": "sl", "hrv": "hr", "srp": "sr", "bos": "bs", "hun": "hu", "ron": "ro", "bul": "bg",
    "rus": "ru", "ukr": "uk", "bel": "be", "ell": "el", "tur": "tr", "ara": "ar", "heb": "he", "fas": "fa",
    "hin": "hi", "urd": "ur", "ben": "bn", "tam": "ta", "tel": "te", "jpn": "ja", "kor": "ko", "zho": "zh",
    "cmn": "zh", "yue": "yue", "tha": "th", "vie": "vi", "ind": "id", "msa": "ms", "tgl": "tl", "afr": "af",
    "cat": "ca", "eus": "eu", "glg": "gl", "cym": "cy", "est": "et", "lav": "lv", "lit": "lt", "mkd": "mk",
    "sqi": "sq", "hye": "hy", "kat": "ka", "aze": "az", "kaz": "kk", "swa": "sw", "yor": "yo", "ltz": "lb",
    "oci": "oc", "bre": "br", "lat": "la", "hat": "ht", "mri": "mi", "haw": "haw", "som": "so", "amh": "am",
}
LANG_NAMES = {"en": "English", "nl": "Dutch", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian",
              "pt": "Portuguese", "sv": "Swedish", "no": "Norwegian", "da": "Danish", "fi": "Finnish", "pl": "Polish"}


def _get(url, timeout=10, data=None):
    req = urllib.request.Request(url, data=data, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), r.headers.get("Content-Type", "")


class SongIdentifier:
    def __init__(self, on_song, api_key=None, fpcalc=None, cache_dir=CACHE, log=print,
                 sources=("rds", "acoustid"), interval=15.0):
        """sources: which ways to identify songs, "rds" and/or "acoustid".
        interval: during music, ask AcoustID every `interval` s of music (rolling
        window), so back-to-back songs and songs joined half way are picked up.
        0: only near the start of a song (after 40 s and 100 s)."""
        self.interval = float(interval or 0)
        self.last_fp = None                  # run_end at the last rolling lookup
        self.on_song = on_song
        self.use_rds = "rds" in sources
        self.use_acoustid = "acoustid" in sources
        self.key = api_key or os.environ.get("ACOUSTID_KEY") or None
        self.fpcalc = find_fpcalc(fpcalc)
        self.cache_dir = cache_dir
        self.cover_dir = os.path.join(cache_dir, "covers")
        os.makedirs(self.cover_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, "songs.json")
        try:
            with open(self.cache_file, encoding="utf-8") as f:
                self.cache = json.load(f)
        except (OSError, ValueError):
            self.cache = {}
        self.log = log
        self.audio = deque()                 # (t_start, 16 kHz block)
        self.dec = _Decimator(signal.firwin(63, 7000, fs=48000), 3)
        self.lock = threading.Lock()
        self.current = None                  # (artist, title) shown now
        self.run_start = None                # when the current stretch of music began
        self.run_end = None
        self.tries = []                      # seconds-into-the-song at which AcoustID was asked
        self.found = False                   # AcoustID already knows the current song
        self.last_rt = None
        self.text_source = "RDS"             # where radio text comes from: "RDS" (FM) or "DAB+" (DLS)
        self.find_language = True            # look up the song's language (MusicBrainz)
        self.song_lang = None                # (language, since) of the identified song
        self.last_request = 0.0
        self.simple = False                  # True: the service refused batched lookups
        self.jobs = deque()
        self.wake = threading.Event()
        self.stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def reset(self):
        """Another station: forget the song, the audio and the radio text."""
        with self.lock:
            self.audio.clear()
            self.jobs.clear()
            self.run_start = self.run_end = self.last_fp = None
            self.tries, self.found = [], False
        self.current = None
        self.last_rt = None
        self.song_lang = None
        self.gen = getattr(self, "gen", 0) + 1     # a lookup still running for the old station is dropped

    def acoustid_ready(self):
        """AcoustID could be used (key and fpcalc present), whether it is switched on or not."""
        return bool(self.key and self.fpcalc)

    def set_sources(self, sources):
        """Switch RDS and/or AcoustID on or off while running (from the web page)."""
        use_rds, use_acoustid = "rds" in sources, "acoustid" in sources
        if use_rds and not self.use_rds:
            self.last_rt = None                  # pick up the radio text that is on now
        if use_acoustid and not self.use_acoustid:
            with self.lock:
                self.last_fp, self.tries, self.found = None, [], False
        self.use_rds, self.use_acoustid = use_rds, use_acoustid
        if not use_acoustid:
            with self.lock:
                self.jobs = deque(j for j in self.jobs if j[0] != "fp")
        if not use_rds:
            with self.lock:
                self.jobs = deque(j for j in self.jobs if j[0] != "rds")

    def status(self):
        parts = ["RDS"] if self.use_rds else []
        if not self.use_acoustid:
            pass
        elif self.key and self.fpcalc:
            parts.append("AcoustID")
        elif not self.fpcalc:
            parts.append("AcoustID off: fpcalc not found (put fpcalc.exe in tools\\)")
        else:
            parts.append("AcoustID off: no API key (--acoustid-key, free at acoustid.org)")
        return ", ".join(parts) or "off"

    # ------------------------------------------------------------ inputs

    def add_audio(self, audio48, captured):
        if len(audio48) % 3:
            return
        a = self.dec(audio48.astype(np.float32))
        with self.lock:
            self.audio.append((captured, a))
            while self.audio and self.audio[0][0] < captured - 150:
                self.audio.popleft()

    def regions(self, regions):
        """From the classifier: [(kind, t0, t1)] of one transcribed piece."""
        with self.lock:
            for kind, t0, t1 in regions:
                if kind == "music":
                    if self.run_start is None or (self.run_end is not None and t0 - self.run_end > 8):
                        self.run_start, self.tries, self.found = t0, [], False   # a new song
                        self.last_fp = None
                    self.run_end = t1
                elif kind == "speech" and t1 - t0 > 6 and self.run_end is not None and t0 > self.run_end:
                    self.run_start = self.run_end = None             # talk: the song is over
                if kind == "speech" and t1 - t0 > 6 and self.song_lang and t0 > self.song_lang[1] + 20:
                    self.song_lang = None                            # its language too
            if self.run_start is None or not (self.use_acoustid and self.key and self.fpcalc):
                return
            length = self.run_end - self.run_start
            if self.interval:
                if any(j[0] == "fp" for j in self.jobs):          # previous one still waiting
                    return
                if length >= 20 and (self.last_fp is None or self.run_end - self.last_fp >= self.interval - 1):
                    self.last_fp = self.run_end
                    t0 = max(self.run_start, self.run_end - 120)  # the song start while we have it
                    self.jobs.append(("fp", t0, self.run_end - t0))
                    self.wake.set()
                return
            if self.found:
                return
            for at in (40, 100):
                if length >= at and at not in self.tries:
                    self.tries.append(at)
                    self.jobs.append(("fp", self.run_start, min(length, 120)))
                    self.wake.set()
                    break

    def radiotext(self, rt, when, station=""):
        """RDS radio text. Text that names the station itself ("Radio 2 - the best hits")
        is a slogan, not a song."""
        if self.use_rds and rt and rt != self.last_rt:
            self.last_rt = rt
            parsed = rds_song(rt, station)
            if parsed:
                self.jobs.append(("rds", parsed, when))
                self.wake.set()

    # ------------------------------------------------------------ worker

    def _run(self):
        while not self.stop.is_set():
            self.wake.wait(1.0)
            self.wake.clear()
            while self.jobs:
                job = self.jobs.popleft()
                self._job_gen = getattr(self, "gen", 0)
                try:
                    if job[0] == "rds":
                        (artist, title), when = job[1], job[2]
                        self._publish({"artist": artist, "title": title, "source": self.text_source}, when)
                    else:
                        self._fingerprint(job[1], job[2])
                except AcoustIdError as exc:
                    self.log(f"song lookup failed: {exc}")
                    if exc.code in (4, 5, 6):          # bad key: no point asking again
                        self.key = None
                        self.log("AcoustID is off for this run; RDS still works.")
                except Exception as exc:    # no internet, service down, ... keep going
                    self.log(f"song lookup failed: {exc}")

    def _fingerprint(self, t0, seconds):
        with self.lock:
            parts = [(t, a) for t, a in self.audio if t >= t0 - 0.2 and t < t0 + seconds]
        if not parts:
            self.log("AcoustID: no audio buffered for this song")
            return
        audio = np.concatenate([a for _, a in parts])
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(RATE)
                w.writeframes(pcm.tobytes())
            out = subprocess.run([self.fpcalc, "-json", "-length", "120", path], capture_output=True,
                                 text=True, timeout=60,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            fp = json.loads(out.stdout)
        finally:
            os.remove(path)
        song = self.lookup(fp["fingerprint"])
        if song:
            with self.lock:
                if self.run_start == t0:
                    self.found = True
            self._publish(song, t0)
        else:
            self.log(f"AcoustID: no match for {seconds:.0f} s of music from "
                     f"{time.strftime('%H:%M:%S', time.localtime(t0))}")

    def lookup(self, fingerprint):
        """AcoustID: fingerprint -> {artist, title, album, rgid, score} or None."""
        form = {"client": self.key, "format": "json", "meta": "recordings releasegroups compress"}
        if not self.simple:
            batch = dict(form, batch=1, maxdurationdiff=30)
            for i, d in enumerate(DURATION_GUESSES):
                batch[f"fingerprint.{i}"] = fingerprint
                batch[f"duration.{i}"] = d
            try:
                return self.parse_lookup(self._post(batch))
            except AcoustIdError as exc:
                if exc.code in (4, 5, 6):     # invalid / user key / unknown application
                    raise
                # the service didn't take the batched request: ask one length at a time
                self.log(f"AcoustID: batched lookup refused ({exc}), using single lookups")
                self.simple = True
        for d in DURATION_GUESSES:
            song = self.parse_lookup(self._post(dict(form, duration=d, fingerprint=fingerprint)))
            if song:
                return song
        return None

    def _post(self, form):
        wait = 0.4 - (time.time() - self.last_request)   # AcoustID allows 3 requests/s
        if wait > 0:
            time.sleep(wait)
        self.last_request = time.time()
        try:
            raw, _ = _get("https://api.acoustid.org/v2/lookup", timeout=20,
                          data=urllib.parse.urlencode(form).encode())
        except urllib.error.HTTPError as exc:          # AcoustID explains a 400 in its JSON body
            try:
                raw = exc.read()
                json.loads(raw)
            except Exception:
                raise RuntimeError(f"AcoustID: HTTP {exc.code} {exc.reason}") from None
        return json.loads(raw)

    # ------------------------------------------------------------ song language (MusicBrainz)

    _mb_last = 0.0

    def _mb(self, path, **params):
        """MusicBrainz web service, at most one request a second (their rule)."""
        wait = SongIdentifier._mb_last + 1.1 - time.time()
        if wait > 0:
            time.sleep(wait)
        SongIdentifier._mb_last = time.time()
        q = urllib.parse.urlencode(dict(params, fmt="json"))
        raw, _ = _get(f"https://musicbrainz.org/ws/2/{path}?{q}", timeout=15)
        return json.loads(raw)

    @staticmethod
    def pick_language(recording):
        """MusicBrainz recording (with work-rels + releases) -> Whisper code, "" for no
        lyrics (instrumental), None when unknown. The work's language first (that is
        the language of the lyrics), else what most releases say."""
        works = set()
        for rel in recording.get("relations") or []:
            w = rel.get("work") or {}
            langs = w.get("languages") or ([w["language"]] if w.get("language") else [])
            if len(langs) == 1:
                works.add(langs[0])
        if len(works) == 1:
            code = works.pop()
            if code == "zxx":
                return ""
            if code in LANG3:
                return LANG3[code]
        count = {}
        for r in recording.get("releases") or []:
            code = (r.get("text-representation") or {}).get("language")
            if code in LANG3:
                count[LANG3[code]] = count.get(LANG3[code], 0) + 1
        return max(count, key=count.get) if count else None

    def language(self, song):
        """-> Whisper language code of the song's lyrics, or None. Cached."""
        ck = f"{song['artist']}|{song['title']}".lower()
        entry = self.cache.setdefault(ck, {})
        if "language" in entry:
            return entry["language"]
        lang = None
        try:
            mbid = song.get("mbid")
            if not mbid:
                clean = lambda t: re.sub(r'["\\]', "", t)
                res = self._mb("recording", limit=5,
                               query=f'recording:"{clean(song["title"])}" AND artist:"{clean(song["artist"])}"')
                hits = [r for r in res.get("recordings") or [] if int(r.get("score", 0)) >= 85]
                mbid = hits[0]["id"] if hits else None
            if mbid:
                lang = self.pick_language(self._mb(f"recording/{mbid}", inc="work-rels+releases"))
        except Exception as exc:
            self.log(f"song language lookup failed: {exc}")
            return None                       # not cached: try again next time
        entry["language"] = lang
        self._save_cache()
        return lang

    def language_at(self, t):
        """Transcriber.music_language: the identified song's language for music heard at t."""
        cur = self.song_lang
        if cur and cur[0] and t >= cur[1] - 10:
            return cur[0]
        return None

    def _save_cache(self):
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=1)
        except OSError:
            pass

    @staticmethod
    def parse_lookup(data):
        if data.get("status") != "ok":
            err = data.get("error") or {}
            raise AcoustIdError(err.get("code"), err.get("message", "unknown error"))
        results = list(data.get("results") or [])
        for fp in data.get("fingerprints") or []:            # batch answer
            results += fp.get("results") or []
        for res in sorted(results, key=lambda r: -r.get("score", 0)):
            if res.get("score", 0) < 0.5:
                break
            for rec in res.get("recordings", []) or []:
                if not rec.get("title"):
                    continue
                artist = "".join(a["name"] + a.get("joinphrase", "") for a in rec.get("artists", []))
                rgs = rec.get("releasegroups") or []
                rg = next((g for g in rgs if g.get("type") == "Album"), rgs[0] if rgs else {})
                return {"artist": artist or "Unknown artist", "title": rec["title"],
                        "album": rg.get("title"), "rgid": rg.get("id"), "mbid": rec.get("id"),
                        "score": round(res["score"], 2), "source": "AcoustID"}
        return None

    # ------------------------------------------------------------ output

    def _publish(self, song, when):
        key = (song["artist"].lower(), song["title"].lower())
        if key == self.current or getattr(self, "_job_gen", 0) != getattr(self, "gen", 0):
            return
        self.current = key
        song["cover"] = self.cover(song)
        song["time"] = when
        if self.find_language:
            song["language"] = self.language(song)
            self.song_lang = (song["language"], when) if song["language"] is not None else None
        song.pop("mbid", None)
        if getattr(self, "_job_gen", 0) != getattr(self, "gen", 0):
            return                       # the station changed while the cover / language was looked up
        self.on_song(song)

    def cover(self, song):
        """-> file name in cache/covers, or None. Cached, also when nothing was found."""
        ck = f"{song['artist']}|{song['title']}".lower()
        if "cover" in self.cache.get(ck, {}):
            for k in ("album", "genre", "year"):
                if self.cache[ck].get(k) and not song.get(k):
                    song[k] = self.cache[ck][k]
            name = self.cache[ck].get("cover")
            return name if name and os.path.exists(os.path.join(self.cover_dir, name)) else None
        name = None
        urls = []
        if song.get("rgid"):
            urls.append(f"https://coverartarchive.org/release-group/{song['rgid']}/front-500")
        try:
            q = urllib.parse.urlencode({"term": f"{song['artist']} {song['title']}", "entity": "song", "limit": 1})
            raw, _ = _get(f"https://itunes.apple.com/search?{q}")
            hits = json.loads(raw).get("results", [])
            if hits and hits[0].get("artworkUrl100"):
                urls.append(hits[0]["artworkUrl100"].replace("100x100bb", "600x600bb"))
                if not song.get("album"):
                    song["album"] = hits[0].get("collectionName")
            if hits:
                song["genre"] = hits[0].get("primaryGenreName")
                song["year"] = (hits[0].get("releaseDate") or "")[:4] or None
        except Exception:
            pass
        for url in urls:
            try:
                img, ctype = _get(url, timeout=15)
                if len(img) > 1000:
                    name = hashlib.sha1(ck.encode()).hexdigest()[:16] + (".png" if "png" in ctype else ".jpg")
                    with open(os.path.join(self.cover_dir, name), "wb") as f:
                        f.write(img)
                    break
            except Exception:
                continue
        self.cache.setdefault(ck, {}).update({"cover": name, "album": song.get("album"), "genre": song.get("genre"),
                                             "year": song.get("year")})
        self._save_cache()
        return name
