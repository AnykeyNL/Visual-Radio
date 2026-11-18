"""Speech to text with faster-whisper (NVIDIA GPU, or CPU).

The radio's audio (48 kHz float) is cut into pieces of ~15 s at quiet moments
(ContinuousChunker); each piece is put on a queue and a background thread
splits it into talk and music, optionally takes the music out (Demucs), and
transcribes it, so the radio keeps running while Whisper works. Results come
back through `results()` as Transcript objects.
"""

from __future__ import annotations

import glob
import os
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
from scipy import signal

WHISPER_RATE = 16000

# Phrases Whisper tends to invent on noise or silence (subtitle credits etc.).
HALLUCINATIONS = {
    "bedankt voor het kijken", "bedankt voor het kijken!", "ondertiteling", "thank you.",
    "thanks for watching!", "thanks for watching.", "thank you for watching.", "you",
    "vielen dank.", "untertitel im auftrag des zdf, 2017", "untertitelung des zdf, 2020",
    "tv gelderland 2021", "...",
}

# Segments that are only music notes / punctuation, or "[Music]", "(muziek)" ...
_NOT_SPEECH = re.compile(r"[\W_♪♫]*|[\[(*][^\])*]*[\])*]")

# What Whisper writes when it hears music but no words: "Muziek", "MUSIC!", "Musik Musik" ...
MUSIC_WORDS = {
    "music", "muziek", "musik", "musique", "música", "musica", "muzyka", "musiikki", "musikk", "muzika",
    "muzică", "zene", "hudba", "glazba", "музыка", "музика", "μουσική", "müzik", "音楽", "음악", "音乐",
    "instrumental", "instrumentaal", "instrumentale", "instrumentaal", "instrumentell", "instrumentalmusik",
    "intro", "outro", "jingle", "achtergrondmuziek", "background", "playing", "plays", "upbeat", "soft",
    "gentle", "dramatic", "melancholic", "suspenseful", "and",
}


def music_placeholder(text):
    """True for text that only says music is playing ("Muziek", "[upbeat music]", "♪")."""
    if _NOT_SPEECH.fullmatch(text.strip()):
        return True
    words = re.findall(r"\w+", text.lower())
    return bool(words) and all(w in MUSIC_WORDS for w in words) and any(
        w not in {"and", "soft", "gentle", "playing", "plays", "background", "upbeat", "dramatic",
                  "melancholic", "suspenseful"} for w in words)


@dataclass
class Transcript:
    channel: int
    start: float        # wall clock (time.time()) when the transmission started
    duration_s: float
    text: str
    language: str = ""
    kind: str = ""      # "speech" / "music" when a content classifier is used
    words: list | None = None   # [(word, start s, end s)] after `start`, with word_times=True
    style: dict | None = None   # music only: {"scene", "confidence", "bpm", "beat" (abs time)}
    speakers: list | None = None   # speech with a speaker tracker: voice id (1, 2 ...) per word
    instrumental: bool = False     # music without words we could understand (text is "")


def _add_cuda_dll_dirs():
    """On Windows, make pip's nvidia-cublas/cudnn DLLs findable for CTranslate2."""
    if os.name != "nt":
        return
    for site in sys.path:
        for d in glob.glob(os.path.join(site, "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(d)
            except OSError:
                continue
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def load_model(model="auto", device="auto"):
    """-> (WhisperModel, description). Tries CUDA first, falls back to CPU."""
    _add_cuda_dll_dirs()
    try:
        from faster_whisper import WhisperModel
    except OSError as exc:  # DLL clash, usually an old vcomp140.dll in the conda env
        raise RuntimeError(
            f"{exc}. On Windows this is usually an old vcomp140.dll; fix with: "
            'conda install -c conda-forge "vc14_runtime>=14.40"') from exc

    tries = []
    if device in ("auto", "cuda"):
        tries.append(("cuda", "float16", "large-v3" if model == "auto" else model))
    if device in ("auto", "cpu"):
        tries.append(("cpu", "int8", "small" if model == "auto" else model))
    errors = []
    for dev, compute, name in tries:
        try:
            m = WhisperModel(name, device=dev, compute_type=compute)
            if dev == "cuda":
                # Missing cuBLAS/cuDNN often only shows up on the first real run.
                list(m.transcribe(np.zeros(WHISPER_RATE, dtype=np.float32), beam_size=1)[0])
            return m, f"{name} on {'GPU (CUDA)' if dev == 'cuda' else 'CPU'}"
        except Exception as exc:  # no GPU, missing DLLs, download failure, ...
            errors.append(f"{dev}: {exc}")
    raise RuntimeError("could not load Whisper: " + " | ".join(errors))


def trim_silence(audio48: np.ndarray) -> np.ndarray:
    """Cut the muted squelch head and tail (exact zeros) off a piece."""
    a = np.asarray(audio48, dtype=np.float32)
    nz = np.flatnonzero(np.abs(a) > 1e-6)
    return a[nz[0]:nz[-1] + 1] if len(nz) else np.zeros(0, dtype=np.float32)


def to_whisper_audio(audio48: np.ndarray, with_gain=False):
    """48 kHz -> 16 kHz, trim the muted squelch tail, normalise the level.
    with_gain: -> (audio, gain used to normalise)."""
    a = trim_silence(audio48)
    if len(a) == 0:
        return (a, 1.0) if with_gain else a
    a = signal.resample_poly(a, 1, 3).astype(np.float32)
    gain = 0.9 / (float(np.max(np.abs(a))) or 1.0)
    a = (a * gain).astype(np.float32)
    return (a, gain) if with_gain else a


class Transcriber:
    def __init__(self, model="auto", device="auto", language=None, min_s=0.6, backend=None,
                 classifier=None, content="both", word_times=False):
        """backend: an object with .transcribe(audio, **kw) like WhisperModel (for tests).

        classifier: a visualradio.content.SoundClassifier. With it, every piece is split
        into speech and music parts; content "speech" / "music" keeps only those,
        "both" keeps both. Each Transcript then says which kind it is.
        """
        self.classifier = classifier
        self.content = content
        self.word_times = word_times
        self.on_regions = None   # optional callback([(kind, t_start, t_end)]) per classified piece
        # optional fn(start, duration_s, regions) -> regions, to overrule the classifier
        # (e.g. RDS saying a song is on); times relative to start
        self.region_override = None
        # optional fn(16 kHz audio) -> [(start, end, voice id)]: who speaks when (speech only)
        self.speakers = None
        # optional fn(start time) -> language code or None: the language of the song
        # playing then (from song recognition); used for music instead of `language`
        self.music_language = None
        # optional fn(mono audio, rate, length) -> 16 kHz vocals (visualradio.vocals.VocalIsolator):
        # Whisper and the speaker colours then hear the voice without the music under it.
        # isolate: which pieces get it, "off" / "talk" / "songs" / "all"
        self.isolator = None
        self.isolate = "off"
        # how long each step takes (for the load panel on the page), see timing_info()
        self._steps = {}           # step -> [average s per piece, last time it ran]
        self._piece = {}           # this piece: step -> s
        self._piece_s = 15.0       # average length of a piece (s)
        self.busy_since = None     # when the piece being worked on was started
        if backend is None:
            backend, self.description = load_model(model, device)
        else:
            self.description = "test backend"
        self.model = backend
        self.language = None if language in (None, "", "auto") else language   # None: Whisper detects it
        self.min_s = min_s
        self.jobs: queue.Queue = queue.Queue()
        self.done: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, channel, start, audio48):
        self.jobs.put((channel, start, audio48))

    def results(self):
        out = []
        while True:
            try:
                out.append(self.done.get_nowait())
            except queue.Empty:
                return out

    def close(self, timeout=30.0):
        """Finish what is queued (up to timeout s), then stop the thread."""
        self.jobs.put(None)
        self.thread.join(timeout)

    def transcribe_all(self, channel, start, audio48) -> list:
        """One piece of audio -> list of Transcripts (split by speech/music if classifying)."""
        if self.classifier is None:
            t = self.transcribe(channel, start, audio48)
            return [t] if t else []
        a, gain = to_whisper_audio(audio48, with_gain=True)
        out = []
        regions = self._timed("classify", self.classifier.regions, a)
        if self.region_override:
            regions = self.region_override(start, len(a) / WHISPER_RATE, regions)
        if self.on_regions:
            self.on_regions([(kind, start + t0, start + t1) for kind, t0, t1 in regions])
        keep = [(k, t0, t1) for k, t0, t1 in regions
                if k is not None and (self.content == "both" or k == self.content)]
        clean = None              # the voice alone, for the whole piece (more context than per part)
        if any(self._isolate_for(k) for k, _, _ in keep):
            clean = self._timed("isolate", self._vocals, audio48, len(a), gain)
        for kind, t0, t1 in keep:
            src = clean if clean is not None and self._isolate_for(kind) else a
            seg = src[int(t0 * WHISPER_RATE):int(t1 * WHISPER_RATE)]
            t = self._timed("whisper", self._whisper, channel, start + t0, seg, kind)
            if t and kind != "music" and self.speakers is not None:
                self._timed("speakers", self._add_speakers, t, seg)
            if t:
                if kind == "music" and hasattr(self.classifier, "style"):
                    try:
                        st = self.classifier.style(a, t0, t1)
                        if "beat" in st:
                            st["beat"] = round(start + t0 + st["beat"], 3)
                        t.style = st
                    except Exception:   # the look is a nice-to-have
                        pass
                out.append(t)
        return out

    def transcribe(self, channel, start, audio48) -> Transcript | None:
        a, gain = to_whisper_audio(audio48, with_gain=True)
        if self._isolate_for(""):
            v = self._timed("isolate", self._vocals, audio48, len(a), gain)
            a = a if v is None else v
        t = self._timed("whisper", self._whisper, channel, start, a)
        if t and self.speakers is not None:
            self._timed("speakers", self._add_speakers, t, a)
        return t

    def _isolate_for(self, kind):
        if self.isolator is None or self.isolate in (None, "off"):
            return False
        from .vocals import wanted
        return wanted(self.isolate, kind)

    def _vocals(self, audio48, length, gain):
        """The voice of this piece without the music (16 kHz, like to_whisper_audio),
        or None when isolation fails: then Whisper just gets the normal audio."""
        if length == 0:
            return None
        try:
            from .vocals import match_level
            return match_level(self.isolator(trim_silence(audio48), 48000, length), gain)
        except Exception as exc:
            if not getattr(self, "_isolate_error", None):
                self._isolate_error = str(exc)
                print(f"\r\x1b[KVoice isolation failed, using the normal audio: {exc}")
            return None

    def _add_speakers(self, t, a):
        """Mark every word with the voice that said it (needs self.speakers and word times)."""
        if self.speakers is None or not t.words:
            return
        try:
            from .speakers import assign_words
            t.speakers = assign_words(t.words, self.speakers(a))
        except Exception as exc:          # colours are a nice-to-have, the text is not
            if not getattr(self, "_speaker_error", None):
                self._speaker_error = str(exc)
                print(f"\r\x1b[KSpeaker detection failed: {exc}")

    def _whisper(self, channel, start, a, kind="") -> Transcript | None:
        dur = len(a) / WHISPER_RATE
        if dur < self.min_s:
            return None
        music = kind == "music"
        lang = self.language
        if music and self.music_language:
            try:
                lang = self.music_language(start) or lang
            except Exception:
                pass
        # Silero VAD (vad_filter) tends to throw away sung vocals, so not for music.
        segments, info = self.model.transcribe(
            a, language=lang, beam_size=5, vad_filter=not music,
            condition_on_previous_text=False, without_timestamps=not self.word_times,
            word_timestamps=self.word_times)
        limit = 0.8 if music else 0.6
        parts, words = [], []
        for seg in segments:
            p = seg.text.strip()
            if (not p or getattr(seg, "no_speech_prob", 0.0) >= limit
                    or p.lower() in HALLUCINATIONS or music_placeholder(p)):
                continue
            parts.append(p)
            for w in getattr(seg, "words", None) or []:
                if w.word.strip():
                    words.append((w.word.strip(), round(float(w.start), 2), round(float(w.end), 2)))
        text = " ".join(parts).strip()
        if not text:
            if music:      # music, but no words: the page shows an instrumental visual instead
                return Transcript(channel, start, dur, "", getattr(info, "language", "") or "", kind,
                                  instrumental=True)
            return None
        return Transcript(channel, start, dur, text, getattr(info, "language", "") or "", kind,
                          words if self.word_times and words else None)

    def _timed(self, step, fn, *args):
        t0 = time.perf_counter()
        try:
            return fn(*args)
        finally:
            self._piece[step] = self._piece.get(step, 0.0) + time.perf_counter() - t0

    def _finish_piece(self, seconds):
        now = time.time()
        if seconds > 0:
            self._piece_s += 0.2 * (seconds - self._piece_s)
        for step, s in self._piece.items():
            old = self._steps.get(step)
            self._steps[step] = [s if old is None or now - old[1] > 60 else old[0] + 0.3 * (s - old[0]), now]
        self._piece = {}

    def timing_info(self):
        """-> {"piece": s, "steps": {"whisper": s, "isolate": s, "speakers": s}, "backlog": n,
        "busy": s}: average seconds per piece of each step that ran in the last minute."""
        now = time.time()
        steps = {k: round(v[0], 3) for k, v in list(self._steps.items()) if now - v[1] < 60}
        busy = self.busy_since
        return {"piece": round(self._piece_s, 1), "steps": steps, "backlog": self.jobs.qsize(),
                "busy": round(now - busy, 1) if busy else 0.0}

    def _run(self):
        while True:
            job = self.jobs.get()
            if job is None:
                return
            self.busy_since = time.time()
            self._piece = {}
            try:
                res = self.transcribe_all(*job)
            except Exception as exc:
                res = [Transcript(job[0], job[1], 0.0, f"[transcription failed: {exc}]")]
            self.busy_since = None
            self._finish_piece(len(job[2]) / 48000 if job[2] is not None else 0)
            for t in res:
                self.done.put(t)


class ContinuousChunker:
    """Cuts a never-ending audio stream (broadcast radio) into pieces for Whisper.

    Pieces are about `chunk_s` long. To avoid cutting through a word, each piece
    ends at the quietest 100 ms in its last third; the rest carries over to the
    next piece. on_done(label, start_time, audio) gets each piece.
    """

    def __init__(self, on_done, label, chunk_s=15.0, rate=48000):
        self.on_done = on_done
        self.label = label
        self.rate = rate
        self.chunk = int(chunk_s * rate)
        self.blocks, self.length = [], 0
        self.start = None

    def add(self, audio, now=None):
        now = time.time() if now is None else now
        if self.start is None:
            self.start = now - len(audio) / self.rate     # when this block's audio began
        self.blocks.append(audio.copy())
        self.length += len(audio)
        if self.length < self.chunk:
            return
        a = np.concatenate(self.blocks)
        win = self.rate // 10
        lo = (2 * self.chunk) // 3
        seg = a[lo:lo + ((len(a) - lo) // win) * win]
        energy = (seg.reshape(-1, win) ** 2).mean(axis=1)
        cut = lo + int(np.argmin(energy)) * win + win // 2
        self.on_done(self.label, self.start, a[:cut])
        rest = a[cut:]
        self.start = now - len(rest) / self.rate
        self.blocks, self.length = [rest], len(rest)

    def flush(self):
        if self.length > 0:
            self.on_done(self.label, self.start, np.concatenate(self.blocks))
        self.blocks, self.length, self.start = [], 0, None
