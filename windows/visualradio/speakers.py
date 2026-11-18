"""Who is speaking? Local speaker diarization with pyannote.audio, so the page can
give every voice its own colour. No names: voices are numbered 1, 2, 3 ... in the
order they are first heard, and the same voice keeps its number across pieces.

How it works:
- each speech piece (~15 s) goes through pyannote's diarization pipeline, which
  says who speaks when *within that piece* (SPEAKER_00, SPEAKER_01 ...) and
  gives one voice embedding per local speaker;
- VoiceBook matches those embeddings (cosine similarity) against the voices heard
  before, so SPEAKER_00 of this piece becomes "voice 3" if it sounds like voice 3;
- assign_words gives every Whisper word the voice that was speaking at that time.

Everything runs on this PC. Only the first start downloads the model from
Hugging Face (needs a free token and accepting the model's terms, see README).
"""

from __future__ import annotations

import os

import numpy as np

DEFAULT_MODELS = {3: "pyannote/speaker-diarization-3.1", 4: "pyannote/speaker-diarization-community-1"}

# ANSI 256-colour codes close to the page's voice colours (console output)
ANSI = (80, 117, 183, 217, 221, 156, 216, 218)


class VoiceBook:
    """Remembers the voices heard so far as (normalised) mean embeddings."""

    def __init__(self, threshold=0.45, max_voices=64, min_seconds=1.0):
        self.threshold = threshold        # cosine similarity to count as the same voice
        self.max_voices = max_voices
        self.min_seconds = min_seconds    # shorter local speakers can't start a new voice
        self.reset()

    def reset(self):
        self.voices = {}                  # id -> {"emb": unit vector, "weight": s, "seen": n}
        self.next_id = 1
        self.turn = 0

    @staticmethod
    def _unit(e):
        e = np.asarray(e, dtype=np.float64).ravel()
        n = np.linalg.norm(e)
        return e / n if n > 1e-9 and np.isfinite(n) else None

    def assign(self, embeddings, seconds):
        """Local speakers of one piece -> voice ids (None: too short / no embedding).

        embeddings: (n, dim) one per local speaker; seconds: their speaking time.
        Two local speakers of the same piece never get the same voice (the pipeline
        already decided they differ)."""
        self.turn += 1
        units = [self._unit(e) for e in embeddings]
        ids = [None] * len(units)
        pairs = []
        for i, u in enumerate(units):
            if u is None:
                continue
            for vid, v in self.voices.items():
                pairs.append((float(u @ v["emb"]), i, vid))
        pairs.sort(reverse=True)
        taken = set()
        for sim, i, vid in pairs:
            if sim < self.threshold:
                break
            if ids[i] is None and vid not in taken:
                ids[i] = vid
                taken.add(vid)
        for i, u in enumerate(units):
            if u is None:
                continue
            if ids[i] is None:
                if seconds[i] < self.min_seconds:
                    continue
                ids[i] = self._new(u)
            self._learn(ids[i], u, seconds[i])
        return ids

    def _new(self, u):
        if len(self.voices) >= self.max_voices:      # forget the voice not heard for longest
            del self.voices[min(self.voices, key=lambda k: self.voices[k]["seen"])]
        vid = self.next_id
        self.next_id += 1
        self.voices[vid] = {"emb": u, "weight": 0.0, "seen": self.turn}
        return vid

    def _learn(self, vid, u, seconds):
        v = self.voices[vid]
        w = min(float(seconds), 30.0)
        total = min(v["weight"] + w, 300.0)            # keep adapting a little to new conditions
        mixed = self._unit(v["emb"] * (total - w) + u * w) if total > 0 else u
        v["emb"] = mixed if mixed is not None else u
        v["weight"] = total
        v["seen"] = self.turn


def assign_words(words, turns, max_gap=1.0):
    """Give each word the voice speaking at that time.

    words: [(word, start, end)] (seconds in the piece); turns: [(start, end, voice id)].
    -> [voice id or None] per word. Words in no turn take the nearest turn (within
    max_gap s); a lone word between two words of one voice joins that voice."""
    turns = [(a, b, v) for a, b, v in turns if v is not None and b > a]
    out = []
    for w in words:
        s, e = float(w[1]), float(w[2]) if len(w) > 2 else float(w[1]) + 0.3
        best, best_ov = None, 0.0
        for a, b, v in turns:
            ov = min(e, b) - max(s, a)
            if ov > best_ov:
                best, best_ov = v, ov
        if best is None:
            mid = (s + e) / 2
            near = [(min(abs(mid - a), abs(mid - b)), v) for a, b, v in turns]
            if near:
                d, v = min(near)
                if d <= max_gap:
                    best = v
        out.append(best)
    for i in range(len(out)):                     # carry over gaps
        if out[i] is None and i and out[i - 1] is not None:
            out[i] = out[i - 1]
    for i in range(len(out) - 2, -1, -1):
        if out[i] is None and out[i + 1] is not None:
            out[i] = out[i + 1]
    for i in range(1, len(out) - 1):              # no one-word flicker
        if out[i - 1] == out[i + 1] != out[i]:
            out[i] = out[i - 1]
    return out if any(v is not None for v in out) else None


def runs(words, speakers):
    """-> [(voice, "text")] grouping consecutive words of one voice."""
    out = []
    for w, v in zip(words, speakers):
        if out and out[-1][0] == v:
            out[-1][1].append(w[0])
        else:
            out.append((v, [w[0]]))
    return [(v, " ".join(ws)) for v, ws in out]


class SpeakerTracker:
    """pyannote pipeline + VoiceBook. Call it with 16 kHz mono audio of one piece:
    -> [(start s, end s, voice id)]."""

    def __init__(self, model=None, token=None, device="auto", threshold=0.45, log=print):
        import torch
        import pyannote.audio
        from pyannote.audio import Pipeline

        self.torch = torch
        major = int(pyannote.audio.__version__.split(".")[0])
        self.v4 = major >= 4
        self.model = model or DEFAULT_MODELS[4 if self.v4 else 3]
        token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or None
        kw = {"token": token} if self.v4 else {"use_auth_token": token}
        try:
            self.pipeline = Pipeline.from_pretrained(self.model, **kw)
        except Exception as exc:
            raise RuntimeError(
                f"could not load {self.model} ({exc}). The first time it is downloaded from Hugging "
                f"Face: accept the terms on https://huggingface.co/{self.model} and pass a token with "
                "--hf-token (or set HF_TOKEN)") from exc
        if self.pipeline is None:     # pyannote 3 returns None when the model is gated
            raise RuntimeError(f"no access to {self.model}: accept its terms on "
                               f"https://huggingface.co/{self.model} and pass --hf-token")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.pipeline.to(torch.device(device))
        self.book = VoiceBook(threshold)
        self.log = log
        self.description = f"{self.model} on {device.upper()}"

    def reset(self):
        self.book.reset()

    def __call__(self, wave16k, sr=16000):
        if len(wave16k) < sr:          # under a second: nothing to tell apart
            return []
        audio = {"waveform": self.torch.from_numpy(np.ascontiguousarray(wave16k, dtype=np.float32))[None],
                 "sample_rate": sr}
        with self.torch.inference_mode():
            if self.v4:
                out = self.pipeline(audio, max_speakers=6)
                ann, labels = out.exclusive_speaker_diarization, out.speaker_diarization.labels()
                emb = out.speaker_embeddings
            else:
                ann, emb = self.pipeline(audio, return_embeddings=True, max_speakers=6)
                labels = ann.labels()
        tracks = [(float(seg.start), float(seg.end), lab) for seg, _, lab in ann.itertracks(yield_label=True)]
        if not tracks:
            return []
        if emb is None:
            emb = np.zeros((len(labels), 1))
        seconds = [sum(b - a for a, b, lab in tracks if lab == l) for l in labels]
        ids = self.book.assign(emb[:len(labels)], seconds)
        voice = dict(zip(labels, ids))
        return [(round(a, 2), round(b, 2), voice.get(lab)) for a, b, lab in tracks]
