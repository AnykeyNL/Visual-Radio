"""Voice isolation before Whisper: take the music out, keep the voice.

Talk over a music bed, jingles and songs are hard for Whisper. Demucs (Meta's
music source separation, MIT licence) splits audio into drums, bass, other and
vocals; only the vocals go to Whisper (and to the speaker colours). The original
audio still goes to the speakers, the talk/music classifier and song recognition.

Runs on this PC. The first use downloads the model (htdemucs, about 80 MB) from
dl.fbaipublicfiles.com; after that it works offline. On a GPU a 15 s piece takes
well under a second; on a CPU it takes several seconds.
"""

from __future__ import annotations

import importlib.util
import threading

import numpy as np
from scipy import signal

MODES = ("off", "talk", "songs", "all")
KIND_MODE = {"speech": "talk", "music": "songs", "": "talk"}


def available():
    """True when Demucs is installed (pip install demucs)."""
    return importlib.util.find_spec("demucs") is not None


def wanted(mode, kind):
    """Should a piece of this kind ("speech" / "music") be isolated in this mode?"""
    return mode == "all" or (mode != "off" and KIND_MODE.get(kind or "", "talk") == mode)


def match_level(vocals, gain, target=0.9, max_boost=4.0):
    """Scale the vocals like the original piece was scaled (gain), then lift them
    towards the target peak, but at most max_boost times: a near-silent vocal track
    (instrumental) must stay quiet, or Whisper starts hearing words in the noise."""
    v = np.asarray(vocals, dtype=np.float32) * float(gain)
    peak = float(np.max(np.abs(v))) if len(v) else 0.0
    if peak > 1e-9:
        v = v * min(target / peak, max_boost)
    return v.astype(np.float32)


class VocalIsolator:
    """Call it with mono audio (rate Hz) -> mono vocals at 16 kHz, `length` samples.

    The model is loaded on the first call (or with load()), so creating one is cheap."""

    def __init__(self, model="htdemucs", device="auto", log=print):
        self.name = model
        self.device_pref = device
        self.log = log
        self.model = None
        self.description = f"Demucs {model}"
        self.lock = threading.RLock()     # Whisper's isolation and the listen stems share the model

    def load(self):
        with self.lock:
            return self._load()

    def _load(self):
        if self.model is not None:
            return self
        import torch
        from demucs.pretrained import get_model

        self.torch = torch
        dev = self.device_pref
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = get_model(self.name)
        model.to(torch.device(dev)).eval()
        if "vocals" not in model.sources:
            raise RuntimeError(f"{self.name} has no vocals stem (sources: {', '.join(model.sources)})")
        self.device = dev
        self.model = model
        self.description = f"Demucs {self.name} on {dev.upper()}"
        if dev == "cpu":
            self.log("Voice isolation runs on the CPU: this is slow, transcripts may fall behind")
        return self

    def _run(self, mono, rate):
        """All stems: -> (tensor [source, channel, time] at the model's rate, model)."""
        from demucs.apply import apply_model

        with self.lock:
            return self._apply(mono, rate, apply_model)

    def _apply(self, mono, rate, apply_model):
        self._load()
        torch, m = self.torch, self.model
        a = np.asarray(mono, dtype=np.float32)
        if rate != m.samplerate:
            g = np.gcd(int(m.samplerate), int(rate))
            a = signal.resample_poly(a, m.samplerate // g, rate // g).astype(np.float32)
        wav = torch.from_numpy(np.ascontiguousarray(a))[None].repeat(m.audio_channels, 1)
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        with torch.inference_mode():
            out = apply_model(m, ((wav - mean) / std)[None], device=self.device,
                              shifts=0, split=True, overlap=0.25, progress=False)[0]
        return out * std + mean, m

    def separate(self, mono, rate):
        """-> vocals, mono float32 at the model's rate (44.1 kHz)."""
        out, m = self._run(mono, rate)
        return out[m.sources.index("vocals")].mean(0).float().cpu().numpy(), int(m.samplerate)

    def voice_and_music(self, mono, rate=48000):
        """For listening: -> (voice, everything but the voice), mono float32 at `rate`,
        as long as the input."""
        out, m = self._run(mono, rate)
        k = m.sources.index("vocals")
        voice = out[k].mean(0).float().cpu().numpy()
        music = (out.sum(0) - out[k]).mean(0).float().cpu().numpy()
        res = []
        for x in (voice, music):
            if int(m.samplerate) != rate:
                g = np.gcd(int(m.samplerate), int(rate))
                x = signal.resample_poly(x, rate // g, int(m.samplerate) // g)
            x = np.asarray(x, dtype=np.float32)
            n = len(mono)
            res.append(x[:n] if len(x) >= n else np.pad(x, (0, n - len(x))))
        return res[0], res[1]

    def __call__(self, mono, rate, length=None):
        vocals, sr = self.separate(mono, rate)
        v = signal.resample_poly(vocals, 160, 441) if sr == 44100 else \
            signal.resample_poly(vocals, 16000 // np.gcd(16000, sr), sr // np.gcd(16000, sr))
        v = np.asarray(v, dtype=np.float32)
        if length is not None:
            v = v[:length] if len(v) >= length else np.pad(v, (0, length - len(v)))
        return v
