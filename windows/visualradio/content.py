"""Speech or music? A small audio classifier (YAMNet) in plain NumPy.

YAMNet is Google's AudioSet sound classifier (MobileNet v1, 521 classes,
Apache 2.0). Its weights are in models/yamnet.npz (float16, ~7 MB); the network
below is a straight NumPy re-implementation, so no TensorFlow or PyTorch is
needed. It looks at 0.96 s windows of 16 kHz audio and gives a score per class;
here only Speech-like and Music-like classes are used.
"""

from __future__ import annotations

import os

import numpy as np

SR = 16000
_LAYERS = [  # (kind, stride, filters) as in the original YAMNet
    ("conv", 2, 32), ("sep", 1, 64), ("sep", 2, 128), ("sep", 1, 128), ("sep", 2, 256),
    ("sep", 1, 256), ("sep", 2, 512), ("sep", 1, 512), ("sep", 1, 512), ("sep", 1, 512),
    ("sep", 1, 512), ("sep", 1, 512), ("sep", 2, 1024), ("sep", 1, 1024),
]
SPEECH = ("Speech", "Male speech, man speaking", "Female speech, woman speaking",
          "Child speech, kid speaking", "Conversation", "Narration, monologue")
MUSIC = ("Music", "Singing", "Musical instrument", "Song", "Pop music", "Rock music",
         "Hip hop music", "Electronic music", "Dance music", "Classical music")

# Sung vocals. Talk over a music bed scores high on both Speech and Music; it is
# talk when Speech is clearly there and these singing classes are not.
VOCAL = ("Singing", "Choir", "Child singing", "Synthetic singing", "Rapping", "Humming",
         "Vocal music", "A capella", "Song", "Chant", "Yodeling")
SPEECH_OVER_MUSIC = 0.4

DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "models", "yamnet.npz")


def _mel_matrix(n_fft=512, n_mels=64, lo=125.0, hi=7500.0):
    """Same as tf.signal.linear_to_mel_weight_matrix (HTK mel, DC bin zeroed)."""
    def mel(f):
        return 1127.0 * np.log1p(np.asarray(f) / 700.0)
    bins = n_fft // 2 + 1
    spec_mel = mel(np.linspace(0, SR / 2, bins)[1:])[:, None]
    edges = np.linspace(mel(lo), mel(hi), n_mels + 2)
    lower, center, upper = edges[:-2], edges[1:-1], edges[2:]
    w = np.maximum(0, np.minimum((spec_mel - lower) / (center - lower),
                                 (upper - spec_mel) / (upper - center)))
    return np.vstack([np.zeros((1, n_mels)), w]).astype(np.float32)


def log_mel_patches(wave16k: np.ndarray) -> np.ndarray:
    """-> (n_patches, 96, 64): 25 ms / 10 ms frames, 0.96 s patches every 0.48 s."""
    x = np.asarray(wave16k, dtype=np.float32)
    min_len = int((0.96 + 0.025 - 0.010) * SR)
    hop = int(0.48 * SR)
    n = max(len(x), min_len)
    n = min_len + int(np.ceil((n - min_len) / hop)) * hop
    x = np.pad(x, (0, n - len(x)))
    win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 400)).astype(np.float32)  # periodic Hann
    idx = np.arange(0, len(x) - 400 + 1, 160)[:, None] + np.arange(400)
    mag = np.abs(np.fft.rfft(x[idx] * win, 512))
    logmel = np.log(mag @ _mel_matrix() + 0.001).astype(np.float32)
    starts = np.arange(0, len(logmel) - 96 + 1, 48)
    return np.stack([logmel[s:s + 96] for s in starts])


def _pad_same(x, stride):
    """TensorFlow 'same' padding for a 3x3 kernel on (N, H, W, C)."""
    def amounts(size):
        out = -(-size // stride)
        total = max((out - 1) * stride + 3 - size, 0)
        return total // 2, total - total // 2
    return np.pad(x, ((0, 0), amounts(x.shape[1]), amounts(x.shape[2]), (0, 0)))


def _taps(x, stride):
    """Yield (di, dj, slice) for the 9 positions of a 3x3 kernel."""
    n, h, w, _ = x.shape
    oh, ow = -(-h // stride), -(-w // stride)
    xp = _pad_same(x, stride)
    for di in range(3):
        for dj in range(3):
            yield di, dj, xp[:, di:di + stride * oh:stride, dj:dj + stride * ow:stride, :]


class SoundClassifier:
    def __init__(self, weights_path=DEFAULT_WEIGHTS):
        z = np.load(weights_path)
        self.w = {k: z[k].astype(np.float32) for k in z.files if k != "class_names"}
        self.names = list(z["class_names"])
        self.speech_idx = [self.names.index(c) for c in SPEECH if c in self.names]
        self.music_idx = [self.names.index(c) for c in MUSIC if c in self.names]
        self.vocal_idx = [self.names.index(c) for c in VOCAL if c in self.names]

    def _labels(self, p):
        """(N, 521) scores -> per window 0 (nothing), 1 (speech) or 2 (music)."""
        sp = p[:, self.speech_idx].max(axis=1)
        mu = p[:, self.music_idx].max(axis=1)
        vo = p[:, self.vocal_idx].max(axis=1) if self.vocal_idx else np.zeros(len(p))
        speech = (sp >= mu) | ((sp > SPEECH_OVER_MUSIC) & (sp > vo))   # talk over a music bed
        return np.where((sp < 0.2) & (mu < 0.2), 0, np.where(speech, 1, 2))

    def _bn_relu(self, x, name):
        w = self.w
        x = (x - w[f"{name}/bn/moving_mean"]) / np.sqrt(w[f"{name}/bn/moving_variance"] + 1e-4)
        return np.maximum(x + w[f"{name}/bn/beta"], 0)

    def scores(self, patches: np.ndarray) -> np.ndarray:
        """(N, 96, 64) log-mel patches -> (N, 521) class scores (0..1)."""
        x = patches[..., None].astype(np.float32)
        for i, (kind, stride, filters) in enumerate(_LAYERS, 1):
            if kind == "conv":
                k = self.w[f"layer{i}/conv/kernel"]                       # (3, 3, 1, F)
                x = sum(s @ k[di, dj] for di, dj, s in _taps(x, stride))
                x = self._bn_relu(x, f"layer{i}/conv")
            else:
                k = self.w[f"layer{i}/depthwise_conv/depthwise_kernel"]   # (3, 3, C, 1)
                x = sum(s * k[di, dj, :, 0] for di, dj, s in _taps(x, stride))
                x = self._bn_relu(x, f"layer{i}/depthwise_conv")
                x = x @ self.w[f"layer{i}/pointwise_conv/kernel"][0, 0]
                x = self._bn_relu(x, f"layer{i}/pointwise_conv")
        emb = x.mean(axis=(1, 2))
        logits = emb @ self.w["logits/kernel"] + self.w["logits/bias"]
        return 1 / (1 + np.exp(-np.clip(logits, -50, 50)))

    def classify(self, wave16k: np.ndarray):
        """-> (speech score, music score), each 0..1, for a piece of 16 kHz audio.

        Scores are the fraction of 0.96 s windows where that class dominates,
        so 0.7 speech / 0.3 music means about 70% talk and 30% music.
        """
        lab = self._labels(self.scores(log_mel_patches(wave16k)))
        if not lab.any():
            return 0.0, 0.0
        return float(np.mean(lab == 1)), float(np.mean(lab == 2))

    def style(self, wave16k, t0, t1):
        """Look of a music stretch t0..t1 (s) of the audio last given to regions():
        {"scene", "confidence", "bpm", "beat"} (beat: s after t0 of one beat)."""
        from .style import SceneScorer, tempo
        if not hasattr(self, "_scene"):
            self._scene = SceneScorer(self.names)
        p = getattr(self, "last_scores", np.zeros((0, len(self.names))))
        scene, conf = self._scene(p[int(t0 / 0.48):max(int(t0 / 0.48) + 1, int((t1 - 0.96) / 0.48) + 1)])
        out = {"scene": scene, "confidence": conf}
        beat = tempo(wave16k[int(t0 * SR):int(t1 * SR)])
        if beat:
            out["bpm"], out["beat"] = beat
        return out

    def top(self, wave16k, k=3):
        p = self.scores(log_mel_patches(wave16k)).mean(axis=0)
        return [(self.names[i], round(float(p[i]), 2)) for i in np.argsort(p)[::-1][:k]]

    def regions(self, wave16k: np.ndarray, min_windows: int = 3):
        """Split audio into runs of speech / music: -> [(kind, start_s, end_s)].

        Each 0.96 s window (every 0.48 s) is labelled speech, music or nothing;
        runs shorter than `min_windows` windows join the run before them.
        """
        p = self.scores(log_mel_patches(wave16k))
        self.last_scores = p
        lab = self._labels(p)
        if len(lab) >= 5:   # 5-window (2.4 s) median smooths odd windows
            pad = np.concatenate([lab[:1], lab[:1], lab, lab[-1:], lab[-1:]])
            lab = np.median(np.stack([pad[i:i + len(lab)] for i in range(5)]), axis=0).astype(int)
        elif len(lab) >= 3:
            pad = np.concatenate([lab[:1], lab, lab[-1:]])
            lab = np.median(np.stack([pad[:-2], pad[1:-1], pad[2:]]), axis=0).astype(int)
        runs = []
        for i, v in enumerate(lab):
            if runs and runs[-1][0] == v:
                runs[-1][2] = i
            else:
                runs.append([v, i, i])
        merged = []
        for r in runs:
            if merged and (r[2] - r[1] + 1 < min_windows or r[0] == merged[-1][0]):
                merged[-1][2] = r[2]
            else:
                merged.append(r)
        if len(merged) > 1 and merged[0][2] - merged[0][1] + 1 < min_windows:
            merged[1][1] = merged[0][1]
            merged.pop(0)
        dur = len(wave16k) / SR
        kinds = {0: None, 1: "speech", 2: "music"}
        return [(kinds[v], a * 0.48, min(dur, b * 0.48 + 0.96)) for v, a, b in merged]
