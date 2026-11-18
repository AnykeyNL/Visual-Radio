"""Per-song look for the web page: genre scene and tempo.

- scene: one of SCENES, from YAMNet's genre classes (works for any song, also
  unknown ones), or from the iTunes genre once the song is identified.
- tempo: beats per minute and the time of one beat, from the audio itself
  (spectral-flux onsets + autocorrelation), so the page can pulse on the beat.
"""

from __future__ import annotations

import re

import numpy as np

SCENES = ("pop", "electronic", "rock", "hiphop", "classical", "jazz", "folk")

# YAMNet class names per scene
SCENE_CLASSES = {
    "pop": ("Pop music", "Happy music"),
    "electronic": ("Electronic music", "House music", "Techno", "Dubstep", "Drum and bass", "Electronica",
                   "Electronic dance music", "Trance music", "Dance music", "Disco"),
    "rock": ("Rock music", "Heavy metal", "Punk rock", "Grunge", "Progressive rock", "Rock and roll",
             "Psychedelic rock", "Angry music"),
    "hiphop": ("Hip hop music", "Rapping", "Beatboxing", "Scratching (performance technique)"),
    "classical": ("Classical music", "Opera", "Ambient music", "New-age music", "Orchestra"),
    "jazz": ("Jazz", "Blues", "Soul music", "Rhythm and blues", "Funk", "Swing music"),
    "folk": ("Country", "Folk music", "Bluegrass", "Traditional music", "Reggae", "Ska",
             "Music of Latin America", "Salsa music", "Flamenco", "Afrobeat", "Music of Africa"),
}

# Instruments that hint at a scene (counted at half weight)
SCENE_INSTRUMENTS = {
    "electronic": ("Synthesizer", "Drum machine", "Sampler"),
    "rock": ("Electric guitar", "Distortion", "Drum kit"),
    "hiphop": ("Scratching (performance technique)",),
    "classical": ("Orchestra", "String section", "Violin, fiddle", "Cello", "Harp", "Harpsichord"),
    "jazz": ("Saxophone", "Trumpet", "Vibraphone", "Hammond organ", "Double bass", "Brass instrument"),
    "folk": ("Banjo", "Acoustic guitar", "Accordion", "Mandolin", "Harmonica", "Steel guitar, slide guitar"),
}

ITUNES_SCENES = [   # (regex on iTunes primaryGenreName, scene), first match wins
    (r"hip.?hop|rap", "hiphop"),
    (r"dance|electro|house|techno|trance|dubstep|edm|drum", "electronic"),
    (r"rock|metal|punk|alternati|grunge|indie", "rock"),
    (r"classical|opera|soundtrack|ambient|new age|instrumental", "classical"),
    (r"jazz|blues|soul|r&b|funk|swing", "jazz"),
    (r"country|folk|singer|reggae|latin|world|ska|afro|schlager|levenslied", "folk"),
    (r"pop", "pop"),
]


def itunes_scene(genre: str | None):
    for pat, scene in ITUNES_SCENES:
        if genre and re.search(pat, genre, re.I):
            return scene
    return None


class SceneScorer:
    def __init__(self, class_names):
        names = list(class_names)
        self.idx = {s: [names.index(c) for c in cls if c in names] for s, cls in SCENE_CLASSES.items()}
        self.inst = {s: [names.index(c) for c in cls if c in names] for s, cls in SCENE_INSTRUMENTS.items()}

    def __call__(self, scores: np.ndarray):
        """(N, 521) YAMNet scores of a music stretch -> (scene, confidence 0..1)."""
        if len(scores) == 0:
            return "pop", 0.0
        mean = scores.mean(axis=0)
        # genre scores are small (often 0.01-0.05) but their ranking is what counts
        val = {s: float(mean[i].max()) if i else 0.0 for s, i in self.idx.items()}
        for s, i in self.inst.items():
            if i:
                val[s] = max(val[s], 0.5 * float(mean[i].max()))
        val["pop"] *= 0.8           # YAMNet calls a lot "pop": only when nothing else stands out
        best = max(val, key=val.get)
        if val[best] < 0.006:
            return "pop", 0.0
        total = sum(val.values()) or 1.0
        return best, round(val[best] / total, 2)


def tempo(wave16k: np.ndarray, sr: int = 16000):
    """-> (bpm, seconds to the first beat) or None when there is no steady beat."""
    hop, nfft = 160, 1024                       # 10 ms frames
    if len(wave16k) < sr * 6:
        return None
    n = 1 + (len(wave16k) - nfft) // hop
    idx = np.arange(nfft)[None, :] + hop * np.arange(n)[:, None]
    spec = np.abs(np.fft.rfft(wave16k[idx] * np.hanning(nfft).astype(np.float32), axis=1))
    edges = np.unique(np.geomspace(2, nfft // 2, 33).astype(int))
    bands = np.stack([spec[:, a:b].sum(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1)
    logb = np.log1p(100 * bands / (bands.max() + 1e-9))
    flux = np.maximum(0, np.diff(logb, axis=0)).sum(axis=1)
    kern = np.ones(50) / 50                       # remove the slow level, keep the hits
    env = np.maximum(0, flux - np.convolve(flux, kern, mode="same"))
    if env.std() < 1e-6:
        return None
    env = (env - env.mean()) / env.std()
    fps = sr / hop
    lags = np.arange(int(fps * 60 / 200), int(fps * 60 / 60) + 1)
    ac = np.array([np.dot(env[:-l], env[l:]) / (len(env) - l) for l in lags])
    bpm_at = 60 * fps / lags
    weight = np.exp(-0.5 * (np.log2(bpm_at / 118) / 0.9) ** 2)   # people tap near 120
    score = ac * weight
    k = int(np.argmax(score))
    if ac[k] < 0.08:                              # no clear periodicity
        return None
    lag = float(lags[k])
    if 0 < k < len(lags) - 1:                     # parabolic refinement
        a, b, c = score[k - 1], score[k], score[k + 1]
        d = a - 2 * b + c
        if d < 0:
            lag += 0.5 * (a - c) / d
    period = int(round(lag))
    comb = [env[o::period].sum() / len(env[o::period]) for o in range(period)]
    first = (int(np.argmax(comb)) + 1) / fps + nfft / sr     # flux peaks as the hit enters the frame
    bpm = 60 * fps / lag
    while bpm < 80:                               # keep in 80-160: the page pulses on every beat
        bpm *= 2
    while bpm > 160:
        bpm /= 2
    return round(float(bpm), 1), round(float(first), 3)
