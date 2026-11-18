import numpy as np

from visualradio.style import SceneScorer, itunes_scene, tempo


def test_tempo_of_a_steady_kick():
    sr = 16000
    t = np.arange(sr * 15) / sr
    x = 0.05 * np.random.default_rng(0).standard_normal(len(t))
    for b in np.arange(0.3, 15, 60 / 128):
        i = int(b * sr)
        x[i:i + 1600] += np.sin(2 * np.pi * 60 * t[:1600]) * np.exp(-t[:1600] * 30)
    bpm, first = tempo(x.astype(np.float32))
    assert abs(bpm - 128) < 2
    assert abs(first - 0.3) < 0.06


def test_no_tempo_in_noise_or_short_audio():
    rng = np.random.default_rng(1)
    assert tempo(rng.standard_normal(16000 * 3).astype(np.float32)) is None


def test_scene_from_scores_and_itunes():
    names = ["Music", "Pop music", "Techno", "Jazz", "Banjo"]
    sc = SceneScorer(names)
    p = np.zeros((10, 5))
    p[:, 2] = 0.05
    assert sc(p)[0] == "electronic"
    p[:, 2] = 0.0
    p[:, 4] = 0.04            # instrument hint at half weight
    assert sc(p)[0] == "folk"
    assert sc(np.zeros((10, 5)))[0] == "pop"
    assert itunes_scene("Hip-Hop/Rap") == "hiphop" and itunes_scene("Nederpop") == "pop"
    assert itunes_scene("Dance") == "electronic" and itunes_scene(None) is None
