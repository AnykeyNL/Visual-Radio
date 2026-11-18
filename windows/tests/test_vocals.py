import sys
import types

import numpy as np

from visualradio import vocals
from visualradio.transcribe import Transcriber


def test_modes_and_level():
    assert vocals.wanted("all", "speech") and vocals.wanted("all", "music")
    assert vocals.wanted("talk", "speech") and not vocals.wanted("talk", "music")
    assert vocals.wanted("songs", "music") and not vocals.wanted("songs", "speech")
    assert not vocals.wanted("off", "speech")
    assert vocals.wanted("talk", "")                           # no classifier: it is talk
    loud = vocals.match_level(np.array([0.1, -0.2], np.float32), gain=2.0)
    assert abs(np.max(np.abs(loud)) - 0.9) < 1e-6              # lifted to the normal level
    quiet = vocals.match_level(np.array([0.001, -0.002], np.float32), gain=1.0)
    assert np.max(np.abs(quiet)) <= 0.008 + 1e-9               # instrumental stays quiet (max 4x)


class _Whisper:
    """Writes down the level of what it hears: original audio ~0.9 peak noise,
    the fake vocals are a constant 0.25."""
    def __init__(self):
        self.heard = []

    def transcribe(self, audio, **kw):
        from types import SimpleNamespace as NS
        self.heard.append(round(float(np.median(np.abs(audio))), 2))
        return iter([NS(text=" hallo", no_speech_prob=0.1, words=[])]), NS(language="nl")


class _Split:
    def regions(self, a):
        d = len(a) / 16000
        return [("speech", 0.0, d / 2), ("music", d / 2, d)]


class _Isolator:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def __call__(self, mono, rate, length):
        self.calls.append((len(mono), rate, length))
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        return np.full(length, 0.25, np.float32)


def test_transcriber_feeds_isolated_voice_to_whisper():
    audio = (np.random.default_rng(3).uniform(-1, 1, 48000 * 6)).astype(np.float32)
    for mode, expect in (("off", [False, False]), ("talk", [True, False]),
                         ("songs", [False, True]), ("all", [True, True])):
        w, iso = _Whisper(), _Isolator()
        tr = Transcriber(backend=w, classifier=_Split())
        tr.isolator, tr.isolate = iso, mode
        res = tr.transcribe_all("x", 0.0, audio)
        tr.close()
        assert [t.kind for t in res] == ["speech", "music"]
        assert [h > 0.5 for h in w.heard] == expect, (mode, w.heard)   # flat 0.25 vocals, lifted
        assert len(iso.calls) == (0 if mode == "off" else 1)            # once per piece
        if iso.calls:
            assert iso.calls[0] == (48000 * 6, 48000, 16000 * 6)


def test_isolation_failure_falls_back_to_normal_audio(capsys):
    audio = (np.random.default_rng(4).uniform(-1, 1, 48000 * 4)).astype(np.float32)
    w = _Whisper()
    tr = Transcriber(backend=w, classifier=_Split())
    tr.isolator, tr.isolate = _Isolator(fail=True), "all"
    assert len(tr.transcribe_all("x", 0.0, audio)) == 2
    assert len(tr.transcribe_all("x", 4.0, audio)) == 2
    tr.close()
    assert all(h < 0.5 for h in w.heard)
    assert capsys.readouterr().out.count("Voice isolation failed") == 1    # said once


def test_isolator_with_fake_demucs(monkeypatch):
    """The Demucs plumbing: resample to 44.1 kHz stereo, take the vocals stem, back to 16 kHz."""
    torch = types.ModuleType("torch")

    class T:                                    # just enough of a tensor
        def __init__(self, a):
            self.a = np.asarray(a, np.float32)
        def __getitem__(self, i):
            return T(self.a[i])
        def repeat(self, n, m):
            return T(np.repeat(self.a, n, axis=0))
        def mean(self, axis=None):
            return T(self.a.mean(axis))
        def std(self):
            return T(self.a.std())
        def __add__(self, o):
            return T(self.a + (o.a if isinstance(o, T) else o))
        def __sub__(self, o):
            return T(self.a - (o.a if isinstance(o, T) else o))
        def __mul__(self, o):
            return T(self.a * (o.a if isinstance(o, T) else o))
        def __truediv__(self, o):
            return T(self.a / (o.a if isinstance(o, T) else o))
        def float(self):
            return self
        def cpu(self):
            return self
        def numpy(self):
            return self.a

    class NoGrad:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    torch.from_numpy = T
    torch.inference_mode = NoGrad
    torch.device = lambda d: d
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    class Model:
        samplerate, audio_channels = 44100, 2
        sources = ["drums", "bass", "other", "vocals"]
        def to(self, dev):
            return self
        def eval(self):
            return self

    seen = {}

    def apply_model(model, mix, **kw):
        seen["shape"], seen["kw"] = mix.a.shape, kw
        out = np.zeros((1, 4) + mix.a.shape[1:], np.float32)
        out[0, 3] = mix.a[0] * 0.5             # "vocals" = half the mix
        return T(out)

    pkg = types.ModuleType("demucs")
    pretrained = types.ModuleType("demucs.pretrained")
    pretrained.get_model = lambda name: (seen.__setitem__("name", name), Model())[1]
    apply = types.ModuleType("demucs.apply")
    apply.apply_model = apply_model
    for name, mod in (("torch", torch), ("demucs", pkg), ("demucs.pretrained", pretrained),
                      ("demucs.apply", apply)):
        monkeypatch.setitem(sys.modules, name, mod)

    logs = []
    iso = vocals.VocalIsolator("htdemucs", log=logs.append)
    t = np.arange(48000 * 2) / 48000
    mono = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    v = iso(mono, 48000, 32000)
    assert seen["name"] == "htdemucs" and seen["shape"] == (1, 2, 88200)
    assert seen["kw"]["device"] == "cpu" and seen["kw"]["shifts"] == 0
    assert iso.description == "Demucs htdemucs on CPU" and logs        # warns that the CPU is slow
    assert v.shape == (32000,) and v.dtype == np.float32
    assert abs(np.max(np.abs(v[1000:-1000])) - 0.25) < 0.02            # half of the 0.5 sine
