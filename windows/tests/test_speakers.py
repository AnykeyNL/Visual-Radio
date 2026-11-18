import json
import urllib.request
from types import SimpleNamespace as NS

import numpy as np

from visualradio.speakers import VoiceBook, assign_words, runs


def _voice(seed, dim=64):
    return np.random.default_rng(seed).standard_normal(dim)


def test_voicebook_keeps_numbers_across_pieces():
    rng = np.random.default_rng(9)
    a, b, c = _voice(1), _voice(2), _voice(3)
    book = VoiceBook(threshold=0.45)
    assert book.assign([a, b], [6.0, 5.0]) == [1, 2]
    # next piece: same two people, other order, a bit of noise
    assert book.assign([b + 0.3 * rng.standard_normal(64), a + 0.3 * rng.standard_normal(64)], [4.0, 7.0]) == [2, 1]
    # a new voice gets the next number; two local speakers never share one
    assert book.assign([c, a], [5.0, 5.0]) == [3, 1]
    assert sorted(book.assign([a, a + 0.01], [5.0, 5.0])) == [1, 4]
    # too short to start a new voice, and no embedding at all
    assert book.assign([_voice(7), np.zeros(64)], [0.4, 3.0]) == [None, None]
    book.reset()
    assert book.assign([c], [3.0]) == [1]


def test_voicebook_forgets_the_oldest_voice():
    book = VoiceBook(max_voices=2)
    book.assign([_voice(1)], [3.0])
    book.assign([_voice(2)], [3.0])
    book.assign([_voice(1)], [3.0])            # voice 1 heard again, voice 2 is now the oldest
    assert book.assign([_voice(3)], [3.0]) == [3]
    assert set(book.voices) == {1, 3}


def test_assign_words_to_turns():
    words = [("Goedenavond", 0.0, 0.6), ("en", 0.7, 0.8), ("welkom.", 0.9, 1.3),
             ("Dank", 2.0, 2.3), ("je", 2.35, 2.45), ("wel.", 2.5, 2.9), ("Graag.", 4.5, 4.9)]
    turns = [(0.0, 1.4, 1), (1.9, 3.0, 2), (3.0, 3.6, None)]
    assert assign_words(words, turns) == [1, 1, 1, 2, 2, 2, 2]       # last word: no turn, keeps voice 2
    # a lone word inside another voice's sentence doesn't flicker
    assert assign_words(words[:3], [(0.0, 0.65, 1), (0.65, 0.85, 2), (0.85, 1.4, 1)]) == [1, 1, 1]
    assert assign_words(words, []) is None
    assert runs(words[:6], [1, 1, 1, 2, 2, 2]) == [(1, "Goedenavond en welkom."), (2, "Dank je wel.")]


def test_transcriber_marks_words_with_voices():
    from visualradio.transcribe import Transcriber

    class WordWhisper:
        def transcribe(self, audio, **kw):
            seg = NS(text=" hallo daar", no_speech_prob=0.1,
                     words=[NS(word=" hallo", start=0.4, end=0.8), NS(word=" daar", start=1.6, end=2.0)])
            return iter([seg]), NS(language="nl")

    got = []
    tr = Transcriber(backend=WordWhisper(), word_times=True)
    tr.speakers = lambda a: got.append(len(a)) or [(0.0, 1.0, 1), (1.5, 2.5, 2)]
    t = tr.transcribe("x", 100.0, np.zeros(48000 * 3, np.float32) + 0.01)
    assert t.speakers == [1, 2] and got == [16000 * 3]

    def broken(a):
        raise RuntimeError("CUDA out of memory")
    tr.speakers = broken                                 # text still comes through
    t = tr.transcribe("x", 100.0, np.zeros(48000 * 3, np.float32) + 0.01)
    tr.close()
    assert t.text == "hallo daar" and t.speakers is None


def test_web_sends_voices():
    from visualradio.web import LiveWeb
    web = LiveWeb(0)
    try:
        web.transcript(NS(start=10.0, kind="speech", language="nl", text="hallo daar", duration_s=2.0,
                          words=[("hallo", 0.4, 0.8), ("daar", 1.6, 2.0)], speakers=[1, 2]))
        web.transcript(NS(start=20.0, kind="speech", language="nl", text="zonder", duration_s=1.0,
                          words=[("zonder", 0.1, 0.5)]))
        state = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{web.port}/state").read())
        assert state["history"][0]["speakers"] == [1, 2]
        assert "speakers" not in state["history"][1]
    finally:
        web.close()


def test_tracker_with_pyannote4_style_output(monkeypatch):
    """SpeakerTracker glue, with stand-ins for torch and a pyannote 4 pipeline."""
    import contextlib
    import sys
    import types

    class Ann:
        def __init__(self, tracks):
            self.tracks = tracks
        def itertracks(self, yield_label=False):
            return [(NS(start=a, end=b), None, lab) for a, b, lab in self.tracks]
        def labels(self):
            return sorted({lab for _, _, lab in self.tracks})

    calls = []

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, name, **kw):
            calls.append((name, kw))
            return cls()
        def to(self, device):
            calls.append(("to", device))
        def __call__(self, audio, max_speakers=None):
            assert audio["sample_rate"] == 16000 and audio["waveform"].shape == (1, 48000)
            tracks = [(0.0, 4.0, "SPEAKER_00"), (4.2, 9.0, "SPEAKER_01"), (9.1, 9.4, "SPEAKER_00")]
            emb = np.stack([_voice(1), _voice(2)]) if "run" not in calls else np.stack([_voice(2), _voice(1)])
            calls.append("run")
            return NS(speaker_diarization=Ann(tracks), exclusive_speaker_diarization=Ann(tracks), speaker_embeddings=emb)

    torch = types.SimpleNamespace(from_numpy=lambda a: a, inference_mode=contextlib.nullcontext,
                                  device=lambda d: d, cuda=NS(is_available=lambda: False))
    audio_mod = types.ModuleType("pyannote.audio")
    audio_mod.__version__ = "4.1.0"
    audio_mod.Pipeline = FakePipeline
    pkg = types.ModuleType("pyannote")
    pkg.audio = audio_mod
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "pyannote", pkg)
    monkeypatch.setitem(sys.modules, "pyannote.audio", audio_mod)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    from visualradio.speakers import SpeakerTracker
    tr = SpeakerTracker(token="hf_x")
    assert calls[0] == ("pyannote/speaker-diarization-community-1", {"token": "hf_x"})
    assert tr.description.endswith("on CPU")
    a = np.zeros(48000, np.float32)
    assert tr(a) == [(0.0, 4.0, 1), (4.2, 9.0, 2), (9.1, 9.4, 1)]
    # next piece: the pipeline's labels are swapped, the voices are not
    assert tr(a) == [(0.0, 4.0, 2), (4.2, 9.0, 1), (9.1, 9.4, 2)]
    assert tr(np.zeros(8000, np.float32)) == []
