import numpy as np

from visualradio.audio import AUDIO_RATE, FmAudio
from visualradio.sources import SimulatedSource


def test_fm_audio_recovers_station_tones():
    src = SimulatedSource(sample_rate=6e6, stations=[(100.7, 40, "BIG FM")], seed=3)
    tones = src.stations[0].tones
    fm = FmAudio(6e6)
    src.tune(fm.tune_hz(100.7e6))
    blk = 600000 - 600000 % fm.block_multiple
    a = np.concatenate([fm.process(src.read(blk)) for _ in range(10)])[AUDIO_RATE // 10:]
    assert abs(len(a) / AUDIO_RATE - 0.9) < 0.01
    spec = np.abs(np.fft.rfft(a * np.hanning(len(a))))
    f = np.fft.rfftfreq(len(a), 1 / AUDIO_RATE)
    floor = np.median(spec)
    for tone in tones:          # each station tone stands far above the rest
        assert spec[np.abs(f - tone) < 5].max() > 100 * floor
    # nothing at the 19 kHz pilot (filtered out)
    assert spec[np.abs(f - 19e3) < 50].max() < spec.max() * 1e-2


def test_chunker_cuts_at_quiet_point():
    from visualradio.transcribe import ContinuousChunker
    got = []
    ch = ContinuousChunker(lambda *a: got.append(a), "x", chunk_s=3.0, rate=1000)
    sig = np.ones(10000, np.float32)
    sig[2500:2600] = 0       # a pause at 2.5 s, in the last third of the first 3 s
    for i in range(0, 10000, 100):
        ch.add(sig[i:i + 100], now=(i + 100) / 1000)
    ch.flush()
    lens = [len(a[2]) for a in got]
    assert sum(lens) == 10000
    assert 2500 <= lens[0] <= 2600
    assert abs(got[1][1] - lens[0] / 1000) < 0.15     # start time of the 2nd piece




def test_content_filter_plumbing():
    import sys
    sys.path.insert(0, "tests")
    from test_transcribe import _FakeWhisper
    from visualradio.transcribe import Transcriber

    class FakeClassifier:
        def regions(self, a):
            d = len(a) / 16000
            return [("speech", 0.0, d / 2), ("music", d / 2, d)]

    audio = (np.random.default_rng(1).standard_normal(48000 * 10) * 0.1).astype(np.float32)
    for content, kinds in (("both", ["speech", "music"]), ("speech", ["speech"]), ("music", ["music"])):
        fake = _FakeWhisper()
        tr = Transcriber(backend=fake, classifier=FakeClassifier(), content=content)
        res = tr.transcribe_all("100.7 MHz", 1000.0, audio)
        tr.close()
        assert [t.kind for t in res] == kinds
        assert all(abs(float(t.text.split()[0]) - 5.0) < 0.2 for t in res)
        if content == "both":
            assert res[1].start == 1005.0
            assert fake.calls[0][1]["vad_filter"] and not fake.calls[1][1]["vad_filter"]


def test_yamnet_numpy_on_simple_sounds():
    from visualradio.content import SoundClassifier, log_mel_patches
    c = SoundClassifier()
    assert log_mel_patches(np.zeros(16000 * 3, np.float32)).shape[1:] == (96, 64)
    assert c.classify(np.zeros(16000 * 2, np.float32)) == (0.0, 0.0)     # silence: neither
    t = np.arange(16000 * 5) / 16000
    chord = sum(np.sin(2 * np.pi * f * t) for f in (261.6, 329.6, 392.0, 523.3)) * 0.2
    sp, mu = c.classify(chord.astype(np.float32))
    assert mu >= sp


def test_delay_line():
    from visual_radio import DelayLine
    d = DelayLine(0.25, rate=1000)           # 250 samples
    x = np.arange(1, 1001, dtype=np.float32)
    out = np.concatenate([d(x[i:i + 100]) for i in range(0, 1000, 100)])
    assert np.all(out[:250] == 0) and np.array_equal(out[250:], x[:750])
    assert DelayLine(0)(x) is x


def test_word_times_pass_through():
    from types import SimpleNamespace as NS
    from visualradio.transcribe import Transcriber

    class WordWhisper:
        def transcribe(self, audio, **kw):
            assert kw["word_timestamps"] and not kw["without_timestamps"]
            seg = NS(text=" hallo daar", no_speech_prob=0.1,
                     words=[NS(word=" hallo", start=0.4, end=0.8), NS(word=" daar", start=0.9, end=1.2)])
            noise = NS(text=" ♪", no_speech_prob=0.1, words=[NS(word=" ♪", start=2.0, end=2.5)])
            return iter([seg, noise]), NS(language="nl")

    tr = Transcriber(backend=WordWhisper(), word_times=True)
    t = tr.transcribe("x", 100.0, (np.random.default_rng(0).standard_normal(48000 * 3) * .1).astype(np.float32))
    tr.close()
    assert t.text == "hallo daar" and t.words == [("hallo", 0.4, 0.8), ("daar", 0.9, 1.2)]


def test_music_without_words_is_instrumental():
    from types import SimpleNamespace as NS
    from visualradio.transcribe import Transcriber, music_placeholder

    for t in ("Muziek", "MUZIEK!", "[Muziek]", "(upbeat music)", "♪ ♪", "Musik Musik", "Instrumentaal"):
        assert music_placeholder(t), t
    for t in ("Ik hou van muziek", "Music by Kevin", "and", "Goedemorgen"):
        assert not music_placeholder(t), t

    class W:
        def __init__(self, text):
            self.text = text
        def transcribe(self, audio, **kw):
            return iter([NS(text=self.text, no_speech_prob=0.1, words=[])]), NS(language="nl")

    class C:
        def __init__(self, kind):
            self.kind = kind
        def regions(self, a):
            return [(self.kind, 0.0, 3.0)]

    a = np.zeros(48000 * 3, np.float32) + 0.01
    tr = Transcriber(backend=W(" Muziek"), classifier=C("music"))
    [t] = tr.transcribe_all("x", 100.0, a)
    assert t.instrumental and t.text == "" and t.kind == "music"
    tr = Transcriber(backend=W(" Muziek"), classifier=C("speech"))
    assert tr.transcribe_all("x", 100.0, a) == []              # talk: just dropped
    tr = Transcriber(backend=W(" la la la"), classifier=C("music"))
    [t] = tr.transcribe_all("x", 100.0, a)
    assert not t.instrumental and t.text == "la la la"
