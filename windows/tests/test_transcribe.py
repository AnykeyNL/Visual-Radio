import numpy as np


class _FakeWhisper:
    """Stands in for faster_whisper.WhisperModel: reports the audio length it got."""

    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kw):
        from types import SimpleNamespace as NS
        self.calls.append((len(audio), kw))
        segs = [NS(text=f" {len(audio) / 16000:.1f} seconds ", no_speech_prob=0.1),
                NS(text=" Bedankt voor het kijken ", no_speech_prob=0.1),
                NS(text=" ruis ", no_speech_prob=0.9)]
        return iter(segs), NS(language=kw.get("language") or "nl")


def test_transcribe_skips_short_and_silent():
    from visualradio.transcribe import Transcriber

    tr = Transcriber(backend=_FakeWhisper())
    assert tr.transcribe(1, 0.0, np.zeros(48000, np.float32)) is None
    assert tr.transcribe(1, 0.0, np.ones(4800, np.float32) * 0.1) is None   # 0.1 s
    tr.close()
