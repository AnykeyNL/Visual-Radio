import struct
import threading
import time
import urllib.request

import numpy as np

from visualradio.stems import MUSIC, VOICE, StemSplitter
from visualradio.web import LiveWeb


def _collect(splitter_kwargs, feed_s=12.0, block=4800, gap_at=None):
    got = {VOICE: [], MUSIC: []}
    done = threading.Event()

    def emit(kind, times, blocks):
        got[kind].extend(zip(times, blocks))

    rng = np.random.default_rng(1)
    x = rng.standard_normal(int(feed_s * 48000)).astype(np.float32)
    sp = StemSplitter(lambda a: (a * 0.25, a * 0.75), emit, log=lambda m: None, **splitter_kwargs)
    t0 = 1000.0
    try:
        for i in range(0, len(x), block):
            sp.feed(x[i:i + block], t0 + i / 48000)
        deadline = time.time() + 5
        while time.time() < deadline and len(got[VOICE]) * 0.25 < feed_s - 4:
            time.sleep(0.05)
    finally:
        sp.close()
        done.set()
    return x, t0, got


def test_stems_are_seamless_and_keep_the_time():
    x, t0, got = _collect({})
    for kind, scale in ((VOICE, 0.25), (MUSIC, 0.75)):
        frames = got[kind]
        assert len(frames) >= 30
        times = [t for t, _ in frames]
        assert np.allclose(np.diff(times), 0.25, atol=1e-6)          # one frame every 0.25 s, no gaps
        assert abs(times[0] - t0) < 1e-6                              # starts with the first sample
        y = np.concatenate([b for _, b in frames])
        assert np.allclose(y, x[:len(y)] * scale, atol=1e-5)          # crossfaded joins are exact


def test_stems_skip_ahead_when_too_slow_and_reset():
    calls = []

    def slow(a):
        calls.append(len(a))
        time.sleep(0.3)
        return a, a

    out = []
    logs = []
    sp = StemSplitter(slow, lambda k, t, b: out.append(t[0]), log=logs.append, max_behind=6.0)
    try:
        x = np.zeros(48000, np.float32)
        for i in range(12):                              # 12 s of audio at once
            sp.feed(x, 50.0 + i)
        assert any("keep up" in m for m in logs)
        time.sleep(0.8)
        sp.reset()
        assert sp.total == 0 and sp.emitted is None
        sp.feed(x, 500.0)                                # jump in time: starts afresh
        assert sp.base == 0 and sp.total == 48000
    finally:
        sp.close()


def test_web_streams_stem_frames():
    web = LiveWeb(0)
    try:
        assert not web.stems_wanted()
        r = urllib.request.urlopen(f"http://127.0.0.1:{web.port}/audio?stems=1", timeout=5)
        deadline = time.time() + 2
        while not web.stems_wanted() and time.time() < deadline:
            time.sleep(0.02)
        assert web.stems_wanted()
        web.audio(np.zeros(4800, np.float32), captured=10.0)          # the original is not on this stream
        web.stem_audio(VOICE, [10.0, 10.25], [np.full(12000, 0.5, np.float32)] * 2)
        web.stem_audio(MUSIC, [10.0], [np.zeros(12000, np.float32)])
        frames = []
        while len(frames) < 3:
            t, w = struct.unpack("<dI", r.read(12))
            n, kind = w & 0xFFFFFF, w >> 24
            pcm = np.frombuffer(r.read(2 * n), "<i2")
            if n:
                frames.append((kind, t, n, pcm))
        assert [(k, t, n) for k, t, n, _ in frames] == [(1, 10.0, 6000), (1, 10.25, 6000), (2, 10.0, 6000)]
        assert abs(int(frames[1][3][3000]) - 16384) < 200                # 0.5 at 24 kHz
        r.close()
    finally:
        web.close()
