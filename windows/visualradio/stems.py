"""Listen to the voice only, or to the music only, in the browser.

While someone on the web page picks "Voice" or "Music", the live audio is cut
into pieces of a few seconds and split by Demucs (visualradio.vocals) into the voice
and everything else. Both come out with the time of the original audio, so the
page plays them in step with the text, exactly like the normal sound.

Each piece is split with a few seconds of the audio before it as context, the
last part is held back (Demucs is less sure at the edge) and redone with the
next piece, and the joins are crossfaded, so the result is seamless. The voice
comes out about `window + hold` seconds (plus the time Demucs takes) after the
radio; with a playback delay (--delay) larger than that you never notice.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

VOICE, MUSIC = 1, 2


class StemSplitter:
    """feed() the live 48 kHz audio; emit(kind, times, blocks) gets the voice
    (kind 1) and the music (kind 2) in frames of `frame` seconds.
    split(mono48) -> (voice48, music48), as long as the input."""

    def __init__(self, split, emit, window=2.5, context=4.0, hold=0.75, fade=0.05, frame=0.25,
                 rate=48000, log=print, max_behind=30.0):
        self.split, self.emit, self.log = split, emit, log
        self.rate = rate
        self.W, self.C, self.H = int(window * rate), int(context * rate), int(hold * rate)
        self.F, self.frame = int(fade * rate), int(frame * rate)
        self.max_behind = int(max_behind * rate)
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.stop = threading.Event()
        self.failed = None
        self.busy = 0.0                 # seconds spent splitting, last piece (for the page)
        self._clear()
        threading.Thread(target=self._run, daemon=True).start()

    def _clear(self):
        self.buf = np.zeros(0, np.float32)
        self.base = 0                   # absolute sample number of buf[0]
        self.total = 0                  # samples fed so far
        self.anchors = deque()          # (sample number, PC time it was received)
        self.emitted = None             # output is done up to this sample number
        self.tail = None                # (voice, music) just past `emitted`, for the crossfade
        self.gen = getattr(self, "gen", 0) + 1

    def reset(self):
        """Another station, or nobody listens to a stem any more: start afresh."""
        with self.lock:
            if self.total:
                self._clear()

    def _time(self, i):
        t = None
        for a, at in self.anchors:
            if a > i:
                break
            t = at + (i - a) / self.rate
        if t is None:
            a, at = self.anchors[0]
            t = at + (i - a) / self.rate
        return t

    def feed(self, block, captured):
        block = np.asarray(block, dtype=np.float32)
        if not len(block):
            return
        with self.lock:
            if self.anchors and abs(captured - self._time(self.total)) > 0.5:
                self._clear()           # a jump in time (retune, gap): no seamless join possible
            self.anchors.append((self.total, captured))
            self.buf = np.concatenate([self.buf, block])
            self.total += len(block)
            if self.emitted is None:
                self.emitted = self.base
            if self.total - self.emitted > self.max_behind:     # the splitter can't keep up
                self.emitted = self.total - self.W - self.H
                self.tail = None
                self.log("Voice / music split can't keep up (Demucs on the CPU?), skipping ahead")
        self.wake.set()

    def _run(self):
        while not self.stop.is_set():
            self.wake.wait(0.5)
            self.wake.clear()
            while not self.stop.is_set():
                with self.lock:
                    if self.emitted is None or self.total < self.emitted + self.W + self.H:
                        break
                    start = max(self.base, self.emitted - self.C)
                    end = self.emitted + self.W + self.H
                    seg = self.buf[start - self.base:end - self.base].copy()
                    gen, emitted, tail = self.gen, self.emitted, self.tail
                t0 = time.time()
                try:
                    voice, music = self.split(seg)
                except Exception as exc:              # out of GPU memory, ...
                    if self.failed is None:
                        self.log(f"Voice / music split failed: {exc}")
                    self.failed = str(exc)
                    with self.lock:
                        if gen == self.gen and self.emitted == emitted:
                            self.emitted, self.tail = emitted + self.W, None
                    time.sleep(2.0)
                    continue
                self.busy = time.time() - t0
                self.failed = None
                e0 = emitted - start
                out = []
                for k, x in enumerate((voice, music)):
                    y = np.array(x[e0:e0 + self.W], dtype=np.float32)
                    if tail is not None and self.F:
                        ramp = np.linspace(0, 1, self.F, dtype=np.float32)
                        y[:self.F] = tail[k] * (1 - ramp) + y[:self.F] * ramp
                    out.append(y)
                new_tail = tuple(np.array(x[e0 + self.W:e0 + self.W + self.F], dtype=np.float32)
                                 for x in (voice, music))
                with self.lock:
                    if gen != self.gen or self.emitted != emitted:   # reset / skipped ahead meanwhile
                        continue
                    self.emitted, self.tail = emitted + self.W, new_tail
                    times = [self._time(emitted + i) for i in range(0, self.W, self.frame)]
                    keep = self.emitted - self.C         # context for the next piece
                    if keep > self.base:
                        self.buf = self.buf[keep - self.base:]
                        self.base = keep
                        while len(self.anchors) > 1 and self.anchors[1][0] <= keep:
                            self.anchors.popleft()
                for kind, y in ((VOICE, out[0]), (MUSIC, out[1])):
                    blocks = [y[i:i + self.frame] for i in range(0, self.W, self.frame)]
                    try:
                        self.emit(kind, times, blocks)
                    except Exception:
                        pass

    def lag(self):
        """Seconds from the radio to its voice / music: play them at least this late."""
        return (self.W + self.H) / self.rate + max(self.busy, 0.6) * 1.5 + 0.6

    def close(self):
        self.stop.set()
        self.wake.set()
