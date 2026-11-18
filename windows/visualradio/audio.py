"""Broadcast FM -> mono audio at 48 kHz, streaming (for listening and speech to text)."""

from __future__ import annotations

import numpy as np
from scipy import signal

from .rds import MPX_RATE, FmFrontEnd, _Decimator

AUDIO_RATE = 48000
_MID_RATE = 50e3


class FmAudio:
    """IQ blocks in (tuned with `tune_hz(station)`), float32 mono audio @ 48 kHz out.

    MPX (250 kHz) -> low-pass 15 kHz, decimate by 5 -> 50 kHz -> 50 us de-emphasis
    (Europe; use 75e-6 in the Americas) -> 48 kHz. Only L+R (mono) is used.
    """

    def __init__(self, sample_rate: float, deemphasis_s: float = 50e-6):
        self.fe = FmFrontEnd(sample_rate)
        self.block_multiple = self.fe.block_multiple * 5
        self.dec = _Decimator(signal.firwin(151, 15e3, fs=MPX_RATE), 5)
        a = np.exp(-1.0 / (_MID_RATE * deemphasis_s))
        self.de_b, self.de_a = [1 - a], [1, -a]
        self.de_zi = np.zeros(1)
        self.resample_pos = 0.0
        self.last = 0.0

    def tune_hz(self, station_hz: float) -> float:
        """Frequency to tune the receiver to, so the station lands where the front end wants it."""
        return station_hz - self.fe.offset_hz

    def process(self, iq: np.ndarray) -> np.ndarray:
        n = len(iq) - len(iq) % self.block_multiple
        mpx = self.fe.process(iq[:n])
        self.mpx = mpx            # FM baseband of this block (250 kHz), e.g. for RDS
        m = self.dec(mpx)
        a, self.de_zi = signal.lfilter(self.de_b, self.de_a, m, zi=self.de_zi)
        ext = np.concatenate([[self.last], a])
        step = _MID_RATE / AUDIO_RATE
        pos = np.arange(self.resample_pos, len(a), step)
        self.resample_pos = pos[-1] + step - len(a) if len(pos) else self.resample_pos - len(a)
        self.last = a[-1]
        return (np.interp(pos + 1, np.arange(len(ext)), ext) * 0.8).astype(np.float32)
