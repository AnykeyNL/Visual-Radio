"""FM demodulation and RDS decoding (station name, PI code, radio text).

Signal chain for one station
----------------------------
    IQ @ 6 or 3 MSPS  (station sits fs/24 above the tuned frequency)
      -> shift station to 0 Hz, low-pass + decimate to 750 kHz, then 250 kHz
      -> FM demodulate                    = "MPX" baseband (mono audio, 19 kHz
                                            stereo pilot, 38 kHz stereo, 57 kHz RDS)
      -> RDS: shift 57 kHz to 0 Hz, low-pass 2.4 kHz, decimate to 25 kHz
      -> carrier phase: locked to the 19 kHz pilot (x3), or, for mono stations
         without a pilot, recovered by squaring the signal
      -> bi-phase matched filter, symbol clock recovery at 1187.5 bit/s
      -> differential decoding -> bits
      -> block sync with the RDS checkwords (26-bit blocks, 4 per group)
      -> groups 0A/0B = station name (PS), 2A/2B = radio text (RT)

The front end works block by block (keeps filter state), so it can run while
samples stream in from the Airspy without holding seconds of raw IQ in memory.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

MPX_RATE = 250e3          # FM-demodulated baseband rate
RDS_RATE = 25e3           # rate after the RDS low-pass
BIT_RATE = 1187.5         # RDS bits per second
FM_DEVIATION = 75e3

# ---------------------------------------------------------------- RDS coding

POLY = 0x1B9              # g(x) = x^10+x^8+x^7+x^5+x^4+x^3+1 without the x^10 term
OFFSETS = {"A": 0x0FC, "B": 0x198, "C": 0x168, "C'": 0x350, "D": 0x1B4}
_OFFSET_BY_VALUE = {v: k for k, v in OFFSETS.items()}


def crc10(data16: int) -> int:
    """RDS checkword (before adding the offset word) for 16 data bits."""
    reg = 0
    for i in range(15, -1, -1):
        fb = ((reg >> 9) & 1) ^ ((data16 >> i) & 1)
        reg = (reg << 1) & 0x3FF
        if fb:
            reg ^= POLY
    return reg


_CRC_TABLE = None
_NAMES = ("A", "B", "C", "C'", "D")
_SYNDROME_TO_NAME = np.full(1024, -1, dtype=np.int8)
for _i, _name in enumerate(_NAMES):
    _SYNDROME_TO_NAME[OFFSETS[_name]] = _i


def _block_offsets(words: np.ndarray) -> np.ndarray:
    """For each 26-bit word: index into _NAMES of the offset that makes it valid, or -1."""
    global _CRC_TABLE
    if _CRC_TABLE is None:
        _CRC_TABLE = np.array([crc10(d) for d in range(1 << 16)], dtype=np.int64)
    return _SYNDROME_TO_NAME[_CRC_TABLE[words >> 10] ^ (words & 0x3FF)]


def encode_group(blocks, version_b=False):
    """Four 16-bit words -> 104 bits (used by the simulator and the tests)."""
    names = ("A", "B", "C'" if version_b else "C", "D")
    bits = []
    for word, name in zip(blocks, names):
        block = (word << 10) | (crc10(word) ^ OFFSETS[name])
        bits.extend((block >> i) & 1 for i in range(25, -1, -1))
    return bits


def ps_groups(pi: int, name: str, pty: int = 10):
    """Group 0A sequence (4 groups) carrying an 8-character station name."""
    name = (name + " " * 8)[:8]
    out = []
    for seg in range(4):
        b = (0 << 12) | (0 << 11) | (pty << 5) | seg
        c1, c2 = name[2 * seg], name[2 * seg + 1]
        out.append((pi, b, 0xE0CD, (ord(c1) << 8) | ord(c2)))
    return out


def rt_groups(pi: int, text: str, pty: int = 10):
    """Group 2A sequence carrying radio text (up to 64 characters)."""
    text = text[:64]
    if len(text) < 64:
        text += "\r"
    text += " " * (-len(text) % 4)
    out = []
    for seg in range(len(text) // 4):
        b = (2 << 12) | (0 << 11) | (pty << 5) | seg
        s = text[4 * seg:4 * seg + 4]
        out.append((pi, b, (ord(s[0]) << 8) | ord(s[1]), (ord(s[2]) << 8) | ord(s[3])))
    return out


# ------------------------------------------------------------ FM front end


class _Decimator:
    """Streaming FIR low-pass + decimate, keeping state between blocks."""

    def __init__(self, taps, factor):
        self.h = np.asarray(taps, dtype=np.float32)
        self.D = factor
        self.tail_len = -(-(len(self.h) - 1) // factor) * factor  # multiple of D
        self.tail = None

    def __call__(self, x):
        if len(x) % self.D:
            raise ValueError("block length must be a multiple of the decimation factor")
        if self.tail is None:
            self.tail = np.zeros(self.tail_len, dtype=x.dtype)
        xe = np.concatenate([self.tail, x])
        y = signal.upfirdn(self.h, xe, 1, self.D)
        start = self.tail_len // self.D
        self.tail = xe[-self.tail_len:]
        return y[start:start + len(x) // self.D].astype(x.dtype)


class FmFrontEnd:
    """IQ blocks in, FM-demodulated MPX (float32 @ 250 kHz) out."""

    def __init__(self, sample_rate: float):
        d1 = sample_rate / 750e3
        if abs(d1 - round(d1)) > 1e-9:
            raise ValueError("RDS decoding needs a sample rate that is a multiple of 750 kHz (3 or 6 MSPS)")
        self.fs = sample_rate
        self.d1 = int(round(d1))
        self.block_multiple = 24 * self.d1   # keeps the mixer table and both decimators aligned
        self.offset_hz = sample_rate / 24    # where the station must sit in the IQ
        self.lo = np.exp(-2j * np.pi * np.arange(24) / 24).astype(np.complex64)
        self.dec1 = _Decimator(signal.firwin(8 * self.d1 + 1, 150e3, fs=sample_rate), self.d1)
        self.dec2 = _Decimator(signal.firwin(101, 110e3, fs=750e3), 3)
        self.prev = np.complex64(0)

    def process(self, iq: np.ndarray) -> np.ndarray:
        n = len(iq) - len(iq) % self.block_multiple
        x = iq[:n] * np.tile(self.lo, n // 24)
        y = self.dec2(self.dec1(x))
        z = np.concatenate([[self.prev], y])
        self.prev = y[-1]
        phase_step = np.angle(z[1:] * np.conj(z[:-1]))
        return (phase_step * (MPX_RATE / (2 * np.pi * FM_DEVIATION))).astype(np.float32)


# ------------------------------------------------------------- RDS decoder


@dataclass
class RdsInfo:
    pi: int | None = None
    ps: str | None = None
    rt: str | None = None
    groups: int = 0
    blocks_ok: int = 0
    ps_segments: dict = field(default_factory=dict)  # seg -> Counter of 2-char strings

    @property
    def ps_complete(self) -> bool:
        return len(self.ps_segments) == 4

    @property
    def pi_hex(self) -> str:
        return f"{self.pi:04X}" if self.pi is not None else ""


def _char(code: int) -> str:
    return chr(code) if 0x20 <= code < 0x7F else ("" if code in (0x0D, 0x00) else "?")


def _baseband(mpx: np.ndarray, use_pilot: bool) -> np.ndarray:
    """MPX @ 250 kHz -> complex RDS baseband @ 25 kHz, carrier phase removed."""
    n = np.arange(len(mpx))
    lp = signal.firwin(511, 2.4e3, fs=MPX_RATE)
    x57 = signal.fftconvolve(mpx * np.exp(-2j * np.pi * 57e3 / MPX_RATE * n), lp, mode="same")
    b = x57[::10]
    if use_pilot:
        box = np.ones(5001) / 5001  # ~50 Hz wide: tracks small pilot offsets
        q = signal.fftconvolve(mpx * np.exp(-2j * np.pi * 19e3 / MPX_RATE * n), box, mode="same")[::10]
        q3 = q ** 3
        b = b * np.conj(q3) / np.maximum(np.abs(q3), 1e-12)
        avg_len = 6001   # pilot already removed the drift; average long
    else:
        avg_len = 501    # free running: short average to follow a few Hz offset
    sq = signal.fftconvolve(b * b, np.ones(avg_len) / avg_len, mode="same")
    theta = np.unwrap(np.angle(sq)) / 2
    return b * np.exp(-1j * theta)


def _bits_from_baseband(b: np.ndarray) -> np.ndarray:
    r = b.real
    sps = RDS_RATE / BIT_RATE
    half = int(round(sps / 2))
    kernel = np.concatenate([np.ones(half), -np.ones(half)])
    m = np.convolve(r, kernel[::-1], mode="same")

    # Symbol clock: m^2 has a tone at the bit rate whose phase marks the bit centres.
    n = np.arange(len(m))
    rot = (m * m) * np.exp(-2j * np.pi * BIT_RATE / RDS_RATE * n)
    win = int(RDS_RATE * 0.1)
    lp = signal.fftconvolve(rot, np.ones(win) / win, mode="same")
    phi = np.unwrap(-np.angle(lp))
    c = BIT_RATE / RDS_RATE * n - phi / (2 * np.pi)
    idx = np.nonzero(np.diff(np.floor(c)) > 0)[0]
    idx = idx[(idx > win // 2) & (idx < len(m) - win // 2 - 1)]  # skip filter edges
    if len(idx) < 2:
        return np.zeros(0, dtype=np.uint8)
    k = np.floor(c[idx + 1])
    frac = (k - c[idx]) / (c[idx + 1] - c[idx])
    vals = m[idx] + frac * (m[idx + 1] - m[idx])
    t = (vals > 0).astype(np.uint8)
    return t[1:] ^ t[:-1]   # differential decoding


def _groups_from_bits(bits: np.ndarray):
    """Yield (A_or_None, B, C_or_None, D, version_b) for every group found."""
    n = len(bits)
    if n < 104:
        return [], 0
    # 26-bit word starting at every bit position.
    words = np.zeros(n - 25, dtype=np.int64)
    for i in range(26):
        words = (words << 1) | bits[i:n - 25 + i]
    offs = [_NAMES[i] if i >= 0 else None for i in _block_offsets(words)]

    groups, ok = [], sum(o is not None for o in offs)
    q = 26
    while q + 52 < len(offs):
        if offs[q] == "B" and offs[q + 52] == "D":
            a = offs[q - 26] == "A"
            c_off = offs[q + 26]
            if a or c_off in ("C", "C'"):
                val = lambda p: int(words[p]) >> 10  # noqa: E731
                groups.append((val(q - 26) if a else None, val(q),
                               val(q + 26) if c_off in ("C", "C'") else None,
                               val(q + 52), c_off == "C'"))
                q += 104
                continue
        q += 1
    return groups, ok


def _decode_groups(groups, blocks_ok) -> RdsInfo:
    info = RdsInfo(groups=len(groups), blocks_ok=blocks_ok)
    pis = Counter()
    rt_ab, rt_chars = None, {}
    for a, b, c, d, version_b in groups:
        if a is not None:
            pis[a] += 1
        elif version_b and c is not None:
            pis[c] += 1
        gtype, vb = b >> 12, (b >> 11) & 1
        if gtype == 0:
            seg = b & 3
            info.ps_segments.setdefault(seg, Counter())[_char(d >> 8) + _char(d & 0xFF)] += 1
        elif gtype == 2:
            ab = (b >> 4) & 1
            if rt_ab is not None and ab != rt_ab:
                rt_chars = {}            # A/B flag toggled: new text
            rt_ab = ab
            seg = b & 0xF
            if vb == 0 and c is not None:
                for i, code in enumerate((c >> 8, c & 0xFF, d >> 8, d & 0xFF)):
                    rt_chars[seg * 4 + i] = code
            elif vb == 1:
                for i, code in enumerate((d >> 8, d & 0xFF)):
                    rt_chars[seg * 2 + i] = code
    if pis:
        info.pi = pis.most_common(1)[0][0]
    if info.ps_segments:
        info.ps = "".join(
            info.ps_segments[s].most_common(1)[0][0] if s in info.ps_segments else "__"
            for s in range(4)).strip()
    if rt_chars:
        end = min([i for i, ch in rt_chars.items() if ch == 0x0D] + [max(rt_chars) + 1])
        text = "".join(_char(rt_chars[i]) if i in rt_chars else "_" for i in range(end))
        info.rt = " ".join(text.split()) or None
    return info


def decode_rds(mpx: np.ndarray) -> RdsInfo:
    """Decode RDS from FM-demodulated MPX sampled at 250 kHz."""
    best = None
    for use_pilot in (True, False):
        bits = _bits_from_baseband(_baseband(mpx, use_pilot))
        groups, ok = _groups_from_bits(bits)
        info = _decode_groups(groups, ok)
        if best is None or (info.groups, info.blocks_ok) > (best.groups, best.blocks_ok):
            best = info
    return best


def read_station_rds(source, freq_hz: float, max_seconds: float = 4.0,
                     block_seconds: float = 0.25) -> RdsInfo:
    """Tune to a station, stream samples through the FM front end and decode RDS.

    Stops early once the full station name has been received twice.
    """
    fe = FmFrontEnd(source.sample_rate)
    source.tune(freq_hz - fe.offset_hz)
    block = int(source.sample_rate * block_seconds)
    block -= block % fe.block_multiple
    mpx_parts = []
    info = RdsInfo()
    elapsed = 0.0
    while elapsed < max_seconds:
        mpx_parts.append(fe.process(source.read(block)))
        elapsed += block_seconds
        if elapsed >= 1.0 and (round(elapsed / block_seconds) % 4 == 0 or elapsed >= max_seconds):
            info = decode_rds(np.concatenate(mpx_parts))
            if info.ps_complete and all(sum(c.values()) >= 2 for c in info.ps_segments.values()):
                break
            if elapsed >= 2.5 and info.groups == 0:
                break  # nothing at all after 2.5 s: most likely no RDS on this station
    return info
