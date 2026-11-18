"""RDS decoder tests (no hardware):  python -m pytest tests"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from visualradio import rds  # noqa: E402
from visualradio.sources import SimulatedSource  # noqa: E402


def _mpx_with_rds(name, text, seconds=2.5, pilot=True, pilot_offset_hz=0.0, noise=0.02):
    """FM baseband at 250 kHz with mono audio, optional pilot and RDS."""
    fs = rds.MPX_RATE
    t = np.arange(int(fs * seconds)) / fs
    groups = rds.ps_groups(0x8203, name) + rds.rt_groups(0x8203, text)
    bits = np.array([b for g in groups * 10 for b in rds.encode_group(g)], dtype=np.uint8)
    tx = np.bitwise_xor.accumulate(bits)
    pos = t * rds.BIT_RATE + 0.37
    k = pos.astype(int)
    a = np.where(tx[k % len(tx)] == 1, 1.0, -1.0)
    sym = np.where(pos - k >= 0.5, -a, a)
    ph = 2 * np.pi * (19e3 + pilot_offset_hz) * t + 1.0
    m = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 3100 * t)
    if pilot:
        m += 0.09 * np.cos(ph)
    m += 0.04 * sym * np.cos(3 * ph + 0.4)
    return (m + noise * np.random.default_rng(0).standard_normal(len(t))).astype(np.float32)


def test_checkword_roundtrip():
    bits = rds.encode_group((0x8203, 0x0141, 0xE0CD, 0x5241))
    assert len(bits) == 104
    words = [int("".join(map(str, bits[i:i + 26])), 2) for i in range(0, 104, 26)]
    names = [rds._NAMES[i] for i in rds._block_offsets(np.array(words, dtype=np.int64))]
    assert names == ["A", "B", "C", "D"]


def test_decode_name_and_text_with_pilot():
    info = rds.decode_rds(_mpx_with_rds("RADIO 1", "Hello from the test"))
    assert (info.pi_hex, info.ps, info.rt) == ("8203", "RADIO 1", "Hello from the test")


def test_decode_mono_station_with_carrier_offset():
    info = rds.decode_rds(_mpx_with_rds("MONO FM", "x", pilot=False, pilot_offset_hz=3.0))
    assert info.ps == "MONO FM"


def test_no_rds_gives_nothing():
    fs = rds.MPX_RATE
    t = np.arange(int(fs * 2)) / fs
    info = rds.decode_rds((0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32))
    assert info.groups == 0 and info.ps is None


def test_full_chain_from_simulated_iq():
    src = SimulatedSource(sample_rate=3e6, stations=[(95.0, 30, "SIMTEST", "Radio text works"),
                                                     (96.0, 30)])
    info = rds.read_station_rds(src, 95.0e6, max_seconds=4)
    assert info.ps == "SIMTEST" and info.pi_hex == "8201"
    assert rds.read_station_rds(src, 96.0e6, max_seconds=3).ps is None
