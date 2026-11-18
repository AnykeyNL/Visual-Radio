"""Tests that run without hardware:  python -m pytest tests   (or: python tests/test_scanner.py)"""

import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from visualradio import sources  # noqa: E402
from visualradio.scanner import plan_centers, power_spectrum, scan  # noqa: E402
from visualradio.sources import DEFAULT_SIM_STATIONS, SimulatedSource  # noqa: E402


def test_plan_covers_band():
    centers = plan_centers(87.5e6, 108e6, 6e6, 0.8)
    half = 6e6 * 0.8 / 2
    assert centers[0] - half <= 87.5e6 and centers[-1] + half >= 108e6
    for a, b in zip(centers, centers[1:]):
        assert b - a <= 2 * half  # no gaps between windows


def test_power_spectrum_sums_to_signal_power():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(8192 * 16) + 1j * rng.standard_normal(8192 * 16)) * 0.1
    assert abs(power_spectrum(x, 8192).sum() / np.mean(np.abs(x) ** 2) - 1) < 0.05


def test_finds_simulated_stations_above_threshold():
    result = scan(SimulatedSource(), threshold_db=10)
    found = {round(s.freq_mhz, 1) for s in result.stations}
    expected = {s[0] for s in DEFAULT_SIM_STATIONS if s[1] >= 11}
    too_weak = {s[0] for s in DEFAULT_SIM_STATIONS if s[1] <= 9}
    assert expected <= found, f"missed {expected - found}"
    assert not (too_weak & found), f"reported weak {too_weak & found}"
    assert found <= {s[0] for s in DEFAULT_SIM_STATIONS}, "false stations"
    for s in result.stations:
        true_snr = {st[0]: st[1] for st in DEFAULT_SIM_STATIONS}[round(s.freq_mhz, 1)]
        assert abs(s.snr_db - true_snr) < 2


def test_close_neighbours_and_band_edges():
    stations = [(87.5, 30), (95.0, 40), (95.2, 15), (95.5, 20), (108.0, 25)]
    result = scan(SimulatedSource(stations=stations), threshold_db=10)
    assert [round(s.freq_mhz, 1) for s in result.stations] == [87.5, 95.0, 95.2, 95.5, 108.0]


class _FakeSoapy(types.SimpleNamespace):
    """Just enough of the SoapySDR API to exercise AirspySource."""


def _fake_soapy(devices):
    sim = SimulatedSource(stations=[(100.0, 30)])

    class Device:
        @staticmethod
        def enumerate(args):
            return devices

        def __init__(self, info):
            pass

        def listSampleRates(self, d, c): return [3e6, 6e6]
        def setSampleRate(self, d, c, r): self.rate = r
        def getSampleRate(self, d, c): return self.rate
        def hasGainMode(self, d, c): return True
        def setGainMode(self, d, c, m): pass
        def setGain(self, d, c, g): self.gain = g
        def getGain(self, d, c): return self.gain
        def setupStream(self, d, fmt): return "stream"
        def activateStream(self, s): pass
        def deactivateStream(self, s): pass
        def closeStream(self, s): pass
        def getStreamMTU(self, s): return 65536
        def setFrequency(self, d, c, f): sim.tune(f)

        def readStream(self, s, bufs, n, timeoutUs=0):
            bufs[0][:n] = sim.read(n)
            return types.SimpleNamespace(ret=n)

    return _FakeSoapy(Device=Device, SOAPY_SDR_RX=0, SOAPY_SDR_CF32="CF32")


def test_airspy_not_found_gives_clear_message(monkeypatch):
    monkeypatch.setattr(sources, "_import_soapy", lambda: _fake_soapy([]))
    try:
        sources.AirspySource()
    except sources.DeviceError as exc:
        assert "No Airspy Mini found" in str(exc)
    else:
        raise AssertionError("expected DeviceError")


def test_airspy_path_with_fake_driver(monkeypatch):
    fake = _fake_soapy([{"driver": "airspy", "label": "AirSpy Mini", "serial": "abc123"}])
    monkeypatch.setattr(sources, "_import_soapy", lambda: fake)
    src = sources.AirspySource(gain=12)
    assert "serial abc123" in src.describe()
    result = scan(src, threshold_db=10)
    assert [round(s.freq_mhz, 1) for s in result.stations] == [100.0]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
