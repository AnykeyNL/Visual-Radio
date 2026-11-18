import os

import numpy as np

from visualradio.airspy_proc import AirspyProcess

FAKE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_soapy")


def test_airspy_in_a_helper_process(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", FAKE + os.pathsep + os.environ.get("PYTHONPATH", ""))
    for _ in range(2):                                   # let go and open again
        src = AirspyProcess(sample_rate=6e6, gain=12)
        try:
            assert src.sample_rate == 6e6 and src.gain == 12 and "Fake Airspy" in src.describe()
            a = src.read(200_000)
            d = np.diff(a.real)
            assert len(a) == 200_000 and np.all((d == 1) | (d == -999))   # nothing lost between frames
            src.tune(100.7e6)
            assert np.allclose(src.read(1000).imag, 0.1007)               # only samples after the retune
        finally:
            src.close()
        assert src.proc is None
