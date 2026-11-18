import json
import urllib.request

import numpy as np

from visualradio import sysload
from visualradio.web import LiveWeb


def test_parse_nvidia_smi():
    d = sysload.parse_smi("NVIDIA GeForce RTX 4070, 37, 6144, 12282, 61, 95.32, 200.00\n")
    assert d == {"name": "NVIDIA GeForce RTX 4070", "util": 37, "mem_used": 6.0, "mem_total": 11.99,
                 "temp": 61, "power": 95.3, "power_max": 200}
    d = sysload.parse_smi("Some GPU, 5, 100, 1024, 40, [N/A], [N/A]")
    assert "power" not in d and d["util"] == 5
    assert sysload.parse_smi("") == {}


def test_sample_never_fails_and_merges_extra():
    s = sysload.SystemLoad(lambda d: None, extra=lambda: {"steps": {"whisper": 1.2}, "piece": 15.0})
    s.gpu = None
    d = s.sample()
    assert d["gpu"] is None and d["steps"] == {"whisper": 1.2}
    assert s.cpu is not None
    assert 0 <= d["cpu"] <= 100 and d["ram_total"] > 0 and d["cores"] >= 1
    s.extra = lambda: 1 / 0                       # a broken extra is just left out
    assert "t" in s.sample()


def test_web_keeps_two_minutes_of_load():
    web = LiveWeb(0)
    try:
        for i in range(130):
            web.set_sys({"cpu": i % 100, "proc": 1.0, "gpu": {"util": 50, "mem_used": 3.0, "mem_total": 12.0}})
        st = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{web.port}/state").read())
        assert len(st["sys_hist"]) == 120 and st["sys_hist"][-1] == [29, 1.0, 50, 25.0]
        assert st["sys"]["cpu"] == 29
    finally:
        web.close()


def test_transcriber_times_its_steps():
    from types import SimpleNamespace as NS
    from visualradio.transcribe import Transcriber

    class W:
        def transcribe(self, audio, **kw):
            return iter([NS(text=" hallo", no_speech_prob=0.1, words=[])]), NS(language="nl")

    class C:
        def regions(self, a):
            return [("speech", 0.0, len(a) / 16000)]

    tr = Transcriber(backend=W(), classifier=C())
    tr.submit("x", 0.0, (np.random.default_rng(0).standard_normal(48000 * 6) * .1).astype(np.float32))
    tr.close(10)
    info = tr.timing_info()
    assert set(info["steps"]) == {"classify", "whisper"}       # no isolation, no speakers
    assert info["backlog"] == 0 and 6 < info["piece"] < 15


def test_cpu_without_psutil(monkeypatch):
    import builtins
    real = builtins.__import__

    def no_psutil(name, *a, **k):
        if name == "psutil":
            raise ImportError("no psutil")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_psutil)
    s = sysload.SystemLoad(lambda d: None)
    s.gpu = None
    assert s.cpu.name in ("/proc", "Windows")
    sum(i * i for i in range(300000))           # some work to measure
    d = s.sample()
    assert 0 <= d["cpu"] <= 100 and 0 <= d["proc"] <= 100 and d["ram_total"] > 0
    assert 0 < d["ram"] < 100
