import json
import time
import urllib.request
from types import SimpleNamespace as NS

import numpy as np

from visualradio.web import LiveWeb


def _events(port, want, timeout=5):
    """Read Server-Sent Events until `want` event names have been seen."""
    got = []
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/events", timeout=timeout) as r:
        event = None
        t0 = time.time()
        while time.time() - t0 < timeout and len(got) < want:
            line = r.readline().decode().rstrip("\n")
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                got.append((event, json.loads(line[6:])))
    return got


def test_live_web_page_and_events():
    web = LiveWeb(0)
    try:
        page = urllib.request.urlopen(f"http://127.0.0.1:{web.port}/").read().decode()
        assert "EventSource" in page and "<title>" in page
        web.set_station(100.7e6, "BIG FM", "Now playing: test")
        web.transcript(NS(start=1.0, kind="music", language="en", text="la la la", duration_s=3.0))

        import threading
        def later():
            time.sleep(0.5)
            web.transcript(NS(start=2.0, kind="speech", language="nl", text="goedenavond", duration_s=2.0))
            web.audio(np.random.default_rng(0).standard_normal(4800).astype(np.float32) * 0.1, 2)
        threading.Thread(target=later, daemon=True).start()

        ev = _events(web.port, 3)
        assert ev[0][0] == "hello"
        hello = ev[0][1]
        assert {k: hello["station"][k] for k in ("freq", "name", "rt")} == {"freq": 100.7, "name": "BIG FM", "rt": "Now playing: test"}
        assert [h["kind"] for h in hello["history"]] == ["music"]
        names = [e for e, _ in ev[1:]]
        assert "transcript" in names and "level" in names
        lvl = dict(ev[1:])["level"]
        assert len(lvl["bands"]) == 48 and lvl["pending"] == 2
        assert all(0 <= b <= 1 for b in lvl["bands"])
    finally:
        web.close()
