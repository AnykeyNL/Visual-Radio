import json
import urllib.error
import urllib.request

import pytest

from visualradio import stations
from visualradio.web import LiveWeb


def test_save_and_load(tmp_path):
    path = str(tmp_path / "stations.json")
    assert stations.load(path) is None
    data = {"scanned": 123.0, "stations": [{"freq": 100.7, "name": "SKY", "snr": 31.2, "rt": ""}]}
    stations.save(data, path)
    assert stations.load(path) == data
    (tmp_path / "bad.json").write_text("{not json")
    assert stations.load(str(tmp_path / "bad.json")) is None
    (tmp_path / "list.json").write_text("[1, 2]")
    assert stations.load(str(tmp_path / "list.json")) is None


def _post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_tune_and_scan_need_control():
    web = LiveWeb(0)
    try:
        web.set_stations({"scanned": 1.0, "stations": [{"freq": 94.3, "name": "RADIO 2", "snr": 25, "rt": ""}]})
        state = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{web.port}/state").read())
        assert state["stations"]["control"] is False
        assert state["stations"]["stations"][0]["name"] == "RADIO 2"
        assert _post(web.port, "/tune", {"freq": 94.3})[0] == 403
        assert _post(web.port, "/scan", {})[0] == 403

        cmds = []
        web.control = lambda c: (cmds.append(c) or (True, "ok"))
        listing = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{web.port}/stations").read())
        assert listing["control"] is True
        assert _post(web.port, "/tune", {"freq": 94.34})[0] == 200
        assert _post(web.port, "/tune", {"freq": 120})[0] == 400
        assert _post(web.port, "/tune", {})[0] == 400
        assert _post(web.port, "/scan", {})[0] == 200
        assert _post(web.port, "/tune", {"band": "dab", "channel": "12c", "sid": "0x8201"})[0] == 200
        assert _post(web.port, "/tune", {"band": "dab", "channel": "14Z", "sid": "0x8201"})[0] == 400
        assert _post(web.port, "/tune", {"band": "dab", "channel": "12C", "sid": "radio1"})[0] == 400
        assert _post(web.port, "/scan", {"band": "dab"})[0] == 200
        assert cmds == [{"cmd": "tune", "freq": 94.3}, {"cmd": "scan", "band": "fm"},
                        {"cmd": "tune", "band": "dab", "channel": "12C", "sid": "0x8201"},
                        {"cmd": "scan", "band": "dab"}]

        web.control = lambda c: (False, "a scan is running")
        assert _post(web.port, "/scan", {}) == (409, {"ok": False, "message": "a scan is running"})
    finally:
        web.close()
