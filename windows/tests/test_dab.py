import json
import socket
import struct
import threading
import time
import urllib.request

import numpy as np

from visualradio import dab
from visualradio.web import LiveWeb


def test_channels_and_names():
    assert len(dab.SCAN_CHANNELS) == 38 and "10N" not in dab.SCAN_CHANNELS
    assert dab.CHANNELS["12C"] == 227.360
    assert dab.norm_sid("0x8DAA") == "0x8daa" and dab.norm_sid(36266) == "0x8daa"
    assert dab.clean("NPO Radio 1     ") == "NPO Radio 1"
    data = {"ensembles": [
        {"channel": "11C", "services": [{"sid": "0x8504", "name": "Radio 10"}, {"sid": "0x8505", "name": "100%NL"}]},
        {"channel": "12C", "services": [{"sid": "0x8201", "name": "NPO Radio 1"}, {"sid": "0x8202", "name": "NPO Radio 2"}]}]}
    assert dab.find_service(data, "radio 1") == ("12C", "0x8201", "NPO Radio 1")     # not Radio 10
    assert dab.find_service(data, "Radio 10") == ("11C", "0x8504", "Radio 10")
    assert dab.find_service(data, "0x8202")[2] == "NPO Radio 2"
    assert dab.find_service(data, "radio", channel="11C")[1] == "0x8504"
    assert dab.find_service(data, "nothing") is None


def test_resampler_is_continuous():
    t = np.arange(32000 * 2) / 32000
    x = np.sin(2 * np.pi * 440 * t)
    rs = dab.Resampler(32000)
    y = np.concatenate([rs(x[i:i + 1536]) for i in range(0, len(x), 1536)])
    assert abs(len(y) - 96000) <= 2
    ref = np.sin(2 * np.pi * 440 * np.arange(len(y)) / 48000)
    lag = int(np.argmax(np.correlate(y[2000:4000], ref[2000:2200], "valid")))   # filter delay
    err = y[3000:90000] - ref[3000 - lag:90000 - lag]
    assert np.max(np.abs(err)) < 0.02                                           # no clicks at block edges


def test_read_frames_parses_pcm_stream():
    a, b = socket.socketpair()
    pcm = (np.arange(8, dtype="<i2")).tobytes()
    a.sendall(b"HTTP/1.0 200 OK\r\nContent-Type: application/octet-stream\r\n\r\n"
              + b"PCM1" + struct.pack("<II", 48000, len(pcm)) + pcm[:5])
    threading.Timer(0.2, lambda: (a.sendall(pcm[5:]), a.close())).start()
    b.settimeout(1.0)
    frames = list(dab.read_frames(b))
    assert len(frames) == 1 and frames[0][0] == 48000 and list(frames[0][1]) == list(range(8))


def test_radio_on_simulated_welle():
    sim = dab.SimWelle(lock_time=0.2).start("12C")
    infos = []
    radio = dab.DabRadio(sim, "12C", "0x8201", log=lambda m: None, on_info=infos.append)
    try:
        got, t0 = [], time.time()
        while time.time() - t0 < 3.5:
            a, _ = radio.read(0.2)
            assert len(a) % 6 == 0
            got.append(a)
        a = np.concatenate(got)
        assert len(a) > 48000 * 2 and a.dtype == np.float32                 # 32 kHz station -> 48 kHz
        assert infos and infos[-1]["name"] == "NPO Radio 1" and infos[-1]["ensemble"] == "NPO"
        assert infos[-1]["dls"] and infos[-1]["slide"] >= 1 and radio.slide[1] == "image/png"
        radio.tune("8A", "0x8702")                                           # another ensemble
        t0 = time.time()
        while time.time() - t0 < 4 and not (radio.info.get("name") == "Sublime"):
            radio.read(0.2)
        assert radio.info["ensemble"] == "Lokaal Mux" and sim.channel == "8A"
        a, _ = radio.read(1.0)
        assert len(a)
        dab.SETTLE = 0.4
        res = radio.scan(channels=["5A", "11C", "12C"], settle=0.4)
        assert [e["channel"] for e in res["ensembles"]] == ["11C", "12C"]
        assert res["ensembles"][1]["services"][0]["name"] == "NPO 3FM"         # sorted by name
    finally:
        dab.SETTLE = 3.0
        radio.close()
        sim.close()


def test_web_station_keys_and_slide():
    web = LiveWeb(0)
    try:
        web.set_station(100.7e6, "SKY", band="fm")
        assert web.station["key"] == "fm:100.7"
        web.set_station(None, "NPO Radio 1", retuned=True, band="dab", channel="12C", sid="0x8201", mhz=227.36)
        st = web.station
        assert st["band"] == "dab" and st["key"] == "dab:12C:0x8201" and st["freq"] is None
        web.set_slide(b"\x89PNG....", "image/png", 3)
        assert web.station["slide"] == "/slide?v=3"
        r = urllib.request.urlopen(f"http://127.0.0.1:{web.port}/slide?v=3")
        assert r.read() == b"\x89PNG...." and r.headers["Content-Type"] == "image/png"
        web.set_station(94.3e6, "", retuned=True, band="fm")
        assert "slide" not in web.station and web.slide is None
        web.set_dab_stations({"scanned": 1.0, "ensembles": [{"channel": "12C", "services": []}]}, ready=True)
        state = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{web.port}/state").read())
        assert state["stations"]["dab"]["ready"] is True and state["stations"]["dab"]["ensembles"][0]["channel"] == "12C"
    finally:
        web.close()


def test_radio_after_a_scan_tunes_back():
    sim = dab.SimWelle(lock_time=0.2).start("5A")
    try:
        assert dab.scan_channel(sim, "11C", settle=0.3)["channel"] == "11C"
        assert sim.channel == "11C"
        radio = dab.DabRadio(sim, "12C", "0x8201", log=lambda m: None)
        try:
            t0, got = time.time(), 0
            while time.time() - t0 < 4 and not got:
                got = len(radio.read(0.3)[0])
            assert got and sim.channel == "12C"
        finally:
            radio.close()
    finally:
        sim.close()
