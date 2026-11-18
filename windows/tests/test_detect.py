import json
import time
import urllib.error
import urllib.request

from visualradio.songid import SongIdentifier
from visualradio.web import LiveWeb


def _post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_detect_switch_on_the_page():
    web = LiveWeb(0)
    try:
        web.set_detection({"talk": "sound", "songs": "rds", "acoustid": False, "song_lookup": True})
        state = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{web.port}/state").read())
        assert state["detect"] == {"talk": "sound", "songs": "rds", "acoustid": False, "song_lookup": True,
                                   "control": False}
        assert _post(web.port, "/detect", {"talk": "rds"})[0] == 403          # read-only page

        cmds = []
        web.control = lambda c: (cmds.append(c) or (True, "OK"))
        assert _post(web.port, "/detect", {"talk": "rds"})[0] == 200
        assert _post(web.port, "/detect", {"talk": "loud"})[0] == 400
        assert _post(web.port, "/detect", {"songs": "both"})[0] == 409        # no AcoustID key
        web.set_detection({"talk": "rds", "songs": "rds", "acoustid": True, "song_lookup": True})
        assert _post(web.port, "/detect", {"songs": "acoustid"})[0] == 200
        assert _post(web.port, "/detect", {"isolate": "all"})[0] == 409       # no Demucs
        assert _post(web.port, "/detect", {"isolate": "loud"})[0] == 400
        web.set_detection({"talk": "rds", "songs": "rds", "acoustid": True, "song_lookup": True,
                           "isolate": "off", "isolate_ready": True})
        assert _post(web.port, "/detect", {"isolate": "talk"})[0] == 200
        assert cmds == [{"cmd": "detect", "talk": "rds", "songs": None, "language": None, "isolate": None},
                        {"cmd": "detect", "talk": None, "songs": "acoustid", "language": None, "isolate": None},
                        {"cmd": "detect", "talk": None, "songs": None, "language": None, "isolate": "talk"}]
    finally:
        web.close()


def test_song_sources_switch_while_running(tmp_path):
    songs = []
    sid = SongIdentifier(songs.append, api_key="KEY", fpcalc=__file__, cache_dir=str(tmp_path),
                         log=lambda m: None, sources=("acoustid",), interval=15)
    sid.cover = lambda song: None
    sid.language = lambda song: None
    fps = []
    sid._fingerprint = lambda t0, seconds: fps.append(t0)
    assert sid.status() == "AcoustID" and sid.acoustid_ready()
    sid.radiotext("Coldplay - Yellow", 100.0)
    time.sleep(0.2)
    assert songs == []                                     # RDS off
    sid.set_sources(("rds",))
    assert sid.status() == "RDS"
    sid.radiotext("Coldplay - Yellow", 101.0)              # same text as before: still picked up
    t0 = time.time() + 1
    while time.time() < t0 and not songs:
        time.sleep(0.05)
    assert [s["title"] for s in songs] == ["Yellow"]
    sid.regions([("music", 1000.0, 1040.0)])               # AcoustID off: no lookups
    time.sleep(0.2)
    assert fps == []
    sid.set_sources(("rds", "acoustid"))
    sid.regions([("music", 1040.0, 1055.0)])
    t0 = time.time() + 1
    while time.time() < t0 and not fps:
        time.sleep(0.05)
    assert fps == [1000.0]
    sid.stop.set()
