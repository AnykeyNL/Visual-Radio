import json
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pytest

from visualradio import songid
from visualradio.songid import SongIdentifier, parse_radiotext
from visualradio.web import LiveWeb


@pytest.mark.parametrize("rt, want", [
    ("Now playing: Coldplay - Yellow", ("Coldplay", "Yellow")),
    ("NU: Doe Maar - Smoorverliefd", ("Doe Maar", "Smoorverliefd")),
    ("Nu op Radio 2: Adele – Hello", ("Adele", "Hello")),
    ("Rolling in the Deep by Adele", ("Adele", "Rolling in the Deep")),
    ("Queen - Bohemian Rhapsody  ·  bel de studio 0800 1234", ("Queen", "Bohemian Rhapsody")),
    ("Het nieuws van 14 uur", None),
    ("", None),
])
def test_parse_radiotext(rt, want):
    assert parse_radiotext(rt) == want


BATCH_ANSWER = {
    "status": "ok",
    "fingerprints": [
        {"index": "0", "results": []},
        {"index": "1", "results": [{"id": "x", "score": 0.91, "recordings": [{
            "id": "r1", "title": "Yellow",
            "artists": [{"name": "Coldplay"}],
            "releasegroups": [{"id": "rg-single", "type": "Single", "title": "Yellow"},
                              {"id": "rg-album", "type": "Album", "title": "Parachutes"}]}]}]},
        {"index": "2", "results": [{"id": "y", "score": 0.3, "recordings": [{"title": "Other"}]}]},
    ],
}


def test_parse_lookup_batch_picks_best_and_album():
    song = SongIdentifier.parse_lookup(BATCH_ANSWER)
    assert song == {"artist": "Coldplay", "title": "Yellow", "album": "Parachutes", "rgid": "rg-album",
                    "mbid": "r1", "score": 0.91, "source": "AcoustID"}
    assert SongIdentifier.parse_lookup({"status": "ok", "results": []}) is None
    with pytest.raises(RuntimeError):
        SongIdentifier.parse_lookup({"status": "error", "error": {"message": "invalid API key"}})


def _identifier(tmp_path, songs, **kw):
    sid = SongIdentifier(songs.append, cache_dir=str(tmp_path), log=lambda m: None, **kw)
    sid.cover = lambda song: None          # no internet in tests
    sid.language = lambda song: None
    return sid


def _wait(cond, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout and not cond():
        time.sleep(0.05)
    return cond()


def test_rds_song_is_published_once(tmp_path):
    songs = []
    sid = _identifier(tmp_path, songs, api_key=None, fpcalc="/nonexistent")
    assert "AcoustID off" in sid.status()
    sid.radiotext("Now playing: Coldplay - Yellow", 100.0)
    sid.radiotext("Now playing: Coldplay - Yellow", 105.0)     # same text: ignored
    sid.radiotext("Call the studio on 0800 1234", 106.0)       # not a song
    sid.radiotext("Radio 538 - Nederlands Hitradio", 107.0, "RADIO538")   # a slogan
    assert _wait(lambda: songs)
    time.sleep(0.3)
    assert [(s["artist"], s["title"], s["source"], s["time"]) for s in songs] == \
        [("Coldplay", "Yellow", "RDS", 100.0)]
    sid.stop.set()


def test_song_source_acoustid_ignores_rds(tmp_path):
    songs = []
    sid = _identifier(tmp_path, songs, api_key="KEY", fpcalc="/nonexistent", sources=("acoustid",))
    assert "RDS" not in sid.status()
    sid.radiotext("Now playing: Coldplay - Yellow", 100.0)
    time.sleep(0.3)
    assert songs == [] and not sid.jobs
    sid.stop.set()


FPCALC = os.environ.get("FPCALC") or (songid.find_fpcalc() if os.name == "nt" else shutil.which("fpcalc"))


@pytest.mark.skipif(not FPCALC, reason="fpcalc not available")
def test_acoustid_fingerprint_at_song_start(tmp_path, monkeypatch):
    sent = []

    def fake_get(url, timeout=10, data=None):
        form = urllib.parse.parse_qs(data.decode())
        sent.append(form)
        return json.dumps(BATCH_ANSWER).encode(), "application/json"

    monkeypatch.setattr(songid, "_get", fake_get)
    songs = []
    sid = _identifier(tmp_path, songs, api_key="KEY", fpcalc=FPCALC, interval=0)
    assert sid.status() == "RDS, AcoustID"
    # 20 s of talk, then 50 s of "music" (a chord with a beat), in 0.1 s blocks
    rng = np.random.default_rng(1)
    t0 = 1000.0
    for i in range(700):
        t = np.arange(4800) / 48000 + i * 0.1
        if i < 200:
            a = 0.1 * rng.standard_normal(4800)
        else:
            a = sum(np.sin(2 * np.pi * f * t) for f in (220, 277, 330)) * (0.2 + 0.2 * ((i // 5) % 2))
        sid.add_audio(a.astype(np.float32), t0 + i * 0.1)
    sid.regions([("speech", t0, t0 + 20), ("music", t0 + 20, t0 + 35)])
    time.sleep(0.2)
    assert not sent                                  # only 15 s of song: too early
    sid.regions([("music", t0 + 35, t0 + 70)])
    assert _wait(lambda: songs)
    form = sent[0]
    assert form["client"] == ["KEY"] and form["batch"] == ["1"] and form["maxdurationdiff"] == ["30"]
    assert [form[f"duration.{i}"][0] for i in range(4)] == [str(d) for d in songid.DURATION_GUESSES]
    assert form["fingerprint.0"] == form["fingerprint.3"] and len(form["fingerprint.0"][0]) > 50
    assert songs[0]["title"] == "Yellow" and songs[0]["time"] == t0 + 20
    # found: no second try at 100 s of the same song
    sid.regions([("music", t0 + 70, t0 + 125)])
    time.sleep(0.3)
    assert len(sent) == 1
    sid.stop.set()


def test_web_song_event_and_cover(tmp_path):
    (tmp_path / "abc.jpg").write_bytes(b"\xff\xd8" + b"x" * 2000)
    web = LiveWeb(0, cover_dir=str(tmp_path))
    try:
        web.set_song({"artist": "Coldplay", "title": "Yellow", "album": "Parachutes", "cover": "abc.jpg",
                      "source": "AcoustID", "score": 0.91, "time": 123.0})
        state = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{web.port}/state").read())
        assert state["song"]["title"] == "Yellow" and state["song"]["cover"] == "/cover/abc.jpg"
        r = urllib.request.urlopen(f"http://127.0.0.1:{web.port}/cover/abc.jpg")
        assert r.headers["Content-Type"] == "image/jpeg" and len(r.read()) == 2002
        for bad in ("/cover/..%2Fsecret.jpg", "/cover/nope.jpg", "/cover/abc.txt"):
            with pytest.raises(urllib.error.HTTPError):
                urllib.request.urlopen(f"http://127.0.0.1:{web.port}{bad}")
    finally:
        web.close()


def _http_400(body):
    import io
    return urllib.error.HTTPError("https://api.acoustid.org/v2/lookup", 400, "Bad Request", {},
                                  io.BytesIO(json.dumps(body).encode()))


def test_bad_key_is_explained(tmp_path, monkeypatch):
    def fake_get(url, timeout=10, data=None):
        raise _http_400({"status": "error", "error": {"code": 4, "message": "invalid API key"}})
    monkeypatch.setattr(songid, "_get", fake_get)
    sid = _identifier(tmp_path, [], api_key="WRONG", fpcalc="/nonexistent")
    with pytest.raises(songid.AcoustIdError, match="invalid API key.*application"):
        sid.lookup("AQAAfake")
    sid.stop.set()


def test_batch_refused_falls_back_to_single_lookups(tmp_path, monkeypatch):
    sent = []

    def fake_get(url, timeout=10, data=None):
        form = urllib.parse.parse_qs(data.decode())
        sent.append(form)
        if "batch" in form:
            raise _http_400({"status": "error", "error": {"code": 2, "message": "invalid parameter"}})
        if form["duration"] == ["210"]:
            return json.dumps({"status": "ok", "results": BATCH_ANSWER["fingerprints"][1]["results"]}).encode(), ""
        return json.dumps({"status": "ok", "results": []}).encode(), ""
    monkeypatch.setattr(songid, "_get", fake_get)
    sid = _identifier(tmp_path, [], api_key="KEY", fpcalc="/nonexistent")
    sid.last_request = -10
    song = sid.lookup("AQAAfake")
    assert song["title"] == "Yellow"
    assert [f.get("duration", ["batch"])[0] for f in sent] == ["batch", "150", "210"]
    assert sid.simple
    sid.stop.set()


def test_rolling_lookups_every_interval(tmp_path):
    calls = []
    sid = _identifier(tmp_path, [], api_key="KEY", fpcalc=__file__, interval=15)
    sid._fingerprint = lambda t0, seconds: calls.append((t0, round(seconds)))
    t = 1000.0
    sid.regions([("speech", t, t + 10), ("music", t + 10, t + 25)])     # 15 s of song: too early
    sid.regions([("music", t + 25, t + 40)])                             # 30 s: first lookup
    assert _wait(lambda: len(calls) == 1)
    sid.regions([("music", t + 40, t + 55)])                             # +15 s: again
    assert _wait(lambda: len(calls) == 2)
    for i in range(8):                                                  # a long song: rolling 120 s window
        sid.regions([("music", t + 55 + 15 * i, t + 70 + 15 * i)])
        assert _wait(lambda: len(calls) == 3 + i)
    assert calls[0] == (t + 10, 30) and calls[1] == (t + 10, 45)
    assert calls[-1] == (t + 175 - 120, 120)
    sid.stop.set()


def test_rds_kind_splits_pieces_at_radiotext_changes():
    from visualradio.songid import RdsKind
    k = RdsKind()
    classifier = [("speech", 0.0, 15.0)]
    assert k.regions(100.0, 15.0, classifier) == classifier          # nothing from RDS yet
    k.update("Goedemorgen met Jan op Radio 2", 90.0, "RADIO 2")
    k.update(None, 95.0)                                              # no text received: no change
    k.update("Coldplay - Yellow", 106.0, "RADIO 2")
    k.update("Coldplay - Yellow", 111.0, "RADIO 2")
    assert k.regions(100.0, 15.0, classifier) == [("speech", 0.0, 6.0), ("music", 6.0, 15.0)]
    assert k.regions(120.0, 15.0, [("speech", 0, 15.0)]) == [("music", 0.0, 15.0)]
    k.update("Radio 2 - de beste muziek", 140.0, "RADIO 2")           # slogan: talk again
    assert k.regions(135.0, 15.0, []) == [("music", 0.0, 5.0), ("speech", 5.0, 15.0)]


def test_pick_language_work_first_then_releases():
    pick = SongIdentifier.pick_language
    rec = {"relations": [{"type": "performance", "work": {"id": "w", "language": "nld", "languages": ["nld"]}}],
           "releases": [{"text-representation": {"language": "eng"}}] * 3}
    assert pick(rec) == "nl"                                        # the work = the lyrics
    assert pick({"releases": [{"text-representation": {"language": "deu"}}, {"text-representation": {"language": "eng"}},
                              {"text-representation": {"language": "deu"}}, {}]}) == "de"
    assert pick({"relations": [{"work": {"languages": ["zxx"]}}]}) == ""      # instrumental
    assert pick({"relations": [{"work": {"languages": ["eng", "spa"]}}]}) is None
    assert pick({}) is None


def test_song_language_lookup_and_use(tmp_path):
    songs, calls = [], []
    sid = SongIdentifier(songs.append, cache_dir=str(tmp_path), log=lambda m: None, api_key=None, fpcalc="/nonexistent")
    sid.cover = lambda song: None

    def fake_mb(path, **params):
        calls.append((path, params.get("query"), params.get("inc")))
        if path == "recording":
            return {"recordings": [{"id": "low", "score": 60}, {"id": "rec-1", "score": 100}][::-1]}
        assert path == "recording/rec-1" and params["inc"] == "work-rels+releases"
        return {"relations": [{"work": {"languages": ["nld"]}}]}
    sid._mb = fake_mb
    sid.radiotext("Doe Maar - Smoorverliefd", 100.0)
    assert _wait(lambda: songs)
    assert songs[0]["language"] == "nl" and "mbid" not in songs[0]
    assert calls[0][1] == 'recording:"Smoorverliefd" AND artist:"Doe Maar"'
    assert sid.language_at(95.0) == "nl" and sid.language_at(80.0) is None
    # talk after the song: back to the normal language
    sid.regions([("music", 100.0, 200.0)])
    sid.regions([("speech", 205.0, 220.0)])
    assert sid.language_at(230.0) is None
    # cached: no second lookup
    n = len(calls)
    assert sid.language({"artist": "Doe Maar", "title": "Smoorverliefd"}) == "nl" and len(calls) == n
    sid.stop.set()


def test_whisper_gets_the_song_language():
    from types import SimpleNamespace as NS
    from visualradio.transcribe import Transcriber
    seen = []

    class W:
        def transcribe(self, audio, **kw):
            seen.append(kw["language"])
            return iter([NS(text=" la la", no_speech_prob=0.1, words=[])]), NS(language=kw["language"] or "en")

    class C:
        def regions(self, a):
            return [("speech", 0.0, 2.0), ("music", 2.0, 4.0)]

    tr = Transcriber(backend=W(), classifier=C(), language="nl")
    tr.music_language = lambda t: "en" if t >= 102.0 else None
    out = tr.transcribe_all("x", 100.0, np.zeros(48000 * 4, np.float32) + 0.01)
    tr.close()
    assert seen == ["nl", "en"] and [t.language for t in out] == ["nl", "en"]


def test_station_change_drops_song_being_looked_up(tmp_path):
    import threading
    import time
    from visualradio.songid import SongIdentifier
    songs, gate = [], threading.Event()
    sid = SongIdentifier(songs.append, cache_dir=str(tmp_path), log=lambda m: None, sources=("rds",))
    sid.language = lambda song: None

    def slow_cover(song):              # the cover lookup takes a while ...
        gate.wait(2)
        return None
    sid.cover = slow_cover
    sid.radiotext("Bruno Mars - The Lazy Song", 100.0)
    time.sleep(0.3)
    sid.reset()                        # ... and meanwhile the station changes
    gate.set()
    time.sleep(0.3)
    assert songs == []
    sid.radiotext("Adele - Hello", 200.0)               # the new station's song still comes through
    t0 = time.time() + 2
    while time.time() < t0 and not songs:
        time.sleep(0.05)
    assert [s["title"] for s in songs] == ["Hello"]
    sid.stop.set()
