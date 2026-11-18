import pytest

import visual_radio as stt
from visualradio.config import parse_args


def _write(tmp_path, text):
    p = tmp_path / "config.ini"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_keys_and_options_from_file(tmp_path, monkeypatch):
    monkeypatch.delenv("ACOUSTID_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    path = _write(tmp_path, "[keys]\nacoustid_key = abc123 ; my key\nhf_token =\n\n"
                            "[options]\nweb = yes\ndelay = 45\nweb-control = on\nsong_source = rds\n")
    args, used, names = parse_args(stt.build_parser(), ["--config", path])
    assert args.acoustid_key == "abc123" and args.hf_token is None
    assert args.web is True and args.web_control is True and args.delay == 45.0 and args.song_source == "rds"
    assert used == path and names == ["acoustid_key", "delay", "song_source", "web", "web_control"]
    # the command line wins
    args, _, _ = parse_args(stt.build_parser(), ["--config", path, "--delay", "30", "--acoustid-key", "cli"])
    assert args.delay == 30.0 and args.acoustid_key == "cli"


def test_environment_beats_file(tmp_path, monkeypatch):
    monkeypatch.setenv("ACOUSTID_KEY", "from-env")
    path = _write(tmp_path, "[keys]\nacoustid_key = from-file\n")
    args, _, names = parse_args(stt.build_parser(), ["--config", path])
    assert args.acoustid_key is None and names == []      # songid.py reads ACOUSTID_KEY itself


def test_no_file_is_fine(tmp_path, monkeypatch):
    import visualradio.config as cfg
    monkeypatch.setattr(cfg, "DEFAULT_FILE", str(tmp_path / "none.ini"))
    args, used, names = parse_args(stt.build_parser(), [])
    assert used is None and names == [] and args.web is False


@pytest.mark.parametrize("text", ["[keys]\nacoustid = x\n", "[stuff]\nweb = yes\n",
                                  "[options]\nweb = maybe\n", "[options]\ndelay = soon\n",
                                  "[options]\ncontent = podcasts\n"])
def test_mistakes_are_reported(tmp_path, text):
    path = _write(tmp_path, text)
    with pytest.raises(SystemExit):
        parse_args(stt.build_parser(), ["--config", path])


def test_example_file_is_valid(tmp_path, monkeypatch):
    import os
    import re
    monkeypatch.delenv("ACOUSTID_KEY", raising=False)
    src = open(os.path.join(os.path.dirname(stt.__file__), "config.example.ini"), encoding="utf-8").read()
    path = _write(tmp_path, re.sub(r"^;(\w+ =)", r"\1", src, flags=re.M))     # every option switched on
    args, _, names = parse_args(stt.build_parser(), ["--config", path])
    assert len(names) >= 14 and args.speakers is True and args.style_from == "rds"


def test_last_station_is_saved_without_touching_the_rest(tmp_path):
    from visualradio.config import read_state, save_state
    path = _write(tmp_path, "; my settings\n[keys]\nacoustid_key = abc ; secret\n\n[options]\n;delay = 45\nweb = yes\n")
    before = open(path, encoding="utf-8").read()
    save_state(path, last_station="102.7")
    after = open(path, encoding="utf-8").read()
    assert after.startswith(before.rstrip("\n")) and "[state]" in after
    assert read_state(path) == {"last_station": "102.7"}
    save_state(path, last_station="94.3")
    assert read_state(path) == {"last_station": "94.3"} and after.count("[state]") == 1
    # [state] in the middle, other sections after it stay where they are
    path2 = _write(tmp_path, "[state]\nlast_station = 88.3\n\n[options]\nweb = yes\n")
    save_state(path2, last_station="100.7", other="x")
    text = open(path2, encoding="utf-8").read()
    assert text == "[state]\nlast_station = 100.7\nother = x\n\n[options]\nweb = yes\n"
    # the [state] section doesn't upset reading the options
    args, _, names = parse_args(stt.build_parser(), ["--config", path2])
    assert args.web is True and args.config_path == path2
    # no file yet: one is made
    path3 = str(tmp_path / "new.ini")
    save_state(path3, last_station="96.8")
    assert read_state(path3) == {"last_station": "96.8"}


def test_save_transcript_switch(tmp_path):
    p = stt.build_parser()
    ini = tmp_path / "c.ini"
    ini.write_text("[options]\nsave_transcript = no\n")
    args, _, names = parse_args(p, ["--config", str(ini)])
    assert args.save_transcript is False and "save_transcript" in names
    ini.write_text("[options]\nsave_transcript = yes\n")
    assert parse_args(p, ["--config", str(ini)])[0].save_transcript is True
    assert parse_args(p, ["--config", str(ini), "--no-save-transcript"])[0].save_transcript is False
    ini.write_text("[keys]\n")
    assert parse_args(p, ["--config", str(ini)])[0].save_transcript is True
