#!/usr/bin/env python3
"""Visual Radio: listen to FM or DAB+ radio and see what is said, live.

Airspy Mini -> FM (or DAB+ via welle-cli) -> talk / music detection -> voice
isolation (Demucs, optional) -> Whisper -> terminal, transcript file and a live
web page. See ../README.md (how it works), INSTALL.md and README.md (all options).

1. Scans 87.5-108 MHz with the Airspy Mini and reads the RDS station names.
2. Shows a numbered list; you type the number of the station you want.
3. Plays that station on the PC speakers and transcribes it with Whisper
   (faster-whisper, on the NVIDIA GPU), in pieces of about 15 seconds.

Examples:
    python visual_radio.py                       # scan, choose, transcribe
    python visual_radio.py --freq 100.7          # skip the scan, go straight to 100.7 MHz
    python visual_radio.py --language nl         # faster and better if you know the language
    python visual_radio.py --model turbo         # faster model, less GPU memory
    python visual_radio.py --no-audio            # transcribe without playing
    python visual_radio.py --content speech      # only talk (presenters, news), skip songs
    python visual_radio.py --content music       # only song lyrics, skip talk
    python visual_radio.py --web                 # live web page, also for other PCs/phones
    python visual_radio.py --web --delay 45      # play 45 s late, so text and sound match
    python visual_radio.py --acoustid-key KEY    # also recognise songs from their sound
    python visual_radio.py --style-from rds      # music style while RDS names a song
    python visual_radio.py --web --web-control   # change station / rescan from the page
    python visual_radio.py --web --speakers      # a colour per speaker (pyannote, local)
    python visual_radio.py --simulate            # no hardware: fake band (tones, no speech)
    python visual_radio.py --band dab            # DAB+ instead of FM (welle-cli, see README)
    python visual_radio.py --dab-station "NPO Radio 1"   # a DAB+ station by name
"""

from __future__ import annotations

import stt_preload  # noqa: F401  (must come before numpy/scipy, see the file)

import argparse
import math
import os
import queue
import sys
import threading
import time
from collections import deque

import numpy as np

from visualradio.stations import decode_all_rds
from visualradio.audio import AUDIO_RATE, FmAudio
from visualradio.scanner import scan
from visualradio.airspy_proc import AirspyProcess
from visualradio.sources import DeviceError, SimulatedSource


def build_parser():
    p = argparse.ArgumentParser(description="Visual Radio: pick an FM or DAB+ station and see what is said (Whisper).")
    p.add_argument("--config", metavar="FILE",
                   help="settings file with API keys and default options (default: config.ini "
                        "next to this script, see config.example.ini)")
    p.add_argument("--simulate", action="store_true", help="fake FM band, no hardware needed")
    p.add_argument("--freq", type=float, metavar="MHZ", help="skip the scan and use this station")
    p.add_argument("--gain", type=float, default=18.0, help="receiver gain 0-45 dB (default 18)")
    p.add_argument("--threshold", type=float, default=10.0,
                   help="dB above the noise to count as a station (default 10)")
    p.add_argument("--no-rds", action="store_true", help="don't read station names (faster scan)")
    p.add_argument("--rds-time", type=float, default=3.0, help="max seconds of RDS per station (default 3)")
    p.add_argument("--volume", type=float, default=1.0, help="audio volume factor (default 1.0)")
    p.add_argument("--no-audio", action="store_true", help="don't play the station, only transcribe")
    p.add_argument("--delay", type=float, default=0.0, metavar="SECONDS",
                   help="play the station this many seconds late, so the text appears at the "
                        "moment you hear it (e.g. 45; needs more than --chunk + a few seconds)")
    p.add_argument("--duration", type=float, default=0, help="stop after this many seconds")
    p.add_argument("--serial", help="pick an Airspy by (part of) its serial number")
    p.add_argument("--web", action="store_true",
                   help="show the transcript live on a web page (reachable from your network)")
    p.add_argument("--web-port", type=int, default=8080, help="port for --web (default 8080)")
    p.add_argument("--web-control", action="store_true",
                   help="allow changing the station (and rescanning) from the web page")
    p.add_argument("--no-load", action="store_true",
                   help="don't show the CPU / GPU load on the web page")
    p.add_argument("--stations", metavar="FILE",
                   help="saved station list (default stations.json next to this script)")
    p.add_argument("--rescan", action="store_true", help="scan the band again instead of using the saved list")
    p.add_argument("--choose", action="store_true",
                   help="show the station list at start, instead of tuning to the last station")
    s = p.add_argument_group("song recognition (RDS radio text, AcoustID)")
    s.add_argument("--acoustid-key", metavar="KEY",
                   help="free AcoustID application key (https://acoustid.org/new-application); "
                        "or put it in config.ini / set ACOUSTID_KEY. Without it: RDS only")
    s.add_argument("--fpcalc", metavar="PATH", help="fpcalc program (default: tools\\fpcalc.exe or PATH)")
    s.add_argument("--song-source", choices=["both", "rds", "acoustid"], default="both",
                   help="how to find the song: RDS radio text, AcoustID fingerprint, or both (default)")
    s.add_argument("--song-interval", type=float, default=15.0, metavar="SECONDS",
                   help="during music, ask AcoustID every this many seconds (default 15; "
                        "0 = only at the start of a song)")
    s.add_argument("--no-songs", action="store_true", help="don't look up which song is playing")
    s.add_argument("--no-song-language", action="store_true",
                   help="don't use the recognised song's language (from MusicBrainz) for its lyrics")
    t = p.add_argument_group("speech to text")
    t.add_argument("--model", default="auto",
                   help="Whisper model: tiny, base, small, medium, large-v3, turbo "
                        "(default: large-v3 on GPU, small on CPU)")
    t.add_argument("--language", help="write all talk in this language, e.g. nl, de, en; talk in other languages is translated "
                        "(roughly, by Whisper itself). Default: auto = detect per piece, written as spoken")
    t.add_argument("--whisper-device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="run Whisper on the GPU or CPU (default: GPU if it works)")
    t.add_argument("--content", choices=["speech", "music", "both"], default="both",
                   help="what to transcribe: speech (talk, news), music (song lyrics) or "
                        "both (default); each line is marked [speech] or [music]")
    t.add_argument("--style-from", choices=["sound", "rds"], default="sound",
                   help="what decides talk or music: the sound (default), or RDS: music while the "
                        "station's radio text names a song ('Artist - Title'), otherwise talk")
    v = p.add_argument_group("speaker colours (pyannote.audio, runs locally)")
    v.add_argument("--speakers", action="store_true",
                   help="tell voices apart: every speaker gets their own colour on the web page "
                        "and in the terminal: the colour changes when the voice does (needs pyannote.audio, see README)")
    v.add_argument("--hf-token", metavar="TOKEN",
                   help="Hugging Face token, only needed for the first download of the speaker "
                        "model (or put it in config.ini / set HF_TOKEN)")
    v.add_argument("--speaker-model", metavar="NAME",
                   help="pyannote pipeline (default: speaker-diarization-community-1 with "
                        "pyannote.audio 4, speaker-diarization-3.1 with 3.x)")
    v.add_argument("--speaker-threshold", type=float, default=0.45, metavar="SIM",
                   help="how alike two pieces must sound to be the same voice, 0-1 "
                        "(default 0.45; higher: more separate voices)")
    i = p.add_argument_group("voice isolation (Demucs, runs locally)")
    i.add_argument("--isolate", choices=["off", "talk", "songs", "all"], default="off",
                   help="take the music out before Whisper, so it hears only the voice: for talk "
                        "(talk over a music bed), songs (lyrics), all, or off (default). "
                        "Needs: pip install demucs. With --web-control it can be switched on the page")
    i.add_argument("--isolate-model", default="htdemucs", metavar="NAME",
                   help="Demucs model (default htdemucs; htdemucs_ft is a bit cleaner but 4x slower)")
    d = p.add_argument_group("DAB+ digital radio (welle-cli, see README)")
    d.add_argument("--band", choices=["fm", "dab"],
                   help="start on FM or DAB+ (default: the band you listened to last)")
    d.add_argument("--dab-station", metavar="NAME",
                   help="DAB+ station by name (or part of it) or service id, from the saved DAB+ list")
    d.add_argument("--dab-channel", metavar="CH", help="DAB+ channel, 5A ... 13F (with --dab-station: "
                   "only look there; alone: the first station on that channel)")
    d.add_argument("--dab-scan", action="store_true", help="scan all DAB+ channels at start (and save the list)")
    d.add_argument("--welle", metavar="PATH", help="welle-cli program (default: tools\\welle-cli.exe or PATH)")
    d.add_argument("--dab-gain", type=int, default=-1, metavar="N",
                   help="Airspy gain step for DAB+, 0-21, or -1 = automatic (default)")
    d.add_argument("--dab-iq", metavar="FILE", help=argparse.SUPPRESS)      # testing: IQ recording, no Airspy
    t.add_argument("--chunk", type=float, default=15.0,
                   help="seconds of audio per transcription piece (default 15)")
    t.add_argument("--transcript", metavar="FILE", default="fm_transcripts.txt",
                   help="append transcripts to this file (default fm_transcripts.txt)")
    t.add_argument("--no-save-transcript", dest="save_transcript", action="store_false",
                   help="don't save the transcript to a file (config.ini: save_transcript = no)")
    return p


def scan_and_save(source, args, path):
    """Scan the band (with RDS names unless --no-rds), save it to `path`. -> data"""
    from visualradio.stations import save, scan_band

    def progress(phase, done, total, data):
        if phase == "sweep":
            sys.stdout.write(f"\r\x1b[KScanning {data:6.1f} MHz  ({done + 1}/{total})")
        elif phase == "found":
            sys.stdout.write(f"\r\x1b[KFound {total} station(s)" + (", reading their names (RDS)" if total and not args.no_rds else ""))
        elif phase == "rds":
            sys.stdout.write(f"\r\x1b[KReading names (RDS) {done}/{total}")
        sys.stdout.flush()

    print("Scanning 87.5-108 MHz ...", flush=True)
    data = scan_band(source, args.threshold, 0 if args.no_rds else args.rds_time, progress=progress)
    print()
    if data["stations"]:
        try:
            save(data, path)
            print(f"Saved the list to {path} (next time no scan is needed; --rescan scans again)")
        except OSError as exc:
            print(f"Could not save the station list ({exc})")
    return data


def choose_station(source, args, path):
    """Show the saved (or freshly scanned) list and ask which one.
    -> ((freq_hz, name) or None, station list data)"""
    from visualradio.stations import load
    data = None if args.rescan else load(path)
    if data and data["stations"]:
        when = time.strftime("%d %b %Y %H:%M", time.localtime(data.get("scanned", 0)))
        print(f"Stations from {path} (scanned {when}; --rescan to scan again)")
    else:
        data = scan_and_save(source, args, path)
    stations = data["stations"]
    if not stations:
        print("No stations found. Try a lower --threshold, more --gain, or a better antenna.")
        return None, data
    print(f"\n{len(stations)} station(s):\n")
    for i, s in enumerate(stations, 1):
        print(f"  {i:2d})  {s['freq']:6.1f} MHz   {s['snr']:4.0f} dB   {s.get('name') or ''}")
    while True:
        try:
            ans = input("\nStation number (or a frequency in MHz, Enter to quit): ").strip()
        except EOFError:
            return None, data
        if not ans:
            return None, data
        try:
            v = float(ans.replace(",", "."))
        except ValueError:
            print("Type a number from the list.")
            continue
        if v.is_integer() and 1 <= v <= len(stations):
            st = stations[int(v) - 1]
            return (st["freq"] * 1e6, st.get("name") or ""), data
        if 87.5 <= v <= 108.0:
            near = min(stations, key=lambda st: abs(st["freq"] - v))
            if abs(near["freq"] - v) < 0.1:
                return (near["freq"] * 1e6, near.get("name") or ""), data
            return (round(v * 10) / 10 * 1e6, ""), data
        print(f"Choose 1-{len(stations)}, or a frequency between 87.5 and 108.")


def dab_scan_console(welle, path):
    """Visit every DAB+ channel (about two minutes), print and save what is on air. -> data"""
    from visualradio import dab
    print("Scanning DAB+ channels 5A-13F (about two minutes) ...", flush=True)
    found, n = [], len(dab.SCAN_CHANNELS)
    for i, ch in enumerate(dab.SCAN_CHANNELS):
        sys.stdout.write(f"\r\x1b[K  channel {ch:>3}  ({i + 1}/{n}), {len(found)} ensemble(s) so far")
        sys.stdout.flush()
        ens = dab.scan_channel(welle, ch)
        if ens:
            found.append(ens)
            print(f"\r\x1b[K  {ch:>3}  {ens['label'] or '(no name)'}: "
                  + ", ".join(s["name"] for s in ens["services"]))
    print(f"\r\x1b[KFound {len(found)} DAB+ ensemble(s), "
          f"{sum(len(e['services']) for e in found)} station(s)")
    data = {"scanned": time.time(), "ensembles": found}
    if found:
        try:
            dab.save(data, path)
            print(f"Saved the list to {path} (--dab-scan scans again)")
        except OSError as exc:
            print(f"Could not save the DAB+ list ({exc})")
    return data


def choose_dab(data):
    """Numbered DAB+ list on the console. -> (channel, sid, name) or None"""
    flat = [(e["channel"], s["sid"], s["name"], e["label"]) for e in data["ensembles"] for s in e["services"]]
    if not flat:
        print("No DAB+ stations found. Check the antenna (DAB+ wants a vertical one of about 40 cm).")
        return None
    print(f"\n{len(flat)} DAB+ station(s):\n")
    for i, (ch, sid, name, ens) in enumerate(flat, 1):
        print(f"  {i:2d})  {name:<18} {ch:>4}  {ens}")
    while True:
        try:
            ans = input("\nStation number (Enter to quit): ").strip()
        except EOFError:
            return None
        if not ans:
            return None
        if ans.isdigit() and 1 <= int(ans) <= len(flat):
            return flat[int(ans) - 1][:3]
        print("Type a number from the list.")


class DelayLine:
    """FIFO for playback: put audio in, get the audio from `seconds` ago out
    (silence until the buffer has filled up)."""

    def __init__(self, seconds, rate=AUDIO_RATE):
        self.n = int(seconds * rate)
        self.buf = np.zeros(self.n, dtype=np.float32)
        self.pos = 0

    def __call__(self, audio):
        if self.n == 0:
            return audio
        out = np.empty_like(audio)
        i = 0
        while i < len(audio):
            k = min(len(audio) - i, self.n - self.pos)
            out[i:i + k] = self.buf[self.pos:self.pos + k]
            self.buf[self.pos:self.pos + k] = audio[i:i + k]
            self.pos = (self.pos + k) % self.n
            i += k
        return out


class LiveRds:
    """Decodes RDS from the last few seconds of FM baseband in a background thread,
    so the station name and "now playing" radio text stay up to date."""

    def __init__(self, on_update, seconds=6.0, every=5.0):
        self.on_update = on_update
        self.parts = deque(maxlen=int(seconds * 10))     # 0.1 s blocks
        self.every = every
        self.stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def add(self, mpx):
        self.parts.append(mpx)

    def _run(self):
        from visualradio.rds import decode_rds
        while not self.stop.wait(self.every):
            if len(self.parts) < 20:
                continue
            try:
                info = decode_rds(np.concatenate(list(self.parts)))
            except Exception:
                continue
            if info.groups:
                self.on_update(info)


def main(argv=None) -> int:
    from visualradio.config import parse_args
    args, cfg_path, cfg_names = parse_args(build_parser(), argv)
    if cfg_path:
        print(f"Settings from {cfg_path}: {', '.join(cfg_names)}")
    fs = 6e6

    print("Loading Whisper (the first time it downloads the model, this can take a while)...")
    try:
        from visualradio.content import SoundClassifier
        from visualradio.transcribe import ContinuousChunker, Transcriber
        transcriber = Transcriber(args.model, args.whisper_device, args.language, min_s=1.0,
                                  classifier=SoundClassifier(), content=args.content,
                                  word_times=args.web or args.speakers)
    except ImportError:
        print("Speech to text needs faster-whisper. Install it with: pip install faster-whisper")
        return 2
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 2
    print(f"Speech to text: {transcriber.description}, language {args.language or 'auto'}, "
          f"transcribing {'speech and music' if args.content == 'both' else args.content + ' only'}")
    speakers = None
    if args.speakers:
        print("Loading the speaker model (pyannote.audio)...")
        try:
            from visualradio.speakers import SpeakerTracker
            speakers = SpeakerTracker(args.speaker_model, args.hf_token, threshold=args.speaker_threshold)
            transcriber.speakers = speakers
            print(f"Speaker colours: {speakers.description}")
        except ImportError:
            print("Speaker colours need pyannote.audio (see README: 'Who is speaking?'). Going on without.")
        except Exception as exc:
            print(f"Speaker colours off: {exc}")

    # Voice isolation: ready whenever Demucs is installed, so the page can switch it on later
    from visualradio import vocals as vocal_mod
    isolator = None
    if vocal_mod.available():
        isolator = vocal_mod.VocalIsolator(args.isolate_model, log=lambda msg: print(f"\r\x1b[K{msg}"))
        transcriber.isolator = isolator
        if args.isolate != "off":
            print("Loading the voice isolation model (Demucs)...")
            try:
                isolator.load()
                transcriber.isolate = args.isolate
                print(f"Voice isolation: {isolator.description}, for "
                      + {"talk": "talk", "songs": "songs", "all": "talk and songs"}[args.isolate])
            except Exception as exc:
                print(f"Voice isolation off: {exc}")
    elif args.isolate != "off":
        print("Voice isolation needs Demucs: pip install demucs (see README). Going on without.")

    from visualradio import config as cfg
    from visualradio import dab as dabmod
    from visualradio.stations import DEFAULT_FILE, load

    def say(msg):
        print(f"\r\x1b[K{msg}")

    sim = "_simulated" if args.simulate else ""
    st_path = args.stations or (DEFAULT_FILE.replace(".json", "_simulated.json") if args.simulate else DEFAULT_FILE)
    st_data = load(st_path) or {"scanned": None, "stations": []}
    dab_path = dabmod.DEFAULT_FILE.replace(".json", sim + ".json")
    dab_data = dabmod.load(dab_path) or {"scanned": None, "ensembles": []}
    state = cfg.read_state(args.config_path)
    state_key = "last_station" + sim
    welle_exe = None if args.simulate else dabmod.find_welle(args.welle)
    dab_ready = bool(args.simulate or welle_exe or args.dab_iq)

    def open_fm():
        if args.simulate:
            return SimulatedSource(sample_rate=fs)
        # in a helper process: the Airspy can then be handed to welle-cli (DAB+) and back
        # without SoapySDR's driver crashing while it closes the device
        return AirspyProcess(sample_rate=fs, gain=args.gain, serial=args.serial)

    def open_welle(channel):
        """welle-cli on `channel` (the Airspy must be free)."""
        if args.simulate:
            return dabmod.SimWelle().start(channel)
        exe = welle_exe or (dabmod.find_welle(args.welle) if args.dab_iq else None)
        if not exe:
            raise dabmod.DabError(dabmod.WELLE_HELP)
        return dabmod.WelleCli(exe, channel, gain=args.dab_gain, iq_file=args.dab_iq, log=say)

    band = args.band or ("dab" if (args.dab_station or args.dab_channel or args.dab_scan) else
                         "fm" if args.freq else state.get("last_band" + sim, "fm"))
    if band not in ("fm", "dab"):
        band = "fm"
    if band == "dab" and not dab_ready:
        if args.band or args.dab_station or args.dab_channel or args.dab_scan:
            print(f"ERROR: {dabmod.WELLE_HELP}", file=sys.stderr)
            transcriber.close(0)
            return 2
        print(f"Last time you listened to DAB+, but: {dabmod.WELLE_HELP} Starting on FM.")
        band = "fm"

    # cur: what the receiver is on. FM: freq_hz; DAB+: channel and service id (sid)
    cur = {"band": band, "freq_hz": None, "channel": None, "sid": None, "name": "", "ensemble": ""}
    source = welle = radio = None
    if band == "fm":
        try:
            source = open_fm()
        except DeviceError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            transcriber.close(0)
            return 2
        print(f"Receiver: {source.describe()}")
        try:
            last = float(state.get(state_key) or 0)
        except ValueError:
            last = 0.0
        try:
            if args.freq:
                chosen = (args.freq * 1e6, "")
                if args.rescan:
                    st_data = scan_and_save(source, args, st_path)
            elif 87.5 <= last <= 108.0 and not args.choose:
                chosen = (last * 1e6, "")
                print(f"Last station: {last:.1f} MHz (--choose to pick another)")
                if args.rescan:
                    st_data = scan_and_save(source, args, st_path)
            else:
                chosen, st_data = choose_station(source, args, st_path)
        except KeyboardInterrupt:
            chosen = None
        if chosen is None:
            source.close()
            transcriber.close(0)
            return 0
        freq_hz, name = chosen
        if not name:   # a frequency typed in: maybe the saved list knows its name
            near = [x for x in st_data["stations"] if abs(x["freq"] * 1e6 - freq_hz) < 50e3]
            name = near[0].get("name", "") if near else ""
        cur.update(freq_hz=freq_hz, name=name)
    else:
        last_dab = (state.get("last_dab" + sim) or "").split()
        first = (args.dab_channel or (last_dab[0] if last_dab else "") or "5A").upper()
        if first not in dabmod.CHANNELS:
            print(f"ERROR: unknown DAB+ channel {first} (5A ... 13F)", file=sys.stderr)
            transcriber.close(0)
            return 2
        try:
            welle = open_welle(first)
        except dabmod.DabError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            transcriber.close(0)
            return 2
        print("Receiver: DAB+ " + ("(simulated)" if args.simulate else f"with {welle.exe}, Airspy, "
                                   + ("automatic gain" if args.dab_gain < 0 else f"gain step {args.dab_gain}")))
        chosen = None
        try:
            if args.dab_scan or not dab_data["ensembles"]:
                dab_data = dab_scan_console(welle, dab_path)
            if args.dab_station:
                chosen = dabmod.find_service(dab_data, args.dab_station, args.dab_channel and args.dab_channel.upper())
                if not chosen:
                    print(f"'{args.dab_station}' is not in the DAB+ list ({dab_path}). "
                          "Scan again with --dab-scan, or pick one:")
                    chosen = choose_dab(dab_data)
            elif args.dab_channel:
                ens = next((e for e in dab_data["ensembles"] if e["channel"] == first), None) \
                    or dabmod.scan_channel(welle, first)
                if ens and ens["services"]:
                    chosen = (first, ens["services"][0]["sid"], ens["services"][0]["name"])
                else:
                    print(f"Nothing found on channel {first}.")
            elif len(last_dab) == 2 and not args.choose:
                chosen = dabmod.find_service(dab_data, last_dab[1], last_dab[0]) or (last_dab[0], last_dab[1], "")
                print(f"Last station: {chosen[2] or chosen[1]} on {chosen[0]} (--choose to pick another)")
            else:
                chosen = choose_dab(dab_data)
        except KeyboardInterrupt:
            chosen = None
        if chosen is None:
            welle.close()
            transcriber.close(0)
            return 0
        cur.update(channel=chosen[0], sid=dabmod.norm_sid(chosen[1]), name=chosen[2])

    def make_label():
        if cur["band"] == "dab":
            return f"{cur['name'] or cur['sid']} (DAB+ {cur['channel']})"
        return f"{cur['freq_hz'] / 1e6:.1f} MHz" + (f" {cur['name']}" if cur["name"] else "")
    label = make_label()

    def remember_station():
        """Save the station in config.ini, so the next start tunes to it straight away."""
        vals = {"last_band" + sim: cur["band"]}
        if cur["band"] == "fm":
            vals[state_key] = f"{cur['freq_hz'] / 1e6:.1f}"
        else:
            vals["last_dab" + sim] = f"{cur['channel']} {cur['sid']}"
        try:
            cfg.save_state(args.config_path, **vals)
        except OSError as exc:
            say(f"Could not save the station in {args.config_path} ({exc})")
    remember_station()

    def dab_name(channel, sid):
        hit = dabmod.find_service(dab_data, sid, channel)
        return hit[2] if hit else ""

    def publish_station(retuned=False):
        if not web:
            return
        if cur["band"] == "fm":
            web.set_station(cur["freq_hz"], cur["name"] or "", retuned=retuned, band="fm")
        else:
            web.set_station(None, cur["name"] or "", retuned=retuned, band="dab", channel=cur["channel"],
                            sid=cur["sid"], mhz=dabmod.CHANNELS.get(cur["channel"]))

    web = None
    stems = None       # voice / music only, for listening on the page (Demucs)
    stem_hist = deque()            # (audio, captured) of the last `delay` seconds, for the stems
    stem_hist_fed = False
    if args.web:
        from visualradio.web import LiveWeb
        try:
            web = LiveWeb(args.web_port, delay=args.delay)
        except OSError as exc:
            print(f"ERROR: can't start the web page on port {args.web_port} ({exc}). "
                  "Try another one with --web-port 8090")
            if source:
                source.close()
            if welle:
                welle.close()
            transcriber.close(0)
            return 2
        publish_station()
        web.set_info(model=transcriber.description, language=args.language or "auto",
                     content=args.content, delay=args.delay, speakers=bool(speakers))
        if isolator:                  # "Voice" / "Music" next to Listen on the page
            from visualradio.stems import StemSplitter

            def stem_log(msg):
                say(msg)
                web.notice(msg, error="failed" in msg)
            stems = StemSplitter(lambda a: isolator.voice_and_music(a, AUDIO_RATE), web.stem_audio, log=stem_log,
                                 max_behind=args.delay + 30)
        web.set_info(stems={"ready": stems is not None, "lag": round(stems.lag(), 1) if stems else None})
        web.set_stations(st_data)
        web.set_dab_stations(dab_data, ready=dab_ready)
        if not args.no_load:          # CPU / GPU load and time per step, for the page
            from visualradio.sysload import SystemLoad
            load = SystemLoad(web.set_sys, extra=transcriber.timing_info).start()
            print(f"Load meter: {load.description()}")
    control = queue.Queue()        # station changes / rescans asked for on the web page
    busy = {"scan": False}
    if web and args.web_control:
        def web_control(cmd):
            if cmd["cmd"] == "detect":                       # small switch, no need to wait
                set_detection(cmd.get("talk"), cmd.get("songs"), cmd.get("language"), cmd.get("isolate"))
                return True, "OK"
            if busy["scan"]:
                return False, "A scan is running, try again in a moment"
            if cmd.get("band") == "dab" and not dab_ready:
                return False, dabmod.WELLE_HELP
            if cmd["cmd"] == "scan":
                busy["scan"] = True
            control.put(cmd)
            if cmd["cmd"] == "scan":
                return True, "Scanning"
            if cmd.get("band") == "dab":
                return True, "Tuning to " + (dab_name(cmd["channel"], cmd["sid"]) or cmd["sid"])
            return True, "Tuning to %.1f MHz" % cmd["freq"]
        web.control = web_control

    songs = None
    held_songs = []    # songs waiting until their audio is played (with --delay)
    if not args.no_songs:
        from visualradio.songid import SongIdentifier

        def on_song(song):
            held_songs.append(song)
            if web:
                web.set_song(song)
        songs = SongIdentifier(on_song, api_key=args.acoustid_key, fpcalc=args.fpcalc,
                               log=lambda msg: print(f"\r\x1b[K{msg}"),
                               sources=("rds", "acoustid") if args.song_source == "both" else (args.song_source,),
                               interval=args.song_interval)
        songs.text_source = "DAB+" if band == "dab" else "RDS"
        transcriber.on_regions = songs.regions
        if args.no_song_language:
            songs.find_language = False
        else:
            transcriber.music_language = songs.language_at   # lyrics in the song's own language
        print(f"Song recognition: {songs.status()}")

    # What decides talk or music ("sound" / "rds") and where songs come from; both can be
    # switched on the web page (--web-control). RDS (on DAB+: DLS) is followed all the time,
    # so a switch to "rds" takes effect at once.
    from visualradio.songid import RdsKind
    rds_kind = RdsKind() if (args.style_from == "rds" or (web and args.web_control)) else None
    detect = {"talk": args.style_from, "songs": "off" if songs is None else
              "rds" if args.song_source == "both" and not songs.acoustid_ready() else args.song_source}

    def region_override(start, duration, regions):
        if detect["talk"] == "rds" and rds_kind:
            return rds_kind.regions(start, duration, regions)
        return regions
    transcriber.region_override = region_override
    if args.style_from == "rds":
        print("Talk / music: from the radio text (RDS / DLS: music while it names a song)")

    def detect_info():
        return {"talk": detect["talk"], "songs": detect["songs"], "language": transcriber.language or "auto",
                "acoustid": bool(songs and songs.acoustid_ready()), "song_lookup": songs is not None,
                "isolate": transcriber.isolate, "isolate_ready": isolator is not None}

    def set_detection(talk=None, song_source=None, language=None, isolate=None):
        """From the web page: switch what decides talk/music, where songs come from,
        the spoken language ("auto": Whisper detects it per piece) and voice isolation."""
        if isolate in vocal_mod.MODES and isolator and isolate != transcriber.isolate:
            transcriber.isolate = isolate     # the model loads on the first piece that needs it
            say("Voice isolation now: " + {"off": "off", "talk": "for talk", "songs": "for songs",
                                           "all": "for talk and songs"}[isolate])
        if language and (None if language == "auto" else language) != transcriber.language:
            transcriber.language = None if language == "auto" else language
            say(f"Spoken language now: {transcriber.language or 'detected per piece (auto)'}")
            if web:
                web.set_info(language=transcriber.language or "auto")
        if talk in ("sound", "rds") and talk != detect["talk"]:
            detect["talk"] = talk
            say("Talk / music now decided by " + ("the radio text" if talk == "rds" else "the sound"))
        if songs and song_source in ("rds", "acoustid", "both") and song_source != detect["songs"]:
            detect["songs"] = song_source
            songs.set_sources(("rds", "acoustid") if song_source == "both" else (song_source,))
            say(f"Song recognition now: {songs.status()}")
        if web:
            web.set_detection(detect_info())
    if web:
        web.set_detection(detect_info())

    rds_gate = {"after": 0.0}      # ignore RDS decoded from before a retune

    def rds_update(info):
        if time.time() < rds_gate["after"] or cur["band"] != "fm":
            return
        if rds_kind:
            rds_kind.update(info.rt, time.time() - 3, (info.ps or "").strip())
        if web:
            web.set_station(name=(info.ps or "").strip() or None, rt=info.rt or None)
        if songs:
            songs.radiotext(info.rt, time.time() - 3, (info.ps or "").strip())
    rds_live = LiveRds(rds_update) if (web or songs or rds_kind) else None

    def dab_info(info):
        """From DabRadio (its own thread): station name, DLS text, slideshow, signal."""
        nonlocal label
        if cur["band"] != "dab" or info.get("sid") != cur["sid"]:
            return
        name, dls = info.get("name") or None, info.get("dls") or None
        if name and name != cur["name"]:             # not in the saved list (yet): its own label
            cur["name"] = name
            label = make_label()
            chunker.label = label
        if web:
            r = radio
            if info.get("slide") and r and r.slide:
                web.set_slide(r.slide[0], r.slide[1], info["slide"])
            web.set_station(name=name, rt=dls, ensemble=info.get("ensemble") or None, snr=info.get("snr"))
        if dls:
            if rds_kind:
                rds_kind.update(dls, time.time() - 2, name or "")
            if songs:
                songs.radiotext(dls, time.time() - 2, name or "")

    fm = FmAudio(fs)
    if cur["band"] == "fm":
        source.tune(fm.tune_hz(cur["freq_hz"]))
    else:
        radio = dabmod.DabRadio(welle, cur["channel"], cur["sid"], log=say, on_info=dab_info)
    chunker = ContinuousChunker(transcriber.submit, label, chunk_s=args.chunk)

    class _NoFile:                  # --no-save-transcript: writes go nowhere
        def write(self, s): pass
        def flush(self): pass
        def close(self): pass
    log = open(args.transcript, "a", encoding="utf-8") if args.save_transcript else _NoFile()

    stream = None
    if not args.no_audio:
        try:
            import sounddevice as sd
            stream = sd.OutputStream(samplerate=AUDIO_RATE, channels=1, dtype="float32")
            stream.start()
        except Exception as exc:  # missing module, no sound device, ...
            print(f"No audio output ({exc}). Install it with: conda install python-sounddevice\n"
                  "Continuing with transcription only.")

    os.system("")  # ANSI support in the Windows console
    print(f"\nListening to {label}. Text appears every ~{args.chunk:g} s "
          + (f"and is saved to {args.transcript}." if args.save_transcript else "(not saved to a file).")
          + " Ctrl+C to stop.")
    if args.delay:
        print(f"Playback is {args.delay:g} s behind the radio, so text shows up as you hear it. "
              f"Sound starts in {args.delay:g} s.")
        if args.delay < args.chunk + 5:
            print(f"Note: Whisper needs about {args.chunk:g} s (--chunk) plus a few seconds, so with "
                  f"--delay {args.delay:g} the text will still be a bit late. "
                  f"Try --delay {args.chunk + 10:g} or more.")
    if web:
        local, lan = web.urls()
        print(f"Live web page: {local}  (on this PC)\n"
              f"               {lan}  (phones, tablets, other PCs on your network)\n"
              "If Windows asks whether Python may use the network, allow it for private networks.")
    print()

    delay_line = DelayLine(args.delay)
    held = []          # transcripts waiting until their audio is played (with --delay)

    def voice_text(t, color=False):
        """The text. Where another voice takes over: a new colour in the console,
        " - " in the transcript file. Voices are not named or numbered."""
        if getattr(t, "instrumental", False):
            bpm = (t.style or {}).get("bpm")
            return "♫ (instrumental" + (f", {bpm:.0f} BPM" if bpm else "") + ")"
        if not getattr(t, "speakers", None) or not t.words:
            return t.text
        from visualradio.speakers import ANSI, runs
        parts = runs(t.words, t.speakers)
        if not color:
            return " - ".join(text for _, text in parts)
        return " ".join(text if v is None else f"\x1b[38;5;{ANSI[(v - 1) % len(ANSI)]}m{text}\x1b[0m"
                        for v, text in parts)

    def show():
        for t in transcriber.results():
            if not t.instrumental:                  # the transcript file keeps words only
                log.write(time.strftime("%Y-%m-%d %H:%M:%S  ", time.localtime(t.start)) + t.channel
                          + (f" [{t.kind}, {t.language}]" if t.kind else "") + f": {voice_text(t)}\n")
                log.flush()
            if web:
                web.transcript(t)
            held.append(t)
        held.sort(key=lambda t: t.start)
        while held and held[0].start + args.delay <= time.time():
            t = held.pop(0)
            clock = time.strftime("%H:%M:%S", time.localtime(t.start))
            tags = ", ".join(x for x in (t.kind, t.language) if x)
            lang = f" [{tags}]" if tags else ""
            line = f"{clock}  {t.channel}{lang}: {voice_text(t, color=True)}"
            print(f"\r\x1b[K{line}")
        while held_songs and (held_songs[0].get("time") or 0) + args.delay <= time.time():
            song = held_songs.pop(0)
            album = f"  ({song['album']})" if song.get("album") else ""
            from visualradio.songid import LANG_NAMES
            lang = song.get("language")
            sung = f"  · sung in {LANG_NAMES.get(lang, lang)}" if lang else ""
            print(f"\r\x1b[K♪ Now playing: {song['artist']} – {song['title']}{album}"
                  f"  [{song['source']}]{sung}")
            log.write(time.strftime("%Y-%m-%d %H:%M:%S  ", time.localtime(song.get("time") or time.time()))
                      + f"{label} [song, {song['source']}]: {song['artist']} - {song['title']}\n")
            log.flush()

    block = int(fs * 0.1)
    block -= block % fm.block_multiple
    t_start = time.time()

    def reopen_fm():
        """The Airspy back from welle-cli to FM (it may need a moment to be free)."""
        for attempt in range(4):
            try:
                return open_fm()
            except DeviceError:
                if attempt == 3:
                    raise
                time.sleep(1.0)

    def to_dab(channel):
        """FM -> DAB+: hand the Airspy to welle-cli. False (and back on FM) when that fails."""
        nonlocal source, welle
        source.close()
        source = None
        try:
            welle = open_welle(channel)
            return True
        except dabmod.DabError as exc:
            say(f"DAB+ is not available: {exc}")
            if web:
                web.notice(f"DAB+ is not available: {exc}", error=True)
            source = reopen_fm()
            source.tune(fm.tune_hz(cur["freq_hz"]))
            return False

    def to_fm():
        """DAB+ -> FM: stop welle-cli, take the Airspy back."""
        nonlocal source, welle, radio
        if radio:
            radio.close()
        if welle:
            welle.close()
        radio = welle = None
        source = reopen_fm()

    def switch(target, announce=True):
        """Move the receiver (and everything that follows one station) to target:
        {"band": "fm", "freq": Hz} or {"band": "dab", "channel": "12C", "sid": "0x8201"}."""
        nonlocal label, fm, chunker, radio
        chunker.flush()
        if target["band"] == "dab":
            if cur["band"] == "fm" and not to_dab(target["channel"]):
                chunker = ContinuousChunker(transcriber.submit, label, chunk_s=args.chunk)
                return
            if radio is None:
                radio = dabmod.DabRadio(welle, target["channel"], target["sid"], log=say, on_info=dab_info)
            else:
                radio.tune(target["channel"], target["sid"])
            cur.update(band="dab", channel=target["channel"], sid=target["sid"], ensemble="", freq_hz=None,
                       name=dab_name(target["channel"], target["sid"]))
        else:
            if cur["band"] == "dab":
                to_fm()
            near = [x for x in st_data["stations"] if abs(x["freq"] * 1e6 - target["freq"]) < 50e3]
            cur.update(band="fm", freq_hz=target["freq"], channel=None, sid=None,
                       name=(near[0].get("name") or "") if near else "")
            fm = FmAudio(fs)
            source.tune(fm.tune_hz(cur["freq_hz"]))
        label = make_label()
        chunker = ContinuousChunker(transcriber.submit, label, chunk_s=args.chunk)
        if rds_live:
            rds_live.parts.clear()
            rds_gate["after"] = time.time() + 1.5
        if songs:
            songs.reset()
            songs.text_source = "DAB+" if cur["band"] == "dab" else "RDS"
        if rds_kind:
            rds_kind.reset()
        if speakers:
            speakers.reset()
        if stems:
            stems.reset()
            web.stems_reset()
            stem_hist.clear()
        if announce:
            remember_station()
            publish_station(retuned=True)
            say(f"Tuned to {label}" + (f" (you hear it in {args.delay:g} s)" if args.delay else ""))
            log.write(time.strftime("%Y-%m-%d %H:%M:%S  ") + f"--- tuned to {label} ---\n")
            log.flush()

    def rescan():
        nonlocal st_data
        from visualradio.stations import save, scan_band
        say("Scanning the band (asked on the web page) ...")
        web.scan_progress("sweep", 0, 0, 87.5)

        def progress(phase, done, total, data):
            if phase == "sweep":
                web.scan_progress("sweep", done, total, data)
            else:
                web.set_stations({"scanned": None, "stations": data})
                web.scan_progress(phase, done, total, None)
        try:
            data = scan_band(source, args.threshold, 0 if args.no_rds else args.rds_time, progress=progress)
            if data["stations"]:
                st_data = data
                try:
                    save(data, st_path)
                except OSError as exc:
                    say(f"Could not save the station list ({exc})")
            say(f"Found {len(data['stations'])} station(s)")
        finally:
            web.set_stations(st_data)
            web.scan_progress("done", len(st_data["stations"]), len(st_data["stations"]))
            busy["scan"] = False
            switch({"band": "fm", "freq": cur["freq_hz"]}, announce=False)

    def rescan_dab():
        """All DAB+ channels, from the web page. On FM the Airspy goes to welle-cli for the
        scan and comes back afterwards; on DAB+ the station plays on when it is done."""
        nonlocal dab_data, source
        say("Scanning DAB+ (asked on the web page) ...")
        web.scan_progress("dab", 0, len(dabmod.SCAN_CHANNELS), dabmod.CHANNELS["5A"], band="dab", channel="5A")
        temp = None
        count = [-1]

        def progress(done, total, ch, found):
            if len(found) != count[0]:
                count[0] = len(found)
                web.set_dab_stations({"scanned": None, "ensembles": found}, ready=True)
            if ch:
                web.scan_progress("dab", done, total, dabmod.CHANNELS[ch], band="dab", channel=ch)
        try:
            if cur["band"] == "fm":
                source.close()
                source = None
                temp = open_welle("5A")
                data = {"scanned": time.time(), "ensembles": []}
                for i, ch in enumerate(dabmod.SCAN_CHANNELS):
                    progress(i, len(dabmod.SCAN_CHANNELS), ch, data["ensembles"])
                    ens = dabmod.scan_channel(temp, ch)
                    if ens:
                        data["ensembles"].append(ens)
                        say(f"DAB+ {ch}: {ens['label']}, {len(ens['services'])} station(s)")
            else:
                data = radio.scan(progress=progress)
            if data["ensembles"]:
                dab_data = data
                try:
                    dabmod.save(data, dab_path)
                except OSError as exc:
                    say(f"Could not save the DAB+ list ({exc})")
            n = sum(len(e["services"]) for e in data["ensembles"])
            say(f"DAB+: found {len(data['ensembles'])} ensemble(s), {n} station(s)")
        except dabmod.DabError as exc:
            say(f"DAB+ scan failed: {exc}")
            web.notice(f"DAB+ scan failed: {exc}", error=True)
        finally:
            if temp:
                temp.close()
            if cur["band"] == "fm" and source is None:
                source = reopen_fm()
                switch({"band": "fm", "freq": cur["freq_hz"]}, announce=False)
            web.set_dab_stations(dab_data, ready=dab_ready)
            n = sum(len(e["services"]) for e in dab_data["ensembles"])
            web.scan_progress("done", n, n, band="dab")
            busy["scan"] = False

    try:
        while not args.duration or time.time() - t_start < args.duration:
            while not control.empty():
                cmd = control.get()
                if cmd["cmd"] == "tune":
                    if cmd.get("band") == "dab":
                        if (cur["band"], cur["channel"], cur["sid"]) != ("dab", cmd["channel"], cmd["sid"]):
                            switch({"band": "dab", "channel": cmd["channel"], "sid": cmd["sid"]})
                    elif cur["band"] != "fm" or abs(cmd["freq"] * 1e6 - cur["freq_hz"]) > 1e3:
                        switch({"band": "fm", "freq": cmd["freq"] * 1e6})
                elif cmd["cmd"] == "scan":
                    rescan_dab() if cmd.get("band") == "dab" else (rescan() if cur["band"] == "fm" else None)
                    busy["scan"] = False
            t_block = time.time()
            if cur["band"] == "fm":
                audio = fm.process(source.read(block))
                captured = time.time() - len(audio) / AUDIO_RATE
                if rds_live:
                    rds_live.add(fm.mpx)
            else:
                audio, captured = radio.read(0.3)
                if not len(audio):                 # finding the ensemble / no signal
                    show()
                    sys.stdout.write(f"\r\x1b[K{label}  |  waiting for audio ...")
                    sys.stdout.flush()
                    continue
            chunker.add(audio)
            if web:
                web.audio(audio, transcriber.jobs.qsize(), captured)
            if stems:
                stem_hist.append((audio, captured))
                while stem_hist and stem_hist[0][1] < captured - args.delay - 5:
                    stem_hist.popleft()
                if web.stems_wanted():
                    if not stem_hist_fed:
                        # someone just picked Voice / Music: split the audio still to be played
                        # (the delay) first, so it starts within seconds, not after the delay
                        stem_hist_fed = True
                        for a, c in list(stem_hist)[:-1]:
                            if c >= captured - args.delay + 2.5:
                                stems.feed(a, c)
                    stems.feed(audio, captured)
                elif stem_hist_fed:
                    stem_hist_fed = False
                    stems.reset()
                    web.stems_reset()
            if songs:
                songs.add_audio(audio, captured)
            if stream is not None:
                stream.write((delay_line(audio) * args.volume).astype(np.float32).reshape(-1, 1))
            elif args.simulate and cur["band"] == "fm":
                time.sleep(max(0.0, 0.1 - (time.time() - t_block)))
            show()
            level = 20 * math.log10(float(np.sqrt(np.mean(audio ** 2))) + 1e-9)
            waiting = transcriber.jobs.qsize()
            sys.stdout.write(f"\r\x1b[K{label}  |  audio {level:5.1f} dB  |  "
                             f"{time.time() - t_start:5.0f} s"
                             + (f"  |  {waiting} piece(s) waiting for Whisper" if waiting > 1 else ""))
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    except DeviceError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
    finally:
        print()
        if source:
            source.close()
        if radio:
            radio.close()
        if welle:
            welle.close()
        if stream is not None:
            stream.stop()
            stream.close()
        chunker.flush()
        print("Finishing speech to text...")
        transcriber.close()
        args.delay = 0     # print whatever is still held back
        show()
        log.close()
        if rds_live:
            rds_live.stop.set()
        if songs:
            songs.stop.set()
        if stems:
            stems.close()
        if web:
            web.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
