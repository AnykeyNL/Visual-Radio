# Visual Radio for Windows

Everything in this folder is the Windows version of Visual Radio: the program
(`visual_radio.py`), its Python package (`visualradio\`), the tools it uses
(`tools\`) and the tests. How the pieces fit together is explained in the
[README one folder up](../README.md); how to install everything is in
[INSTALL.md](INSTALL.md).

Contents: [Quick start](#quick-start) ·
[Listening and transcribing](#listening-and-transcribing) ·
[Settings file](#settings-file-configini) · [Web page](#live-web-page---web) ·
[DAB+](#dab-digital-radio) · [Delay](#delayed-playback---delay) ·
[Speakers](#who-is-speaking---speakers) · [Isolation](#taking-the-music-out---isolate) ·
[Voice / music only](#only-the-voice-or-only-the-music-in-the-browser) ·
[Songs](#which-song-is-playing-rds-and-acoustid) · [Tests](#tests) · [Files](#files)

## Quick start

After [INSTALL.md](INSTALL.md), open the **Miniforge Prompt** and:

```
cd C:\projects\visual-radio\windows
conda activate visualradio
python visual_radio.py --simulate --web      # no Airspy needed: a fake band, to try it out
python visual_radio.py --web --delay 30      # the real thing: open http://localhost:8080
```

The first real start scans the FM band (about a minute) and shows a numbered
list; type a number to listen. The web page shows the words as they are spoken,
also on phones and tablets in your network (the address is printed at start).

A typical `config.ini` (copy `config.example.ini`) so a plain
`python visual_radio.py` does it all:

```ini
[keys]
acoustid_key = YOURKEY
hf_token = hf_xxxxxxxx

[options]
web = yes
web_control = yes
delay = 30
speakers = yes
isolate = talk
language = nl
```

## Listening and transcribing

`visual_radio.py` scans the FM band, shows a numbered list of stations
(with RDS names), and after you type a number it plays that station and
writes out what is said with Whisper on the GPU.

```
python visual_radio.py                  # scan, choose a number, transcribe
python visual_radio.py --freq 100.7     # skip the scan
python visual_radio.py --language nl    # faster and better if you know the language
python visual_radio.py --model turbo    # faster model, less GPU memory
python visual_radio.py --no-audio       # don't play, only transcribe
```

Text appears in pieces of about 15 seconds (`--chunk`), each cut at a quiet
moment so words are not split, and is saved to `fm_transcripts.txt`
(`--transcript FILE`). Don't want a file? `--no-save-transcript`, or
`save_transcript = no` in `config.ini`. Whisper skips music without vocals; song lyrics do get
transcribed. Audio is mono with 50 us de-emphasis (Europe).

Speech, music or both (`--content`):

```
python visual_radio.py --content speech   # only talk: presenters, news, ads
python visual_radio.py --content music    # only song lyrics
python visual_radio.py                    # both (default), lines marked [speech] / [music]
```

Each piece is first split into speech and music parts by
[YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet),
Google's AudioSet sound classifier (Apache 2.0). It runs in plain NumPy
(`visualradio/content.py`, weights in `models/yamnet.npz`, ~7 MB), so nothing extra
has to be installed, and takes about 0.3 s per 15 s piece. A presenter talking
over a song intro usually counts as speech; singing counts as music. Lyrics are
transcribed without Whisper's voice-activity filter (it tends to drop sung
vocals), so expect them to be rougher than speech.

Talk over background music or jingles can still be taken for a song. If your
station sends "Artist - Title" in its RDS radio text, let RDS decide instead:

```
python visual_radio.py --web --style-from rds
```

While the radio text names a song, everything is treated as music (lyrics
style, Whisper set up for singing); otherwise as talk. Until the first radio
text arrives, the sound decides. It is only as good as the station's RDS: some
keep the last song in the radio text while the presenter talks, others send
it a few seconds late.

The band scan is saved to `stations.json` next to the script, so the next
start shows the list straight away. `--rescan` scans again (and saves the new
list), `--stations FILE` keeps a list elsewhere (handy for a second antenna
or location).

```
python visual_radio.py --rescan              # scan again, save, choose
python visual_radio.py --freq 100.7 --rescan # tune at once, refresh the list too
```

### Settings file (`config.ini`)

Keys and the options you always use can go in `config.ini` next to the script,
so you don't have to type them every time:

```
copy config.example.ini config.ini
notepad config.ini
```

```ini
[keys]
acoustid_key = YOURKEY
hf_token = hf_xxxxxxxx

[options]
web = yes
web_control = yes
delay = 45
speakers = yes
```

Then `python visual_radio.py` is enough. `[options]` takes any
command-line option without the `--` (`web_port` or `web-port`, `yes`/`no` for
switches); `config.example.ini` lists the common ones. At start the script says
which settings it took from the file (never the key itself).

- The command line still wins (`--delay 30` overrides `delay = 45`), then the
  environment variables `ACOUSTID_KEY` / `HF_TOKEN`, then the file.
- `config.ini` is in `.gitignore`, so your keys are not committed.
- `--config other.ini` uses another file, e.g. one per antenna or room.
- A typo (unknown setting, `web = maybe`) stops the script with a message
  saying which line is wrong.
- `language = auto` (or no `language` line) lets Whisper detect the language of
  every piece; `language = nl` writes all talk in Dutch, translating other
  languages on the fly (rough, Whisper's own side effect).

**Last station.** The script writes the station you listen to into a
`[state]` section at the end of `config.ini` (every time you change station,
also from the web page), and the next start tunes to it straight away instead
of showing the list. The rest of the file, your comments included, is left as
it is. `--choose` shows the list anyway; `--freq` still picks a station
directly.

### Live web page (`--web`)

```
python visual_radio.py --web                 # page on port 8080
python visual_radio.py --web --web-port 8090
```

The script prints two addresses: `http://localhost:8080` for this PC and
`http://<your PC's IP>:8080` for phones, tablets and other computers on your
network. The first time, Windows asks whether Python may use the network:
allow it for private networks (otherwise other machines can't reach it).

The page needs no internet. It shows the frequency, the RDS station name and
"now playing" radio text (decoded live while listening), a live spectrum of the
audio, and the transcript as it comes in, word by word:

- Text appears sentence by sentence (lyrics line by line) at the moment it is
  heard. The word being spoken is highlighted and "fills in" from left to right
  while it is said; the words still to come in that sentence are shown faintly.
- The newest sentence is large; every older one is a step smaller, dimmer and
  softer, so the page reads like text receding into the distance.
- **Speech** reads like captions: clean sans-serif text, left-aligned, cool
  teal/blue, with a glass highlight that glides from word to word.
- **Songs** look like karaoke lyrics: big centred italic serif with a warm
  amber-to-pink colour wipe and a glow under the word being sung.
- A status panel at the top shows what is on air now (Talking / Lyrics), the
  language with its flag (from Whisper's language detection) and when the
  current line was said.
- The background shows it too: talk gets calm cool rings and a soft voice
  waveform; songs get warm stage-light beams, rising bokeh and floating notes.
  Both pulse with the audio and cross-fade when the programme switches.
- Older lines fade out; scroll back up to read them ("Back to live" returns).
  Press F (or the button) for fullscreen: nice on a TV or a spare tablet.

### Changing station from the page (`--web-control`)

```
python visual_radio.py --web --web-control
python visual_radio.py --web --web-control --delay 45
```

With `--web-control` the station name at the top of the page becomes a button
(and there is a **Stations** chip). It opens the station picker:

- An **FM | DAB+** switch at the top (see "DAB+ digital radio").
- A tuning dial from 87.5 to 108 MHz with a pin for every saved station.
  Move over the dial and the needle snaps to the nearest station; click to tune.
- Below it a card per station: RDS name, frequency, signal strength and the
  last radio text. The one on air has a live equaliser; arrow keys move
  through the cards, Enter tunes, Esc closes. On a phone it slides up as a
  sheet from the bottom.
- **Rescan band** sweeps the band from the page: a light beam runs over the dial,
  pins appear as stations are found and the names fill in while RDS is read.
  The new list is saved to `stations.json` and the receiver goes back to the
  station it was on.

After you pick a station the receiver switches at once, but with `--delay` you
still hear the old one for a while. The page says so ("you'll hear it in 32 s"),
shows a **Next** pill with a countdown, and switches the header, the colours and
a small "Now on ..." divider in the transcript at the moment the new station
is actually heard.

**Switching detection.** The status panel shows what decides talk or music,
e.g. `TALKING via SOUND`. With `--web-control` that badge is a button: it opens
a small panel with switches that work at once, without a restart:

- *Talk or music*: **Sound** (the classifier listens, like the default) or
  **RDS** (music while the radio text names a song, like `--style-from rds`).
- *Spoken language*: **Auto** (Whisper detects it for every piece) or a fixed
  language such as NL or EN (like `--language`), marked **Auto translate**: with
  a fixed language Whisper writes everything in that language, so an English
  interview on a Dutch station comes out in Dutch. That's Whisper's own side
  effect, rougher than a real translation (names and idioms can go wrong).
  Recognised songs keep using their own language either way.
- *Which song*: **RDS** radio text, **Sound** (AcoustID fingerprint) or
  **Both** (like `--song-source`). The fingerprint choices need `--acoustid-key`.
- *Isolate the voice*: **Off**, **Talk**, **Songs** or **All** (like
  `--isolate`, see "Taking the music out"). Needs `pip install demucs`.

The command-line options set where it starts; the footer always shows the
current choice ("Detect sound · song: RDS").

Without `--web-control` the page is read-only: nobody can change the station or the detection
(the server refuses it too, not just the page). With it, **anyone who can open
the page on your network can change the station**, so only use it on a network
you trust.

### DAB+ digital radio

```
python visual_radio.py --band dab --web --web-control     # first time: scans all DAB+ channels
python visual_radio.py --dab-station "NPO Radio 1" --web  # straight to a station
python visual_radio.py --dab-scan                          # scan the DAB+ channels again
```

The same Airspy also receives DAB+ (Band III, 174-240 MHz). Everything works
the same as on FM: Whisper, speaker colours, voice isolation, song recognition,
the web page. What RDS does on FM, DAB+ does with its own data:

- the **DLS** text (DAB's radio text, often "Artist - Title") fills the *Now*
  line, names the song (marked **DAB+**) and can decide talk or music (the
  *RDS* switch is called *DLS* while you listen to DAB+);
- the **slideshow** picture appears next to the station name, crossfades when
  the station sends a new one, and tints the page while no album cover does. When
  a song has no MusicBrainz cover, the slide (often the album art) is used.

DAB+ is decoded by **welle-cli** (from welle.io, GPL-2), a small program that
runs in the background. The Windows build in `tools\welle-cli.exe` was made for
this project: it talks to the Airspy itself, only listens on this PC
(127.0.0.1) and hands the decoded audio over as raw PCM (an extra `/pcm/`
address; the normal welle-cli only gives MP3). Python and welle-cli can't use
the Airspy at the same time, so on a switch between FM and DAB+ the script lets
go of it and welle-cli takes it (and the other way round); that takes about
two seconds. On FM the Airspy runs in a small helper process: some SoapySDR
Airspy drivers crash while they close the device, and ending the helper lets go
of it safely. `python tools\airspy_handover_test.py` checks the hand-over on
your PC (stop the script first). Switching between stations on the same channel (an *ensemble*,
e.g. all NPO stations on 12C) is instant.

**Station list.** The first DAB+ start scans the 38 channels 5A-13F (about two
minutes) and saves what it finds in `stations_dab.json`. Afterwards it starts on
the last station (like FM: `--choose` shows the list, `--dab-scan` scans again).
On the web page the station picker has an **FM | DAB+** switch: the dial
becomes Band III with a pin per ensemble, the cards are the stations grouped per
ensemble, and **Scan DAB+** scans from the page (on FM the Airspy goes to
welle-cli for the scan and comes back afterwards).

Options: `--band fm|dab` (default: the band you listened to last),
`--dab-station NAME` (name, part of it, or service id like 0x8201),
`--dab-channel 12C`, `--dab-scan`, `--dab-gain N` (0-21, default -1 =
automatic), `--welle PATH` (another welle-cli). All of them also work in
`config.ini` (`band = dab`, `dab_station = NPO Radio 1`).

welle-cli runs the Airspy at 4.096 MSPS. Current Airspy firmware (also on the
Mini) accepts that; if DAB+ stops with "the Airspy refused 4.096 MSPS", update
the firmware with `airspy_spiflash` from the Airspy tools (airspy.com).

DAB+ wants a vertical antenna of about 40 cm (a quarter wave for Band III);
an FM antenna works, but less well. `--simulate --band dab` runs a fake
DAB+ receiver (tones, changing DLS and slides) to try the page without hardware.

### Delayed playback (`--delay`)

Whisper needs a whole piece (~15 s) before it can write it out, so normally the
text comes after you hear it. With `--delay` the sound is held back instead:

```
python visual_radio.py --delay 45            # speakers play 45 s behind the radio
python visual_radio.py --web --delay 45      # + web page in step with the sound
```

- The PC speakers, the terminal lines, and the web page (text, spectrum and
  colours) all run `--delay` seconds behind the live broadcast.
- On the web page every word lights up at the moment it is spoken, using
  Whisper's word timestamps.
- The **Listen** button on the page plays the station in that browser, on the
  same delay, so phones and other PCs hear it in step with the text. Someone who
  joins later starts at once (the page keeps the last seconds buffered).
- Use at least `--chunk` + 10 s (25 s with the default 15 s pieces); 30-45 s
  is comfortable. Sound starts after the delay has passed.

### A look per song (`--web`)

During music the page picks a scene that fits the song:

| Scene | Background | Lyrics |
|---|---|---|
| Pop | light beams, bokeh, notes | warm italic serif |
| Electronic / dance | neon grid floor, sun, lasers | wide upright capitals, neon glow |
| Rock / metal | lightning on the beat, sparks, red heat; the text shakes | heavy capitals |
| Hip hop | pumping squares, halftone dots | fat type, sticker highlight |
| Classical / ambient | silk strands, a light from above | light upright serif, gold |
| Jazz / soul / blues | amber smoke, spotlight, glints | elegant italic |
| Folk / country / world | sun rays, fireflies | plain warm serif |

- The genre comes from the sound (the same classifier that tells talk from
  music), so it works for every song. Once the song is identified, the iTunes
  genre is used instead.
- The tempo (BPM) is measured from the audio. Highlights, beams and grids
  pulse on the beat, also with `--delay` and in the browser player.
- The album cover sets the colours; a dark cover gives a moodier room, a
  bright one a lighter one.
- The status panel shows the scene and tempo, e.g. `ELECTRONIC · 124 BPM`.
- **Music without words.** When Whisper hears music but no lyrics it often
  writes just "Muziek", "[Music]" or "♪". Those lines are not shown any more.
  Instead the transcript gets a live visual in the song's scene, on its beat
  and moving with the sound: a capsule equaliser with bubbles (pop), a neon
  oscilloscope (electronic), lightning and sparks (rock), a bouncing block
  meter (hip hop), drifting silk strands (classical), a horn line through smoke
  with notes (jazz) or plucked strings with fireflies (folk). Instrumental
  pieces in a row share one visual, with a running time underneath, and the
  status panel says "Instrumental". The terminal shows `♫ (instrumental)`;
  `fm_transcripts.txt` only keeps real words.

### Who is speaking? (`--speakers`)

```
python visual_radio.py --web --speakers --delay 45
```

When another voice takes over, the text changes colour. Voices are not named or
counted, there are no labels: you just see where the speaker changes. On the
page the sentence's line, the words and the highlight take the new colour, and
the background leans toward it. In the terminal the text itself is coloured, and
`fm_transcripts.txt` puts ` - ` where the voice changes.

Someone who comes back keeps the colour they had before (for the whole session,
until you change station), so a host and a guest stay two steady colours rather
than flipping at every piece. Only talk is analysed, song lyrics are not.

It runs entirely on this PC with
[pyannote.audio](https://github.com/pyannote/pyannote-audio) (MIT licence).
Whisper can't tell voices apart, so this is a separate model. It takes about
half a second per 15 s piece on the GPU and a few seconds on the CPU.

Setup (once):

1. Install PyTorch with CUDA, then pyannote, in the `visualradio` environment:

   ```
   conda activate visualradio
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
   pip install pyannote.audio
   ```

   (Install torch first: the plain `pip install pyannote.audio` would pull in a
   CPU-only torch on Windows.)

2. The model is free but you have to accept its terms on Hugging Face, once:
   - make a free account on https://huggingface.co,
   - open https://huggingface.co/pyannote/speaker-diarization-community-1 and
     accept the conditions,
   - make a token (type "Read") on https://huggingface.co/settings/tokens.

3. Run it once with the token so it can download the model:

   ```
   python visual_radio.py --web --speakers --hf-token hf_xxxxxxxx
   ```

   After that the model is in `%USERPROFILE%\.cache\huggingface` and no token
   or internet is needed. Or put the token in `config.ini` (`hf_token = ...`,
   see "Settings file" above) and leave out `--hf-token`.

If `pip install pyannote.audio` fails on `torchcodec`, use the previous
version instead: `pip install "pyannote.audio<4"`. That one uses the
`pyannote/speaker-diarization-3.1` model; accept the terms of both
https://huggingface.co/pyannote/speaker-diarization-3.1 and
https://huggingface.co/pyannote/segmentation-3.0. The script picks the right
model by itself.

Tuning: if one presenter keeps changing colour, lower
`--speaker-threshold` (e.g. 0.35). If two different people share a colour,
raise it (e.g. 0.55). Talk over loud music, phone-in callers and adverts are the
hardest (`--isolate talk` below helps). If the speaker model can't be loaded,
the script says why and goes on without colours.

### Taking the music out (`--isolate`)

```
python visual_radio.py --web --isolate all
```

Talk over a music bed, jingles and songs are hard for Whisper: the music gets in
the way of the words. With `--isolate` the script first splits every piece with
[Demucs](https://github.com/facebookresearch/demucs) (Meta's music source
separation, MIT licence) and Whisper hears **only the voice**. The speaker
colours use that clean voice too. What you hear, the talk/music detection and
song recognition still use the normal sound.

- `--isolate talk`: only for talk (presenters over music, adverts, jingles).
- `--isolate songs`: only for songs. Whisper hears the singer alone, which
  helps a lot for lyrics.
- `--isolate all`: both. `--isolate off` is the default.

With `--web-control` it is the fourth switch in the Detection panel
(*Isolate the voice*: Off, Talk, Songs, All), so you can compare live. In
`config.ini`: `isolate = all`.

Setup (once), in the `visualradio` environment, after PyTorch with CUDA (see "Who is
speaking?" step 1):

```
conda activate visualradio
pip install demucs
```

The first time it is used it downloads the model (`htdemucs`, about 80 MB) to
`%USERPROFILE%\.cache\torch`; after that it works offline. It runs on the GPU
next to Whisper and pyannote and takes well under a second per 15 s piece there
(about 1 GB of GPU memory). On a CPU it takes several seconds per piece, so the
text may fall behind. `--isolate-model htdemucs_ft` is a little cleaner but four
times slower. If Demucs isn't installed or fails, the script says so once and
Whisper gets the normal sound.

On clean talk without music it changes little, so `talk` mainly pays off on
stations that talk over music a lot.

### Only the voice, or only the music, in the browser

While you listen on the web page (**Listen**), a switch unfolds under it next
to the clock: **Original**, **Voice** and **Music**. Voice plays only what is
said or sung, with the music taken out. Music plays the station without the
voice, like a karaoke track. It is the same Demucs as above, so it needs
`pip install demucs` (the switch is greyed out without it), but `--isolate`
doesn't have to be on.

- Demucs only runs while at least one page listens to Voice or Music. Then it
  splits the sound in pieces of 2.5 s (with 4 s of the audio before each piece
  as context, and seamless joins). On the GPU that takes a fraction of a
  second per piece. On a CPU it may not keep up, and the page then plays the
  original until the split catches up.
- All three play in step with the text, and the switch crossfades between them,
  so you can flip back and forth while a song plays.
- With a playback delay (`--delay 20` or more) switching is seamless. When you
  pick Voice, the PC first splits the sound that is still waiting in the delay,
  so it starts within a second or two. Without a delay, Demucs needs about 5 s
  of head start. The sound then pauses once for those seconds, and the page
  tells you so.
- Every browser chooses for itself: a phone can play the music while the PC
  plays the original. The choice is remembered per browser.

### Engine room: CPU and GPU load

With `--web` the footer shows how hard the PC is working: small rings for CPU,
GPU and GPU memory and a two-minute trail, in the colour of what is playing. A
thin glow on the footer's top edge grows with the GPU load. Click the rings for
the **Engine room** card:

- CPU and GPU over the last two minutes, with this program's own CPU share and
  the GPU memory as dashed lines,
- GPU memory and RAM bars, GPU temperature and power,
- *Work per piece*: where the time of every ~15 s piece goes (talk/music
  detection, voice isolation, Whisper, speaker colours), and how many times
  faster than the radio it is. If Whisper can't keep up it says
  **Falling behind** and how many pieces are waiting.

Press `L` on the page to hide the meters (the browser remembers it), `L` again
to bring them back. `--no-load` (or `no_load = yes` in `config.ini`) turns them
off completely.

It works without extra packages: CPU and RAM come straight from Windows, the GPU
from `nvidia-smi` (comes with the NVIDIA driver). These two are a little more
precise and lighter, if you like:

```
pip install psutil nvidia-ml-py
```

The start-up line `Load meter: ...` says what it uses. Whatever can't be read
is simply left out.

### Which song is playing? (RDS and AcoustID)

While music plays, the program also finds out which song it is, prints
`♪ Now playing: Artist – Title`, writes it to the transcript file, and shows it
on the web page with the album cover on a spinning record. The cover also
colours the music background.

It tries two ways:

1. **RDS radio text.** Many stations send "Artist - Title" themselves. This
   needs no key and no internet (only the cover does).
2. **AcoustID.** For stations that don't, the sound itself is recognised:
   `fpcalc` (Chromaprint) makes a fingerprint and the free AcoustID service
   looks it up in MusicBrainz. **This needs a free API key:**
   1. Go to https://acoustid.org/new-application and sign in (with a
      MusicBrainz or Google account).
   2. Register an application (name e.g. "FM speech to text", version 1.0).
   3. Copy the **API key** it shows and use it like this:

   ```
   python visual_radio.py --web --acoustid-key YOURKEY
   ```

   or put it in `config.ini` once (`acoustid_key = YOURKEY`, see "Settings
   file" above); then no option is needed. `fpcalc.exe` is included in `tools\`.

Covers come from the Cover Art Archive (MusicBrainz), otherwise from the
iTunes search. Neither needs a key. Covers and results are kept in `cache\`,
so a song is looked up only once.

**Lyrics in the song's own language.** Once a song is recognised (by RDS or
AcoustID), its language is looked up in MusicBrainz: the language of the
song's lyrics, or else the language most of its releases list. Whisper then
transcribes that song's lyrics in that language, even when you start with
`--language nl` (that still applies to talk and to songs that aren't
recognised). After the song, when someone talks for a while, it goes back to
normal. The song card shows a small flag with the language, and the terminal
says e.g. `Now playing: Queen - Bohemian Rhapsody [RDS] · sung in English`.
No key is needed, and the result is kept in `cache\`. `--no-song-language`
turns it off.

Good to know:

- During music the program asks AcoustID every 15 s (`--song-interval 15`,
  in practice once per Whisper piece, see `--chunk`). It sends the music since
  the song began, up to the last 2 minutes. That is about 4 lookups a minute
  while music plays, far below AcoustID's limit of 3 per second. When the
  answer is a different song, the display changes.
- AcoustID works best from the **start** of a song. A song that was already
  playing when you tuned in, or one that runs straight into the next without
  talk in between, may be recognised late or not at all. RDS has no such
  limit. `--song-interval 0` asks only near the start of a song (after 40 s
  and 100 s), which uses far fewer lookups.
- AcoustID only knows songs that someone has submitted. Big hits are almost
  always there, local or very new songs sometimes not.
- Without a key, without `fpcalc` or without internet everything else keeps
  working; the start-up line says what is on (`Song recognition: RDS, AcoustID`).
- `--song-source acoustid` uses only AcoustID (RDS radio text is ignored for
  songs; the station name still comes from RDS). `--song-source rds` uses
  only RDS. `--no-songs` turns all of this off.

## Tests

```
cd windows
python -m pytest tests
```

The tests use simulators and stand-ins (a fake SoapySDR, a fake welle-cli, a
fake Whisper, a fake Demucs), so they need no Airspy, no GPU and no internet.
They cover the FM front end and RDS, the band scan, the station list and
config file, the Airspy helper process, DAB+ (channels, PCM stream, resampler,
tuning and scanning on a simulated welle-cli), the web page server (stations,
transcripts, audio and voice / music frames), song recognition, speaker
colours, voice isolation and the live voice / music split.

Hardware checks for your own PC (stop `visual_radio.py` first):

```
python tools\airspy_handover_test.py              # FM -> DAB+ -> FM with the real Airspy
python tools\airspy_handover_test.py --reception  # DAB+ signal, gains and welle-cli's messages
python tools\airspy_release_test.py               # how the Airspy driver behaves when closed
```

## Files

```
visual_radio.py           the program: receiver, pipeline, terminal, web page
stt_preload.py            loads Whisper's libraries first (a Windows DLL quirk) and turns on crash reports
config.example.ini        all settings, commented; copy to config.ini
environment.yml           the conda environment (Python, SoapySDR + Airspy driver, ...)
requirements.txt          the pip packages (Whisper, PyTorch, Demucs, pyannote, ...)
visualradio\
  sources.py              Airspy Mini (SoapySDR) and the FM simulator
  airspy_proc.py          the Airspy in a helper process, so it can be handed to welle-cli and back
  scanner.py              sweep the FM band, find stations
  rds.py                  FM demodulator and RDS decoder (station name, radio text)
  audio.py                FM -> mono 48 kHz audio
  stations.py             the station list: scan once, keep it in stations.json
  dab.py                  DAB+ through welle-cli: channels, scan, audio, DLS text, slideshow
  content.py              talk or music? YAMNet in plain NumPy (weights in models\yamnet.npz)
  vocals.py               voice isolation with Demucs (--isolate)
  stems.py                voice only / music only while listening on the web page
  transcribe.py           pieces of ~15 s -> faster-whisper, in a background thread
  speakers.py             who is speaking: pyannote.audio, a colour per voice
  songid.py               which song: RDS radio text, AcoustID, MusicBrainz language, covers
  style.py                a look per song: genre scene and tempo
  sysload.py              CPU / GPU load for the engine room
  config.py               config.ini and the last station
  web.py, web_page.html   the live web page (no internet needed)
models\yamnet.npz         YAMNet sound classifier weights (Apache 2.0, ~7 MB)
tools\
  welle-cli.exe           DAB+ decoder (welle.io, GPL), built with raw PCM output: see welle-cli.txt
  welle-cli-pcm.diff      the change to welle.io's source for that
  fpcalc.exe              Chromaprint audio fingerprints for AcoustID (LGPL): see fpcalc.txt
  airspy_*_test.py        hardware checks, see Tests
tests\                   python -m pytest tests
```

Files the program makes next to itself (not in git): `config.ini` (your keys and
the last station), `stations.json` and `stations_dab.json` (the station lists),
`fm_transcripts.txt` (the text), `cache\` (covers and song lookups).
