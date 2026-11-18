# Installing Visual Radio on Windows

About half an hour, most of it downloads. Every step is done once.

## What you need

- **Windows 10 or 11**, 64-bit.
- **An Airspy Mini** (Airspy R2 works too) with an antenna. FM is easy. DAB+
  (174-240 MHz) wants a vertical rod of about 32 cm; an FM dipole picks up
  little there.
- **An NVIDIA graphics card** with a recent driver, ideally 8 GB of memory or
  more. Whisper `large-v3`, Demucs and pyannote together use about 6-7 GB.
  Without a GPU everything still runs on the CPU, with a smaller Whisper model
  and slower.
- About **15 GB of disk** for the environment and the models.

## 1. Miniforge

Download `Miniforge3-Windows-x86_64.exe` from
https://github.com/conda-forge/miniforge/releases and install it with the
defaults. From now on, use the **Miniforge Prompt** from the Start menu.

(SoapySDR and its Airspy driver are not on pip, but conda-forge has both. That
is why this uses conda.)

## 2. The code

With git:

```
cd C:\projects
git clone https://github.com/AnykeyNL/Visual-Radio.git
cd visual-radio\windows
```

Or download the repository as a zip, unpack it to `C:\projects\visual-radio`,
and `cd C:\projects\visual-radio\windows`.

## 3. The environment

```
conda env create -f environment.yml
conda activate visualradio
```

That installs Python 3.11, NumPy, SciPy, SoapySDR with the Airspy module,
sounddevice (speakers), pytest, and with pip: faster-whisper with NVIDIA's
cuBLAS and cuDNN, psutil and nvidia-ml-py. Run `conda activate visualradio` in
every new prompt.

## 4. PyTorch with CUDA

Voice isolation (Demucs), the voice / music switch and the speaker colours run
on PyTorch. Install the CUDA build **first**, because a plain `pip install` of
the other packages would pull in PyTorch for the CPU only:

```
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available())"     # should print True
```

No NVIDIA card? Use `pip install torch torchaudio` instead (CPU).

## 5. The optional parts

All of these can be added later. The program says at start what is on and what
is missing.

| Feature | Install | Extra |
|---|---|---|
| Voice isolation, Voice / Music switch | `pip install demucs` | the model (80 MB) downloads on first use |
| Speaker colours | `pip install pyannote.audio` | a free Hugging Face token, see below |
| Song recognition by sound | (`tools\fpcalc.exe` is included) | a free AcoustID key, see below |
| DAB+ | (`tools\welle-cli.exe` is included) | an antenna for 174-240 MHz |

Or everything at once (after step 4): `pip install -r requirements.txt`.

**Hugging Face token (speaker colours).** Make a free account on
https://huggingface.co, accept the conditions on
https://huggingface.co/pyannote/speaker-diarization-community-1, and make a
token (type "Read") on https://huggingface.co/settings/tokens. Put it in
`config.ini` (`hf_token = hf_...`). It is needed once, to download the model.
If `pip install pyannote.audio` fails on `torchcodec`, use
`pip install "pyannote.audio<4"` and also accept
https://huggingface.co/pyannote/speaker-diarization-3.1 and
https://huggingface.co/pyannote/segmentation-3.0.

**AcoustID key (songs).** Sign in on https://acoustid.org/new-application,
register an application (any name), and put the API key in `config.ini`
(`acoustid_key = ...`). Stations that send "Artist - Title" in RDS or DAB+ need
no key.

## 6. The Airspy

Plug it in. Windows normally installs the WinUSB driver by itself. Check that
SoapySDR sees it:

```
SoapySDRUtil --find="driver=airspy"
```

You should see `driver = airspy` and a serial number. If not:

- try another USB port or cable (no unpowered hubs);
- if SDR# or SDR++ can't see it either, install the driver with **Zadig**
  (https://zadig.akeo.ie): *Options > List All Devices*, pick *AIRSPY*,
  choose **WinUSB**, *Install Driver*;
- close any other program that uses the Airspy.

## 7. Settings

```
copy config.example.ini config.ini
notepad config.ini
```

Put your keys under `[keys]` and the options you always want under `[options]`
(see the comments in the file, and [README.md](README.md#settings-file-configini)).
`config.ini` is in `.gitignore`, so your keys never end up in git.

## 8. Check it

```
python -m pytest tests                          # about a minute, no hardware needed
python visual_radio.py --simulate --web         # a fake band; open http://localhost:8080
python tools\airspy_handover_test.py            # the real Airspy: FM -> DAB+ -> FM
```

If Windows asks whether Python may use the network, allow it for **private
networks**: that is what lets phones and tablets open the page.

## 9. First real start

```
python visual_radio.py --web --delay 30
```

The first start downloads Whisper `large-v3` (about 3 GB) into
`%USERPROFILE%\.cache\huggingface`, scans the FM band and shows the stations.
After that it starts in seconds on the last station. DAB+: press **Stations**
on the page, switch to **DAB+** and **Scan DAB+**, or start with `--band dab`.

## Troubleshooting

- **"procedure could not be found" or "OMP: Error #15"** when Whisper loads:
  `stt_preload.py` handles both; make sure you start `visual_radio.py` itself
  and not an older copy.
- **Whisper falls back to the CPU** ("cublas64_12.dll not found" / cuDNN): run
  `pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"` again in the
  `visualradio` environment and update the NVIDIA driver.
- **"No Airspy Mini found"** or **"could not be opened"**: see step 6; also
  close SDR#, SDR++ or a second copy of Visual Radio.
- **DAB+ finds nothing**: `python tools\airspy_handover_test.py --reception`
  shows the signal-to-noise per channel and gain, and what welle-cli says.
  Around 0 dB it barely decodes: a better antenna (higher, vertical) helps most. "The Airspy refused
  4.096 MSPS" means old Airspy firmware: update it with `airspy_spiflash`
  (airspy.com).
- **A crash with "Windows fatal exception"**: the program prints where it
  happened (`stt_preload.py` turns that on). Some SoapySDR Airspy drivers crash
  when they close the device; that is why the Airspy runs in a helper process
  (`visualradio\airspy_proc.py`).
- **The Voice / Music buttons are grey**: Demucs isn't installed (step 5).

## Updating

```
cd C:\projects\visual-radio
git pull
cd windows
conda env update -f environment.yml
```
