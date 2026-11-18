# Visual Radio for macOS (planned)

Not built yet. The Windows version in [`../windows`](../windows) is the
reference; this is what changes for a Mac:

| Step | Windows | Mac (plan) |
|---|---|---|
| Airspy | SoapySDR + SoapyAirspy from conda-forge | the same conda packages exist for macOS (arm64) |
| DAB+ | `welle-cli.exe` built for Windows | welle-cli built for macOS (Homebrew or from source, with the same PCM patch) |
| Whisper | faster-whisper on CUDA | Apple Silicon: whisper.cpp / MLX Whisper on the GPU (faster-whisper on Metal is not available) |
| Demucs, pyannote | PyTorch with CUDA | PyTorch with Metal (`mps`) |
| GPU meters | nvidia-smi / NVML | macOS power metrics, or left out |
| Song fingerprints | `fpcalc.exe` | `fpcalc` from Homebrew (`brew install chromaprint`) |

The web page, the FM and RDS decoding, the talk / music detection and the rest of
the Python code are platform independent and can be shared.
