"""Import this FIRST (before numpy, scipy or SoapySDR) in scripts that use Whisper.

Windows quirks with faster-whisper in a conda environment:
- CTranslate2 must be loaded before numpy/scipy/SoapySDR, or an older
  vcomp140.dll they pull in makes it fail with "procedure could not be found".
- CTranslate2 brings Intel's OpenMP (libiomp5md.dll) and conda-forge's
  numpy/scipy bring LLVM's (libomp.dll). Both in one process is fine here
  (Whisper runs on the GPU), but the Intel one aborts unless told it's allowed.
"""

import faulthandler
import os
import sys

# A crash inside a driver (SoapySDR, CUDA ...) would otherwise close the program
# without a word: print where it happened.
if sys.stderr is not None:
    faulthandler.enable()

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
try:
    import ctranslate2  # noqa: F401
except Exception:
    pass   # reported properly later, when the model is loaded
