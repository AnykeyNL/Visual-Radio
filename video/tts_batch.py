# python tts_batch.py voiceover.json OUTDIR   (needs kokoro-v1.0.onnx + voices-v1.0.bin next to it)
# pip install kokoro-onnx soundfile ; models: https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0
import sys, json, os, soundfile as sf
from kokoro_onnx import Kokoro
here = os.path.dirname(os.path.abspath(__file__))
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
k = Kokoro(os.path.join(here, "kokoro-v1.0.onnx"), os.path.join(here, "voices-v1.0.bin"))
for grp, lines in json.load(open(src, encoding="utf-8")).items():
    for key, text in lines:
        s, sr = k.create(text, voice="af_heart", speed=1.05, lang="en-us")
        sf.write(f"{out}/{grp}_{key}.wav", s, sr); print(grp, key, round(len(s)/sr, 2))
