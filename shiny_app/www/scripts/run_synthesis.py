#!/usr/bin/env python3
"""
Simple inference script: takes a text string and outputs a WAV file.
Prints latency (ms) for each call.
Usage: python run_synthesis.py "Your text here" [output.wav]
  If output path is omitted, writes to synthesis_output.wav in current directory.
Set TTS_PROJECT_ROOT to the project root (parent of inference/) so artifacts/final_model is found.
"""
import os
import sys
from pathlib import Path

# Ensure project root is set and inference module is importable from project root
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if not os.environ.get("TTS_PROJECT_ROOT"):
    os.environ["TTS_PROJECT_ROOT"] = str(_project_root)
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from tts_engine import synthesize


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_synthesis.py \"<text>\" [output.wav]", file=sys.stderr)
        sys.exit(1)
    text = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "synthesis_output.wav"

    wav_bytes, latency_ms = synthesize(text)
    with open(out_path, "wb") as f:
        f.write(wav_bytes)
    print(f"Latency: {latency_ms:.0f} ms")
    print(f"WAV written: {out_path}")


if __name__ == "__main__":
    main()
