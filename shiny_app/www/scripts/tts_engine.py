"""
TTS inference engine. Loaded by R Shiny via reticulate.
Uses Hugging Face transformers VITS (facebook/mms-tts-kin) for Kinyarwanda speech.
Load model once; synthesize(text, speaker_id?) returns (wav_bytes, latency_ms).

Audio quality notes:
- noise_scale / noise_scale_duration are lowered at load time for cleaner, less static output.
- Post-processing uses only a gentle DC-removal high-pass + peak normalization.
  The heavy bandpass (80–7kHz order-4) was removed because it caused ringing artifacts
  that made the output sound like an FM radio with no signal.
"""
import os
import time
import io
from pathlib import Path

_model = None

DEFAULT_MODEL = "facebook/mms-tts-kin"

_WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "tf_model.h5",
    "flax_model.msgpack",
)

# ── VITS noise parameters ─────────────────────────────────────────────────────
# Lower values  → cleaner, less varied speech (less stochastic noise injected).
# Higher values → more natural variation but more audible noise.
# Default in pretrained model: noise_scale=0.667, noise_scale_duration=0.8
# At 0.333/0.4 the output is noticeably cleaner without sounding robotic.
_NOISE_SCALE          = float(os.environ.get("TTS_NOISE_SCALE",          "0.333"))
_NOISE_SCALE_DURATION = float(os.environ.get("TTS_NOISE_SCALE_DURATION", "0.4"))


def _has_weights(model_dir: Path) -> bool:
    return any((model_dir / f).exists() for f in _WEIGHT_FILES)


def get_model_dir() -> Path:
    root = os.environ.get("TTS_PROJECT_ROOT")
    if not root:
        # __file__ is not defined when loaded via reticulate::source_python();
        # TTS_PROJECT_ROOT is always set by app.R before sourcing, so this
        # fallback only matters when running the script directly from a terminal.
        try:
            root = Path(__file__).resolve().parents[1]
        except NameError:
            root = Path.cwd()
    return Path(root) / "artifacts" / "final_model"


def load_model():
    global _model
    if _model is not None:
        return

    model_dir  = get_model_dir()
    model_path = os.environ.get("TTS_MODEL_PATH", DEFAULT_MODEL)
    use_pre    = os.environ.get("TTS_USE_PRETRAINED", "").lower() in ("1", "true", "yes")

    if not use_pre and model_dir.exists() and (model_dir / "config.json").exists() and _has_weights(model_dir):
        model_path = str(model_dir)
        print(f"[TTS] Loading fine-tuned model: {model_path}", flush=True)
    else:
        if not use_pre and model_dir.exists() and not _has_weights(model_dir):
            print(f"[TTS] No weights in {model_dir} — falling back to pretrained {DEFAULT_MODEL}.", flush=True)
        else:
            print(f"[TTS] Loading pretrained model: {model_path}", flush=True)

    from transformers import pipeline
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = pipeline("text-to-speech", model=model_path, device=0 if device == "cuda" else -1)

    # Cache sampling rate from model config
    _model._sampling_rate = getattr(
        _model.model.config, "sampling_rate",
        getattr(_model.model.config, "sample_rate", 16000),
    )

    # ── Lower noise scales for cleaner output ──────────────────────────────────
    # The default noise_scale=0.667 injects too much stochastic noise on CPU,
    # producing the "FM radio static" effect. Reducing it gives much cleaner speech.
    try:
        _model.model.config.noise_scale          = _NOISE_SCALE
        _model.model.config.noise_scale_duration = _NOISE_SCALE_DURATION
        print(
            f"[TTS] noise_scale={_NOISE_SCALE}, noise_scale_duration={_NOISE_SCALE_DURATION}",
            flush=True,
        )
    except Exception as e:
        print(f"[TTS] Could not set noise_scale on config: {e}", flush=True)

    print(f"[TTS] Ready — device={device}, sr={_model._sampling_rate} Hz", flush=True)


# ── Post-processing ────────────────────────────────────────────────────────────

def _remove_dc(audio, sr, cutoff_hz=60, order=1):
    """
    Very gentle high-pass to remove DC offset and sub-bass rumble only.
    Order 1 at 60 Hz — barely touches the speech band, no ringing.
    NOTE: The previous order-4 bandpass (80–7kHz) was removed because it
    caused heavy ringing artifacts on VITS output (the 'FM static' sound).
    """
    import numpy as np
    from scipy.signal import butter, filtfilt
    nyq  = sr / 2.0
    high = max(cutoff_hz / nyq, 1e-4)
    b, a = butter(order, min(high, 0.99), btype="high")
    return filtfilt(b, a, audio.astype(np.float64)).astype(np.float32)


def _to_wav_bytes(audio, sampling_rate: int, target_sr: int = 16000) -> bytes:
    import numpy as np
    import soundfile as sf

    # Handle torch tensors
    if hasattr(audio, "cpu"):
        audio = audio.cpu().numpy()
    audio = np.asarray(audio, dtype=np.float64).squeeze()
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

    # Resample only if necessary
    if sampling_rate != target_sr:
        from scipy.signal import resample_poly
        audio = resample_poly(audio, target_sr, sampling_rate).astype(np.float32)
        sampling_rate = target_sr

    # Remove DC offset / sub-bass rumble (very gentle — no ringing)
    audio = _remove_dc(audio, sampling_rate, cutoff_hz=60, order=1)
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

    # Peak-normalize to 95 % of full scale
    peak = np.abs(audio).max()
    if peak > 1e-6:
        audio = (audio / peak) * 0.95

    buf = io.BytesIO()
    sf.write(buf, audio, sampling_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ── Public API ─────────────────────────────────────────────────────────────────

def synthesize(text: str, speaker_id=None):
    if speaker_id is not None:
        try:
            speaker_id = int(speaker_id)
        except (TypeError, ValueError):
            speaker_id = None

    load_model()
    t0 = time.perf_counter()

    # Pass lower noise scales explicitly (supported in transformers ≥4.37)
    try:
        output = _model(
            text,
            forward_params={
                "noise_scale":          _NOISE_SCALE,
                "noise_scale_duration": _NOISE_SCALE_DURATION,
            },
        )
    except TypeError:
        # Older transformers versions don't support forward_params — fall back
        output = _model(text)

    latency_ms = (time.perf_counter() - t0) * 1000

    # Unpack pipeline output
    if isinstance(output, (list, tuple)):
        output = output[0] if output else {}
    if isinstance(output, dict):
        audio = output.get("audio", output.get("output"))
        sr    = output.get("sampling_rate", output.get("sample_rate"))
    else:
        audio = output
        sr    = None

    # Always use model config rate so WAV header matches waveform speed exactly
    sr = getattr(_model, "_sampling_rate", 16000)

    if audio is None:
        import numpy as np
        audio = np.zeros(int(sr * 2), dtype=np.float32)

    wav_bytes = _to_wav_bytes(audio, sr)
    return wav_bytes, latency_ms
