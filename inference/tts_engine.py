"""
TTS inference engine. Loaded by R Plumber via reticulate.
Uses Hugging Face transformers VITS (facebook/mms-tts-kin) for real Kinyarwanda speech.
Load model once; synthesize(text, speaker_id?) returns (wav_bytes, latency_ms).
"""
import os
import time
import io
from pathlib import Path

_model = None

# Pretrained Kinyarwanda MMS-TTS model (or path to fine-tuned)
DEFAULT_MODEL = "facebook/mms-tts-kin"

def get_model_dir():
    root = os.environ.get("TTS_PROJECT_ROOT")
    if not root:
        root = Path(__file__).resolve().parents[1]
    return Path(root) / "artifacts" / "final_model"

def load_model():
    global _model
    if _model is not None:
        return
    model_dir = get_model_dir()
    model_path = os.environ.get("TTS_MODEL_PATH", DEFAULT_MODEL)
    # Use pretrained if env set (e.g. when fine-tuned model outputs silence), else use local if present
    use_pretrained = os.environ.get("TTS_USE_PRETRAINED", "").lower() in ("1", "true", "yes")
    if not use_pretrained and model_dir.exists() and (model_dir / "config.json").exists():
        model_path = str(model_dir)
    from transformers import pipeline
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = pipeline(
        task="text-to-speech",
        model=model_path,
        device=0 if device == "cuda" else -1,
    )
    _model._sampling_rate = getattr(
        _model.model.config, "sampling_rate",
        getattr(_model.model.config, "sample_rate", 16000)
    )

def _bandpass_speech(audio, sr, low_hz=80, high_hz=7000, order=4):
    """Keep only the speech band: removes rumble and hiss so voice sounds clearer and closer."""
    import numpy as np
    from scipy.signal import butter, filtfilt
    nyq = sr / 2.0
    low = max(low_hz / nyq, 0.001)
    high = min(high_hz / nyq, 0.999)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, audio.astype(np.float64)).astype(np.float32)


def _to_wav_bytes(audio, sampling_rate, target_sr=16000):
    """Convert audio to WAV: bandpass (reduce noise), then peak-normalize for clear, close playback."""
    import numpy as np
    import soundfile as sf
    # Handle torch tensors (pipeline may return these)
    if hasattr(audio, "cpu"):
        audio = audio.cpu().numpy()
    audio = np.asarray(audio, dtype=np.float64).squeeze()
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    # Resample only if needed
    if sampling_rate != target_sr:
        from scipy.signal import resample_poly
        audio = resample_poly(audio, target_sr, sampling_rate).astype(np.float32)
        sampling_rate = target_sr
    # Bandpass: keep 80 Hz–7 kHz (speech band), removes rumble and high hiss so voice sounds closer
    audio = _bandpass_speech(audio, sampling_rate, low_hz=80, high_hz=7000)
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    # Peak-normalize for clear, loud playback
    peak = np.abs(audio).max()
    if peak > 1e-6:
        audio = (audio / peak) * 0.95
    buf = io.BytesIO()
    sf.write(buf, audio, sampling_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()

def synthesize(text, speaker_id=None):
    if speaker_id is not None:
        try:
            speaker_id = int(speaker_id)
        except (TypeError, ValueError):
            speaker_id = None
    load_model()
    t0 = time.perf_counter()
    # MMS-TTS VITS is single-speaker; speaker_id ignored for pretrained
    output = _model(text)
    latency_ms = (time.perf_counter() - t0) * 1000
    if isinstance(output, (list, tuple)):
        output = output[0] if output else {}
    if isinstance(output, dict):
        audio = output.get("audio", output.get("output"))
        sr = output.get("sampling_rate", output.get("sample_rate"))
    else:
        audio = output
        sr = None
    # Always use model config rate so WAV header matches actual waveform (avoids "wrong speed" / radio sound)
    if hasattr(_model, "_sampling_rate"):
        sr = _model._sampling_rate
    if sr is None:
        sr = 16000
    if audio is None:
        import numpy as np
        sr = getattr(_model, "_sampling_rate", 16000)
        audio = np.zeros(int(sr * 2), dtype=np.float32)  # 2 sec silence fallback
    wav_bytes = _to_wav_bytes(audio, sr)
    return wav_bytes, latency_ms
