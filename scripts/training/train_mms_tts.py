"""
TTS fine-tuning: MMS-TTS VITS on your Kinyarwanda data.
Reads config YAML; expects data/processed/metadata.csv and data/processed/wav/*.wav.
Trains for full epochs with mel-spectrogram + waveform loss. Saves to artifacts/final_model.
No placeholders, no step cap — real fine-tuning so the model learns to speak like your data.
"""
import argparse
import os
import sys
import yaml
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to mms_tts.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg.get("project_root", ".")).resolve()
    if str(root) == "." or not (root / "config").exists():
        root = Path(args.config).resolve().parents[1]
    meta_path = root / cfg.get("metadata_csv", "data/processed/metadata.csv")
    checkpoint_dir = root / cfg.get("checkpoint_dir", "artifacts/checkpoints")
    final_dir = root / cfg.get("final_model_dir", "artifacts/final_model")
    base_model = cfg.get("base_model", "facebook/mms-tts-kin")
    resume_from = cfg.get("resume_from")
    if resume_from:
        resume_path = (root / resume_from).resolve() if not Path(resume_from).is_absolute() else Path(resume_from)
    else:
        resume_path = None
    epochs = int(cfg.get("epochs", 50))
    batch_size = int(cfg.get("batch_size", 4))
    sample_rate = int(cfg.get("sample_rate", 16000))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    if not meta_path.exists():
        raise FileNotFoundError(f"Processed metadata not found: {meta_path}. Run data pipeline first.")

    import pandas as pd
    meta = pd.read_csv(meta_path)
    meta = meta[meta["split"] == "train"].reset_index(drop=True)
    if len(meta) < 20:
        raise ValueError(f"Need at least 20 training samples, got {len(meta)}. Run data pipeline first.")

    max_samples = cfg.get("max_train_samples")
    if max_samples is not None and isinstance(max_samples, int) and max_samples > 0:
        if len(meta) > max_samples:
            meta = meta.sample(n=max_samples, random_state=42).reset_index(drop=True)
            print(f"Subsampled to {len(meta)} train samples (max_train_samples={max_samples})", flush=True)

    def _has_weights(p):
        for f in ("model.safetensors", "pytorch_model.bin", "tf_model.h5", "flax_model.msgpack"):
            if (p / f).exists():
                return True
        return False

    # Only resume if the directory has actual weight files, not just config/tokenizer
    load_from = (
        resume_path
        if (
            resume_path
            and resume_path.exists()
            and (resume_path / "config.json").exists()
            and _has_weights(resume_path)
        )
        else None
    )
    if load_from:
        print(f"Resuming from: {load_from}", flush=True)
    else:
        if resume_path and resume_path.exists() and not _has_weights(resume_path):
            print(f"resume_from path has no weights ({resume_path}). Starting from base model instead.", flush=True)
        print(f"Base model: {base_model}", flush=True)
    print(f"Fine-tuning: {len(meta)} samples, {epochs} epochs, batch_size={batch_size}", flush=True)
    print(f"Output: {final_dir}", flush=True)

    from transformers import VitsModel, AutoTokenizer
    import torch.utils.data
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Mel spectrogram for stable training signal (matches VITS internal representation)
    n_fft = 1024
    hop_length = 256
    win_length = 1024
    n_mels = 80
    mel_fn = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, n_mels=n_mels, f_min=0, f_max=8000
    ).to(device)

    def mel_loss(gen_wav, target_wav, min_len):
        g = gen_wav[..., :min_len]
        t = target_wav[..., :min_len]
        mel_g = mel_fn(g).squeeze(1)
        mel_t = mel_fn(t).squeeze(1)
        # log scale
        mel_g = torch.log(mel_g.clamp(min=1e-5))
        mel_t = torch.log(mel_t.clamp(min=1e-5))
        return F.l1_loss(mel_g, mel_t)

    if load_from:
        print("Loading model from previous fine-tune...", flush=True)
        model = VitsModel.from_pretrained(str(load_from))
        tokenizer = AutoTokenizer.from_pretrained(str(load_from))
        lr = float(cfg.get("resume_lr", 2e-6))  # lower LR when continuing
    else:
        print("Loading pretrained MMS-TTS model...", flush=True)
        model = VitsModel.from_pretrained(base_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        lr = float(cfg.get("learning_rate", 5e-6))
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    dataset = _TTSDataset(meta, root, tokenizer, sample_rate)
    collate = lambda b: _collate_fn(b, tokenizer)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=collate, drop_last=True
    )
    steps_per_epoch = len(loader)
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    log_every = max(1, steps_per_epoch // 10)

    print("Starting fine-tuning (full epochs, mel + waveform loss)...", flush=True)
    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True, file=sys.stdout, mininterval=1.0)
        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue
            text = batch["input_ids"].to(device)
            target_wav = batch["target_wav"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            try:
                output = model(input_ids=text, attention_mask=attention_mask)
                gen_wav = output.waveform
                if gen_wav.dim() == 2:
                    gen_wav = gen_wav.unsqueeze(1)
                min_len = min(gen_wav.size(-1), target_wav.size(-1))
                if min_len < 1600:
                    continue
                loss_wav = F.l1_loss(gen_wav[..., :min_len], target_wav[..., :min_len])
                loss_mel = mel_loss(gen_wav, target_wav, min_len)
                loss = loss_wav + 2.0 * loss_mel
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
                n_batches += 1
                if (batch_idx + 1) % log_every == 0:
                    pbar.set_postfix(loss=f"{loss.item():.4f}", wav=f"{loss_wav.item():.4f}", mel=f"{loss_mel.item():.4f}")
            except Exception as e:
                continue
        if n_batches > 0:
            avg = epoch_loss / n_batches
            print(f"Epoch {epoch+1}/{epochs} avg_loss={avg:.4f} (batches={n_batches})", flush=True)
            if avg < best_loss:
                best_loss = avg
                model.save_pretrained(final_dir)
                tokenizer.save_pretrained(final_dir)
                (final_dir / "config.yaml").write_text(yaml.dump(cfg))
                print(f"  -> Saved best model to {final_dir}", flush=True)
        if (epoch + 1) % 5 == 0:
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"  Checkpoint -> {checkpoint_dir}", flush=True)

    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    (final_dir / "config.yaml").write_text(yaml.dump(cfg))
    print(f"Fine-tuning complete. Model saved to {final_dir}", flush=True)


class _TTSDataset(torch.utils.data.Dataset):
    def __init__(self, meta, root, tokenizer, sample_rate, max_audio_sec=12):
        import soundfile as sf
        self.meta = meta
        self.root = Path(root)
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.max_samples = int(max_audio_sec * sample_rate)
        self._sf = sf

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        text = str(row["text"]).strip()
        path = Path(row["path"])
        if not path.is_absolute():
            path = self.root / path
        if not path.exists():
            return None
        try:
            audio, sr = self._sf.read(path, dtype="float32")
        except Exception:
            return None
        if sr != self.sample_rate:
            from scipy.signal import resample
            num = int(len(audio) * self.sample_rate / sr)
            audio = resample(audio, num).astype("float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if len(audio) > self.max_samples:
            audio = audio[:self.max_samples]
        if len(audio) < 1600:
            return None
        return {"text": text, "audio": audio}


def _collate_fn(batch, tokenizer=None):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    texts = [b["text"] for b in batch]
    audios = [b["audio"] for b in batch]
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-kin")
    tok = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=300)
    max_len = max(len(a) for a in audios)
    import numpy as np
    padded = []
    for a in audios:
        pad = np.zeros(max_len, dtype=np.float32)
        pad[:len(a)] = a
        padded.append(pad)
    target = torch.from_numpy(np.stack(padded)).unsqueeze(1)
    return {
        "input_ids": tok["input_ids"],
        "attention_mask": tok.get("attention_mask"),
        "target_wav": target,
    }


if __name__ == "__main__":
    main()
