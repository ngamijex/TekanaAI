"""
Helper: load local parquet shards (HF audio column), export WAV + metadata CSV.
Uses pyarrow + soundfile directly to avoid datasets' torchcodec requirement.
Called from R 01_load_dataset.R with args: data_dir out_dir
"""
import io
import os
import sys
import glob
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

TARGET_SR = 16000

def decode_audio_bytes(audio_bytes, target_sr=TARGET_SR):
    """Decode HF audio bytes (WAV/FLAC etc) and optionally resample."""
    buf = io.BytesIO(audio_bytes)
    arr, sr = sf.read(buf, dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)  # mono
    if sr != target_sr:
        from scipy.signal import resample
        num = int(len(arr) * target_sr / sr)
        arr = resample(arr, num).astype("float32")
        sr = target_sr
    return arr, sr

def main():
    if len(sys.argv) < 3:
        print("Usage: decode_parquet_to_wav.py <data_dir> <out_dir>", file=sys.stderr)
        sys.exit(1)
    data_dir = sys.argv[1]
    out_dir = os.path.abspath(sys.argv[2])
    wav_dir = os.path.join(out_dir, "wav")
    os.makedirs(wav_dir, exist_ok=True)

    train_files = sorted(glob.glob(os.path.join(data_dir, "train-*-of-*.parquet")))
    valid_files = sorted(glob.glob(os.path.join(data_dir, "validation-*-of-*.parquet")))
    test_files = sorted(glob.glob(os.path.join(data_dir, "test-*-of-*.parquet")))

    def run_split(name, files):
        if not files:
            return pd.DataFrame()
        # Concatenate multiple parquet shards
        tables = [pq.read_table(f) for f in files]
        table = pa.concat_tables(tables)
        rows = []
        for i in range(table.num_rows):
            row = table.slice(i, 1)
            audio_col = row.column("audio")
            val = audio_col[0]
            if val is None:
                continue
            if hasattr(val, "as_py"):
                val = val.as_py()
            if isinstance(val, dict):
                audio_bytes = val.get("bytes")
            else:
                audio_bytes = getattr(val, "bytes", None) or (val[0] if isinstance(val, (list, tuple)) else None)
            if not audio_bytes:
                continue
            try:
                arr, sr = decode_audio_bytes(audio_bytes)
            except Exception as e:
                print(f"Skip row {i}: {e}", file=sys.stderr)
                continue
            path = os.path.join(wav_dir, f"{name}_{i:06d}.wav")
            sf.write(path, arr, sr)
            dur = len(arr) / sr
            tc = row.column("text")[0]
            text = tc.as_py() if hasattr(tc, "as_py") else str(tc)
            spk = row.column("speaker_id")[0]
            spk = spk.as_py() if hasattr(spk, "as_py") else int(spk)
            rows.append({"split": name, "path": path, "text": text, "speaker_id": int(spk), "duration_sec": round(dur, 4)})
        return pd.DataFrame(rows)

    dfs = []
    for name, files in [("train", train_files), ("validation", valid_files), ("test", test_files)]:
        if files:
            dfs.append(run_split(name, files))
    if not dfs:
        print("No parquet files found.", file=sys.stderr)
        sys.exit(1)
    meta = pd.concat(dfs, ignore_index=True)
    csv_path = os.path.join(out_dir, "metadata.csv")
    meta.to_csv(csv_path, index=False)
    print(f"Wrote {len(meta)} rows to {csv_path}")

if __name__ == "__main__":
    main()
