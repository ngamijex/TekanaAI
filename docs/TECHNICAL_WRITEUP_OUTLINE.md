# Technical Write-Up — Outline

1. **Model choice reasoning**
   - Why MMS-TTS (facebook/mms-tts) over Coqui TTS, Piper, VITS-from-scratch: multilingual support, low-resource fine-tuning, model size &lt; 1GB, single checkpoint.

2. **Data cleaning decisions**
   - Silence trimming threshold and method (waveform amplitude).
   - Loudness normalization target (e.g. -23 LUFS / RMS-based).
   - Clip length filter (2–12 s): justification from duration distribution in audit.
   - Decision to export WAV (not train from parquet): tooling, reproducibility, segment slicing.

3. **Training hyperparameters**
   - Batch size, epochs, learning rate, sample rate.
   - Base model and checkpoint selection.
   - Train/validation split usage.

4. **Stopping criteria**
   - Epoch-based vs. early stopping.
   - Perceptual quality checks on fixed eval sentences (subjective or metric).

5. **Evaluation results**
   - Table: 5 sentences, mean latency (ms), p95 latency (ms), model size (MB).
   - WAV outputs stored in `reports/eval_wavs/`.
   - Reference: `reports/eval_report.json`.

6. **Latency discussion**
   - Cold vs. warm inference.
   - Target &lt; 800 ms for ~10-word sentences; measured mean/p95.
   - Where time is spent (model forward, WAV encoding).

7. **Limitations**
   - Dataset size and speaker coverage.
   - Domain (e.g. finance/legal) vs. general Kinyarwanda.
   - No external APIs; local only.

8. **AI tools used**
   - List of tools (e.g. Cursor, model cards, documentation) used during design and implementation.
