# Execution Order (R-First)

Run from project root `TekanaAI/`. All pipeline scripts are in R except a small Python helper for parquet audio decode.

## Prerequisites

- R 4.x with: `arrow`, `tuneR`, `jsonlite`, `httr`, `base64enc`, `plumber`, `reticulate`, `shiny`
- Python 3.10+ (for 01_load_dataset helper and inference engine): `datasets`, `soundfile`, `numpy`, `pyyaml`, `pandas`
- Optional: Docker for full stack

---

## Phase 1 — Data pipeline (R + Python helper)

```r
# In R (RStudio or R script):
setwd("path/to/TekanaAI")
source("scripts/data_pipeline/01_load_dataset.R")   # calls Python to decode parquet → WAV + metadata
source("scripts/data_pipeline/02_audit_report.R")   # pure R: report → reports/dataset_report.json
source("scripts/data_pipeline/03_clean_export.R")   # pure R: trim, normalize, 2–12s → data/processed/
```

Or from shell (Rscript):

```bash
cd path/to/TekanaAI
Rscript scripts/data_pipeline/01_load_dataset.R
Rscript scripts/data_pipeline/02_audit_report.R
Rscript scripts/data_pipeline/03_clean_export.R
```

- **01**: Requires Python with `datasets`, `soundfile`. Writes `data/raw_loaded/wav/*.wav` and `data/raw_loaded/metadata.csv`.
- **02**: Reads `data/raw_loaded/metadata.csv`; writes `reports/dataset_report.json`.
- **03**: Reads raw WAVs + metadata; writes `data/processed/wav/*.wav` and `data/processed/metadata.csv`.

---

## Phase 2 — Training (R orchestrates Python)

```r
setwd("path/to/TekanaAI")
source("scripts/training/run_training.R")
```

- Calls `scripts/training/train_mms_tts.py` with `config/mms_tts.yaml`. Checkpoints → `artifacts/checkpoints`, final → `artifacts/final_model`.

---

## Phase 3 — Evaluation (R)

```r
# Start inference first (Phase 4), then:
setwd("path/to/TekanaAI")
source("scripts/evaluation/run_evaluation.R")
```

- Calls local inference API; generates WAVs for 5 sentences; writes `reports/eval_report.json` and `reports/eval_wavs/*.wav`. Set `TTS_INFERENCE_URL` if API is not on localhost:8000.

---

## Phase 4 — Inference (R Plumber)

```r
setwd("path/to/TekanaAI")
source("inference/run_plumber.R")
```

- Serves `POST /synthesize` (text, optional speaker_id → WAV base64 + latency_ms). Port 8000.

---

## Phase 5 — Shiny UI (R)

```r
setwd("path/to/TekanaAI")
shiny::runApp("shiny_app", port = 3838)
```

- Set `TTS_INFERENCE_URL=http://localhost:8000` if needed. UI: text input, speaker selector, Generate, playback, download, latency badge, 5 quick sentences.

---

## Phase 6 — Full system (Docker)

```bash
cd path/to/TekanaAI
docker compose up
```

- Uses root-level `docker-compose.yml`. Builds and runs inference (8000) and Shiny (3838). One command starts the entire system.

---

## File manifest (R-first)

| Order | Script | Language | Purpose |
|-------|--------|----------|---------|
| 1 | `scripts/data_pipeline/01_load_dataset.R` | R | Drive Python helper; output WAV + metadata |
| 2 | `scripts/data_pipeline/decode_parquet_to_wav.py` | Python | Decode HF parquet audio → WAV |
| 3 | `scripts/data_pipeline/02_audit_report.R` | R | Audit → dataset_report.json |
| 4 | `scripts/data_pipeline/03_clean_export.R` | R | Clean → processed WAV + metadata |
| 5 | `scripts/training/run_training.R` | R | Orchestrate training |
| 6 | `scripts/training/train_mms_tts.py` | Python | TTS training loop |
| 7 | `scripts/evaluation/run_evaluation.R` | R | Eval 5 sentences, latency, report |
| 8 | `inference/api_plumber.R` + `run_plumber.R` | R | Plumber API; `tts_engine.py` (reticulate) |
| 9 | `shiny_app/app.R` | R | Shiny UI |
| 10 | `docker/Dockerfile.inference`, `Dockerfile.shiny`, `docker-compose.yml` | — | Dockerization |
