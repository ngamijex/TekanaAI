# TekanaAI — Kinyarwanda Text-to-Speech

A production-ready **Kinyarwanda TTS** system for the Tekana voice-based AI. It converts text into natural speech using a fine-tuned **VITS** (MMS-TTS) model, with a full pipeline from data preparation and training to inference, a **Shiny** web app for demonstration, and a **REST API** or **Docker** deployment.

**Live demo (deployed Shiny app):** [https://didier-ngamije.shinyapps.io/shiny_app/](https://didier-ngamije.shinyapps.io/shiny_app/) — Writings tab (technical report + downloadable scripts) and Model demonstration tab (synthesise, play, download WAV).

---

## Overview

This repository implements:

- **End-to-end pipeline**: load Kinyarwanda speech data (e.g. from Hugging Face parquet), audit and clean it, train or fine-tune the **facebook/mms-tts-kin** model, and run inference.
- **Inference options**: a simple **CLI script** (text → WAV, with **latency printed** per call), an **R Plumber API** (port 8000), and in-app synthesis in the Shiny UI (no separate API required when running the app with reticulate).
- **Shiny app**: two tabs — **Writings** (technical report and downloadable scripts) and **Model demonstration** (synthesise, play, and download WAVs; quick sentences and custom text).
- **Docker**: single-command run (`docker compose up`) to start the inference service and the Shiny app.

Design constraints targeted: inference latency under ~800 ms for short sentences, model size within 1 GB, and no storage or transmission of user voice to external APIs.

---

## What’s in This Repo

| Component | Description |
|-----------|-------------|
| **Data pipeline** | R scripts (+ Python helper) to load parquet → WAV + metadata, audit, and clean (trim, normalise, 2–12 s filter). Orchestrated by `run_pipeline.R`. |
| **Training** | R orchestrates Python training (`train_mms_tts.py`) using `config/mms_tts.yaml`; checkpoints and final model go to `artifacts/`. |
| **Inference** | `inference/tts_engine.py` (Python, used by R via reticulate), `inference/run_synthesis.py` (CLI: text → WAV, prints latency), `inference/run_plumber.R` (API on port 8000). |
| **Shiny app** | `shiny_app/`: Writings tab + Model demonstration tab; runs with `shiny::runApp("shiny_app")` from project root. |
| **Docker** | `docker-compose.yml` at root; `docker/Dockerfile.inference` and `docker/Dockerfile.shiny` for API and Shiny. |
| **Config** | `config/mms_tts.yaml` — epochs, batch size, learning rate, `max_train_samples`, `resume_from`, etc. |

**Data and model weights** are not in the repo (see `.gitignore`). You need to add your dataset (e.g. under `data/`) and/or place a trained model in `artifacts/final_model` (or use the pretrained `facebook/mms-tts-kin` by setting `TTS_USE_PRETRAINED=1`).

---

## Prerequisites

- **R** 4.x with packages: `arrow`, `tuneR`, `jsonlite`, `httr`, `base64enc`, `plumber`, `reticulate`, `shiny`
- **Python** 3.10+ for:
  - Data pipeline: `datasets`, `soundfile`, `numpy`, `pyyaml`, `pandas` (see `requirements-data.txt`)
  - Inference/training: `transformers`, `torch`, `torchaudio`, `soundfile`, `scipy` (see `requirements.txt` / `requirements-inference.txt`)
- **Optional**: Docker and Docker Compose for `docker compose up`

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ngamijex/TekanaAI.git
   cd TekanaAI
   ```

2. **R packages** (from R or RStudio)
   ```r
   install.packages(c("arrow", "tuneR", "jsonlite", "httr", "base64enc", "plumber", "reticulate", "shiny"))
   ```

3. **Python environment** (inference and simple CLI)
   ```bash
   pip install -r requirements.txt
   ```
   For data pipeline only: `pip install -r requirements-data.txt`.  
   For training: `pip install -r requirements-train.txt`.

4. **Data and model**
   - Place your Kinyarwanda dataset (e.g. parquet or already processed WAVs + `metadata.csv`) as expected by the pipeline (see **Execution order** below), or
   - Place a trained model in `artifacts/final_model` (Hugging Face–style: `config.json`, tokenizer files, weights), or
   - Use pretrained only by setting `TTS_USE_PRETRAINED=1` when running inference (no local `artifacts/final_model` required).

---

## Quick Start

**Option A — Shiny app only (in-app TTS via reticulate)**  
From project root in R:
```r
setwd("path/to/TekanaAI")
shiny::runApp("shiny_app")
# Or: runApp("shiny_app", port = 3838)
```
Open the URL shown (e.g. `http://127.0.0.1:3838`). No separate inference server needed.

**Option B — Inference API + Shiny**  
Terminal 1:
```bash
cd path/to/TekanaAI
Rscript inference/run_plumber.R
# API at http://0.0.0.0:8000
```
Terminal 2 (R): set `TTS_INFERENCE_URL=http://localhost:8000` if needed, then `shiny::runApp("shiny_app", port = 3838)`.

**Option C — Docker (single command)**  
From project root:
```bash
docker compose up
```
Inference on port 8000, Shiny on 3838 (see `docker-compose.yml`).

---

## Usage in Detail

### Master pipeline (data → training → evaluation)

Run from **project root**:

```bash
# All steps (1–5): load data, audit, clean, train, evaluate
Rscript run_pipeline.R

# Only steps 1, 2, 3 (data load, audit, clean)
Rscript run_pipeline.R 1 2 3

# Skip steps whose outputs already exist
Rscript run_pipeline.R --skip-if-exists
```

Steps: (1) Load dataset (parquet → WAV + metadata), (2) Audit report, (3) Clean & export (trim, normalise, 2–12 s), (4) Train TTS model, (5) Evaluation (optional; needs inference API). See **docs/EXECUTION_ORDER.md** for phase-by-phase instructions.

### Simple inference (text → WAV, latency printed)

From project root (requires `artifacts/final_model` or pretrained):

```bash
python inference/run_synthesis.py "Muraho, nagufasha gute?" output.wav
```

Latency (ms) is printed for each call. Omit the output path to write `synthesis_output.wav` in the current directory.

### Inference API (Plumber)

```bash
Rscript inference/run_plumber.R
```

- **POST** `/synthesize`: JSON body with `text` (and optional `speaker_id`) → returns WAV as base64 + `latency_ms`.
- **POST** `/synthesize_wav`: same input → response is raw `audio/wav`.

### Shiny app

- **Writings**: technical report; code boxes and **downloadable** main scripts (`run_pipeline.R`, `run_synthesis.py`, `requirements.txt`, `docker-compose.yml`, training config, etc.).
- **Model demonstration**: text input, quick sentences, Generate, play, download WAV, latency shown.

### Configuration

- **Training**: edit `config/mms_tts.yaml` (epochs, `batch_size`, `max_train_samples`, `resume_from`, learning rate, etc.).
- **Inference**: set `TTS_PROJECT_ROOT` to project root so `artifacts/final_model` is found; optionally `TTS_USE_PRETRAINED=1` to use only the pretrained Hugging Face model.

---

## Project Structure (summary)

```
TekanaAI/
├── config/
│   └── mms_tts.yaml           # Training config
├── data/                      # Not in repo; add dataset or use pipeline output
├── artifacts/                 # Not in repo; trained model (or from Release)
├── reports/                   # Generated reports (not in repo)
├── scripts/
│   ├── data_pipeline/         # 01 load, 02 audit, 03 clean
│   ├── training/              # run_training.R, train_mms_tts.py
│   └── evaluation/           # run_evaluation.R
├── inference/
│   ├── tts_engine.py          # Core synthesis (reticulate)
│   ├── run_synthesis.py       # CLI: text → WAV, prints latency
│   ├── run_plumber.R          # Start API
│   └── api_plumber.R          # Plumber routes
├── shiny_app/                 # Shiny app (Writings + Demo)
├── docker/
│   ├── Dockerfile.inference
│   ├── Dockerfile.shiny
│   └── docker-compose.yml
├── docker-compose.yml        # Root: docker compose up
├── run_pipeline.R            # Master pipeline (steps 1–5)
├── requirements.txt          # Python deps for inference
├── requirements-inference.txt
├── requirements-train.txt
├── requirements-data.txt
└── docs/                     # ARCHITECTURE, EXECUTION_ORDER, PROJECT_STRUCTURE, etc.
```

---

## Requirements Files

| File | Purpose |
|------|---------|
| `requirements.txt` | Main Python deps for inference (transformers, torch, soundfile, scipy). |
| `requirements-inference.txt` | Same plus optional API stack (e.g. FastAPI/uvicorn) if used. |
| `requirements-train.txt` | Training (PyTorch, transformers, datasets, etc.). |
| `requirements-data.txt` | Data pipeline (datasets, soundfile, pyarrow, etc.). |

---

## Documentation

- **docs/EXECUTION_ORDER.md** — Step-by-step execution order (data → training → evaluation → inference → Shiny).
- **docs/PROJECT_STRUCTURE.md** — Folder layout and file roles.
- **docs/ARCHITECTURE.md** — System design and data flow.
- **docs/TECHNICAL_WRITEUP_OUTLINE.md** — Outline for model choice, evaluation, latency, limitations.

The **Shiny app → Writings** tab contains the full technical report and download links for the main scripts and configs.

---

## License and Data

- Code in this repository is provided as-is for the Tekana / Kinyarwanda TTS project.
- **Dataset and model weights** are not included; add data locally and optionally publish the model via GitHub Releases or a separate link (see README or Shiny Writings for instructions).
