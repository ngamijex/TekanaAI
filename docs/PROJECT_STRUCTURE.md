# Project Folder Structure

```
TekanaAI/
├── config/
│   └── mms_tts.yaml              # TTS training config
├── data/
│   ├── README.md                 # Dataset info (existing)
│   ├── train-00000-of-00003.parquet
│   ├── train-00001-of-00003.parquet
│   ├── train-00002-of-00003.parquet
│   ├── validation-00000-of-00001.parquet
│   ├── test-00000-of-00001.parquet
│   ├── raw_loaded.parquet        # (optional) output of 01_load_dataset
│   └── processed/
│       ├── metadata.csv         # path, text, speaker_id, duration
│       └── wav/
│           └── *.wav
├── artifacts/
│   ├── checkpoints/             # Training checkpoints
│   └── final_model/             # Production model
├── reports/
│   ├── dataset_report.json     # Phase 1 audit
│   ├── eval_report.json        # Phase 3 evaluation
│   └── eval_wavs/              # WAVs for 5 fixed sentences
├── scripts/
│   ├── data_pipeline/
│   │   ├── 01_load_dataset.R    # R: calls Python helper
│   │   ├── decode_parquet_to_wav.py
│   │   ├── 02_audit_report.R
│   │   └── 03_clean_export.R
│   ├── training/
│   │   ├── run_training.R       # R: orchestrates Python
│   │   └── train_mms_tts.py
│   └── evaluation/
│       └── run_evaluation.R     # R
├── inference/
│   ├── api_plumber.R            # R Plumber API
│   ├── run_plumber.R
│   └── tts_engine.py            # Python (reticulate)
├── shiny_app/
│   ├── app.R
│   └── www/
│       └── style.css
├── docker/
│   ├── Dockerfile.inference     # R + Plumber + reticulate
│   ├── Dockerfile.shiny
│   └── docker-compose.yml
├── docker-compose.yml           # at root: docker compose up
├── docs/
│   ├── ARCHITECTURE.md
│   ├── EXECUTION_ORDER.md
│   ├── PROJECT_STRUCTURE.md
│   └── TECHNICAL_WRITEUP_OUTLINE.md
├── requirements-data.txt
├── requirements-train.txt
├── requirements-inference.txt
├── TekanaAI.Rproj
└── README.md
```
