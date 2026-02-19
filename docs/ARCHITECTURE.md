# Kinyarwanda TTS — Production Architecture

## System Overview (Text Description)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         OFFLINE / ONE-TIME PIPELINES                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│  data/ (parquet shards)                                                          │
│       │                                                                          │
│       ▼                                                                          │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐             │
│  │ 01_load_dataset  │───▶│ 02_audit_report │───▶│ 03_clean_export │             │
│  │ (load parquet)   │    │ (duration, etc) │    │ (trim, norm,     │             │
│  │                  │    │ → report JSON   │    │  2–12s, → WAV)   │             │
│  └─────────────────┘    └─────────────────┘    └────────┬────────┘             │
│                                                          │                      │
│                                                          ▼                      │
│                                               data/processed/wav/                │
│                                               data/processed/metadata.csv        │
│                                                          │                      │
│                                                          ▼                      │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐             │
│  │ 04_train_tts    │◀───│ config/         │    │ 05_evaluate     │             │
│  │ (MMS-TTS fine-  │    │ mms_tts.yaml   │    │ (5 sentences,    │             │
│  │  tune, ckpts)   │    │                 │    │  latency, size) │             │
│  └────────┬────────┘    └─────────────────┘    └────────┬────────┘             │
│           │                                              │                      │
│           ▼                                              ▼                      │
│  artifacts/checkpoints/                          reports/eval_report.json       │
│  artifacts/final_model/                                                          │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                         RUNTIME (PRODUCTION)                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────────────┐         HTTP (localhost)         ┌──────────────────┐  │
│  │  R Shiny App         │  ◀─────────────────────────────▶ │  Python FastAPI  │  │
│  │  (UI: text, speaker, │   POST /synthesize               │  Inference       │  │
│  │   generate, play,    │   → WAV bytes, latency_ms        │  (model loaded   │  │
│  │   download, 5 btns)  │                                  │   once at start) │  │
│  └──────────────────────┘                                  └────────┬─────────┘  │
│           │                                                          │           │
│           │  Docker network (or host)                                │           │
│           ▼                                                          ▼           │
│  ┌──────────────────────┐                                  artifacts/final_model │
│  │  shiny container     │                                  (read-only mount)    │
│  │  port 3838           │                                                       │
│  └──────────────────────┘                                                       │
│  ┌──────────────────────┐                                                       │
│  │  inference container │  port 8000                                            │
│  └──────────────────────┘                                                       │
│                                                                                  │
│  docker compose up  →  both services start; Shiny calls inference by service     │
│                        name (e.g. http://inference:8000).                        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

- **Ingestion**: Local parquet only. No upload, no external APIs.
- **Training input**: WAV files on disk + metadata CSV (path, text, speaker_id). Choice: export WAV for compatibility with MMS-TTS dataloaders, reproducibility, and standard tooling.
- **Inference**: Text → FastAPI → model → WAV bytes → Shiny plays/downloads.

## Key Engineering Decisions

| Decision | Choice | Justification |
|----------|--------|----------------|
| Training data format | Export WAV | MMS-TTS expects file paths; segment slicing and augmentation easier; no custom parquet dataloader. |
| Base model | facebook/mms-tts | Multilingual, low-resource friendly, fine-tunable, <1GB; Coqui/Piper/VITS-from-scratch less suitable for 3-speaker Kinyarwanda. |
| Inference API | FastAPI | Single process, load model once, async-ready, simple contract (text → WAV + latency). |
| UI | Pure R Shiny (no shinydashboard) | Single app.R + www/ CSS; communicates with inference via httr/curl. |

## Latency and Size Targets

- Inference latency: &lt; 800 ms for ~10-word sentences (mean and p95 measured in evaluation).
- Model size: &lt; 1 GB (target &lt; 200 MB with MMS-TTS small).
