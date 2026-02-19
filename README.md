# Kinyarwanda TTS

Production-ready Kinyarwanda Text-to-Speech with **R** data pipeline, **R** Shiny UI, and **R** Plumber inference (Python used only for parquet audio decode and TTS engine via reticulate).

## Quick start (after data and model are ready)

1. Start inference: `Rscript inference/run_plumber.R` (port 8000).
2. Start Shiny: `shiny::runApp("shiny_app", port = 3838)`.
3. Or run everything: `docker compose up`.

## Execution order

See **docs/EXECUTION_ORDER.md**. Summary:

1. **Data pipeline (R)**: `01_load_dataset.R` → `02_audit_report.R` → `03_clean_export.R` (01 calls a small Python helper to decode parquet audio).
2. **Training (R calls Python)**: `run_training.R` → writes to `artifacts/`.
3. **Evaluation (R)**: `run_evaluation.R` (calls inference API) → `reports/eval_report.json`.
4. **Inference**: R Plumber at port 8000.
5. **UI**: R Shiny at port 3838.
6. **Docker**: `docker compose up` from project root.

## R packages

- Data: `tuneR`, `jsonlite` (and Python with `datasets`, `soundfile` for 01).
- Inference: `plumber`, `jsonlite`, `base64enc`, `reticulate`.
- Shiny: `shiny`, `httr`, `jsonlite`, `base64enc`.

Install: `install.packages(c("tuneR","jsonlite","httr","base64enc","plumber","reticulate","shiny"))`

## Docs

- **docs/ARCHITECTURE.md** — system design and data flow.
- **docs/PROJECT_STRUCTURE.md** — folder layout.
- **docs/TECHNICAL_WRITEUP_OUTLINE.md** — write-up outline (model choice, evaluation, latency, limitations).
