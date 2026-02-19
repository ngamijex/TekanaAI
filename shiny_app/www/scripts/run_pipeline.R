# =============================================================================
# Kinyarwanda TTS — Master Pipeline
# =============================================================================
# Runs the full pipeline from data loading to model readiness:
#   1. Load dataset (parquet -> WAV + metadata)
#   2. Audit report
#   3. Clean & export (trim, normalize, filter 2–12 s)
#   4. Train TTS model
#   5. Evaluation (optional; requires inference API running for latency metrics)
#
# Usage: Rscript run_pipeline.R              # run all steps
#        Rscript run_pipeline.R 1 2 3        # run only steps 1, 2, 3
#        Rscript run_pipeline.R --skip-if-exists   # skip steps whose outputs exist
#
# After completion, start the inference API and Shiny app to use the model.
# =============================================================================

options(stringsAsFactors = FALSE)

# Parse args
args <- commandArgs(trailingOnly = TRUE)
SKIP_IF_EXISTS <- "--skip-if-exists" %in% args
steps_arg <- as.integer(args[args != "--skip-if-exists"])
RUN_STEPS <- if (length(steps_arg) > 0) steps_arg else 1:5

# Detect project root
args0 <- commandArgs(trailingOnly = FALSE)
farg <- args0[grepl("^--file=", args0)]
if (length(farg) > 0) {
  ROOT <- normalizePath(dirname(sub("^--file=", "", farg[1])))
} else {
  ROOT <- normalizePath(getwd())
}

message("=== Kinyarwanda TTS Pipeline ===")
message("Project root: ", ROOT)
message("Steps to run: ", paste(RUN_STEPS, collapse = ", "))
if (SKIP_IF_EXISTS) message("(Skipping steps whose outputs already exist)")
setwd(ROOT)

run_step <- function(name, fun, required_output = NULL) {
  if (SKIP_IF_EXISTS && !is.null(required_output) && file.exists(required_output)) {
    message("[SKIP] ", name, " (output exists: ", basename(required_output), ")")
    return(invisible(TRUE))
  }
  message("\n--- ", name, " ---")
  tryCatch({
    fun()
    message("[OK] ", name)
  }, error = function(e) {
    message("[FAIL] ", name, ": ", conditionMessage(e))
    stop(e)
  })
}

# -----------------------------------------------------------------------------
# Step 1: Load dataset (parquet -> WAV + metadata)
# -----------------------------------------------------------------------------
step1 <- function() {
  wd_old <- getwd()
  on.exit(setwd(wd_old))
  setwd(file.path(ROOT, "scripts", "data_pipeline"))
  source(file.path(ROOT, "scripts", "data_pipeline", "01_load_dataset.R"), local = new.env())
}

# -----------------------------------------------------------------------------
# Step 2: Audit report
# -----------------------------------------------------------------------------
step2 <- function() {
  wd_old <- getwd()
  on.exit(setwd(wd_old))
  setwd(file.path(ROOT, "scripts", "data_pipeline"))
  source(file.path(ROOT, "scripts", "data_pipeline", "02_audit_report.R"), local = new.env())
}

# -----------------------------------------------------------------------------
# Step 3: Clean and export
# -----------------------------------------------------------------------------
step3 <- function() {
  wd_old <- getwd()
  on.exit(setwd(wd_old))
  setwd(file.path(ROOT, "scripts", "data_pipeline"))
  source(file.path(ROOT, "scripts", "data_pipeline", "03_clean_export.R"), local = new.env())
}

# -----------------------------------------------------------------------------
# Step 4: Train model
# -----------------------------------------------------------------------------
step4 <- function() {
  wd_old <- getwd()
  on.exit(setwd(wd_old))
  setwd(file.path(ROOT, "scripts", "training"))
  source(file.path(ROOT, "scripts", "training", "run_training.R"), local = new.env())
}

# -----------------------------------------------------------------------------
# Step 5: Evaluation
# -----------------------------------------------------------------------------
step5 <- function() {
  wd_old <- getwd()
  on.exit(setwd(wd_old))
  setwd(file.path(ROOT, "scripts", "evaluation"))
  source(file.path(ROOT, "scripts", "evaluation", "run_evaluation.R"), local = new.env())
}

# -----------------------------------------------------------------------------
# Run pipeline
# -----------------------------------------------------------------------------
if (1 %in% RUN_STEPS)
  run_step("1. Load dataset", step1,
    required_output = file.path(ROOT, "data", "raw_loaded", "metadata.csv"))

if (2 %in% RUN_STEPS)
  run_step("2. Audit report", step2,
    required_output = file.path(ROOT, "reports", "dataset_report.json"))

if (3 %in% RUN_STEPS)
  run_step("3. Clean and export", step3,
    required_output = file.path(ROOT, "data", "processed", "metadata.csv"))

if (4 %in% RUN_STEPS)
  run_step("4. Train model", step4,
    required_output = file.path(ROOT, "artifacts", "final_model"))

if (5 %in% RUN_STEPS)
  run_step("5. Evaluation", step5,
    required_output = file.path(ROOT, "reports", "eval_report.json"))

# -----------------------------------------------------------------------------
# Done — instructions for Shiny
# -----------------------------------------------------------------------------
message("\n", strrep("=", 60))
message("Pipeline complete. Model is ready.")
message(strrep("=", 60))
message("\nTo use the model in the Shiny app:\n")
message("  1. Start the inference API (in a separate terminal):")
message("     Rscript inference/run_plumber.R")
message("     (or: setwd('inference'); plumber::pr_run(plumber::plumb('api_plumber.R'), port=8000))\n")
message("  2. Start the Shiny app (in another terminal):")
message("     shiny::runApp('shiny_app')\n")
message("  3. Open the app in your browser and generate speech.")
message(strrep("=", 60))
