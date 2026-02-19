# =============================================================================
# Kinyarwanda TTS — Start inference API + Shiny app (single entry point)
# =============================================================================
# Starts the Plumber TTS API in the background, waits until it is ready,
# then runs the Shiny app. Use this when running locally or deploying so the
# app works with the trained model without starting the API manually.
#
# Usage:   Rscript run_app.R
#   or in R: source("run_app.R")
#
# Deploy:  Run this script as your app entry point (e.g. Rscript run_app.R)
#          so both the API and the Shiny app start; the app will use the
#          trained model at artifacts/final_model.
# Optional: install.packages("processx") for cleaner background API process.
# =============================================================================

options(stringsAsFactors = FALSE)

# Project root (same logic as run_pipeline.R)
args0 <- commandArgs(trailingOnly = FALSE)
farg <- args0[grepl("^--file=", args0)]
if (length(farg) > 0) {
  ROOT <- normalizePath(dirname(sub("^--file=", "", farg[1])))
} else {
  ROOT <- normalizePath(getwd())
}

# Paths
run_plumber_path <- file.path(ROOT, "inference", "run_plumber.R")
shiny_app_path <- file.path(ROOT, "shiny_app")
if (!file.exists(run_plumber_path)) stop("Not found: ", run_plumber_path)
if (!dir.exists(shiny_app_path)) stop("Not found: ", shiny_app_path)

# Ensure API can find project root
Sys.setenv(TTS_PROJECT_ROOT = ROOT)

# Check if API is already up (reuse existing)
api_ready <- function() {
  tryCatch({
    r <- httr::GET("http://127.0.0.1:8000/health", timeout(2))
    httr::status_code(r) == 200
  }, error = function(e) FALSE)
}

api_process <- NULL

if (api_ready()) {
  message("TTS API already running on http://127.0.0.1:8000")
} else {
  message("Starting TTS inference API in background...")
  Rscript <- file.path(R.home("bin"), "Rscript")
  if (!file.exists(Rscript)) Rscript <- "Rscript"

  if (requireNamespace("processx", quietly = TRUE)) {
    api_process <- processx::process$new(
      Rscript,
      c(run_plumber_path),
      wd = ROOT,
      stdout = "|",
      stderr = "|",
      supervise = FALSE
    )
  } else {
    system2(Rscript, run_plumber_path, wait = FALSE, stdout = NULL, stderr = NULL)
  }

  # Wait for API to be ready (up to 90 s; model load can be slow)
  for (i in 1:90) {
    Sys.sleep(1)
    if (api_ready()) {
      message("TTS API ready at http://127.0.0.1:8000")
      break
    }
    if (i == 90) stop("API did not become ready in time. Check inference/run_plumber.R and logs.")
  }
}

# Stop API when this script exits (e.g. user stops Shiny app)
if (!is.null(api_process)) {
  on.exit(try(api_process$kill(), silent = TRUE), add = TRUE)
}

# Run Shiny app (uses http://localhost:8000 by default)
message("Starting Shiny app...")
setwd(ROOT)
shiny::runApp(shiny_app_path, launch.browser = TRUE)
