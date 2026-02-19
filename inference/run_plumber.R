# Start TTS inference API (Plumber). Set TTS_PROJECT_ROOT to project root so artifacts/final_model is found.
options(stringsAsFactors = FALSE)
infer_dir <- getwd()
if (exists("sys.frame") && length(sys.frame()) > 0 && !is.null(sys.frame(1)$ofile)) {
  infer_dir <- dirname(sys.frame(1)$ofile)
} else {
  args <- commandArgs(trailingOnly = FALSE)
  farg <- args[grepl("^--file=", args)]
  if (length(farg)) infer_dir <- dirname(normalizePath(sub("^--file=", "", farg[1])))
}
infer_dir <- normalizePath(infer_dir)
root <- Sys.getenv("TTS_PROJECT_ROOT")
if (!nzchar(root)) root <- normalizePath(file.path(infer_dir, ".."))
Sys.setenv(TTS_PROJECT_ROOT = root)
setwd(infer_dir)
pr <- plumber::plumb("api_plumber.R")
message("Starting TTS inference on http://0.0.0.0:8000")
pr$run(host = "0.0.0.0", port = 8000)
