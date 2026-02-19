# Orchestrate TTS training: read config, call Python training script.
# Run from project root or setwd(ROOT). Requires Python env with torch, datasets, soundfile, pyyaml.

options(stringsAsFactors = FALSE)
script_dir <- getwd()
if (exists("sys.frame") && length(sys.frame()) > 0) {
  ofile <- sys.frame(1)$ofile
  if (!is.null(ofile) && length(ofile) > 0 && is.character(ofile)) {
    script_dir <- dirname(ofile)
  }
}
ROOT <- normalizePath(file.path(script_dir, "..", ".."))

config_path <- file.path(ROOT, "config", "mms_tts.yaml")
if (!file.exists(config_path)) stop("Config not found: ", config_path)

python_script <- file.path(script_dir, "train_mms_tts.py")
if (!file.exists(python_script)) stop("Training script not found: ", python_script)

python <- Sys.which("python")
if (nzchar(Sys.getenv("VIRTUAL_ENV"))) {
  venv_python <- file.path(Sys.getenv("VIRTUAL_ENV"), "Scripts", "python.exe")
  if (!file.exists(venv_python)) venv_python <- file.path(Sys.getenv("VIRTUAL_ENV"), "bin", "python")
  if (file.exists(venv_python)) python <- venv_python
}

message("Starting training with config: ", config_path)
# -u = unbuffered Python so detailed training logs stream to R console in real time
cmd <- paste(c(shQuote(python), "-u", shQuote(python_script), "--config", shQuote(normalizePath(config_path))), collapse = " ")
result <- system(cmd)
if (result != 0) {
  stop("Training failed with status ", result)
}
message("Training finished. Checkpoints: ", file.path(ROOT, "artifacts", "checkpoints"), ", final: ", file.path(ROOT, "artifacts", "final_model"))
