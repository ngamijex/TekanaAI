# Load Kinyarwanda TTS dataset from local parquet shards.
# Calls Python helper to decode Hugging Face audio from parquet; writes WAV + metadata. Downstream steps are pure R.
# No data leaves the machine.

options(stringsAsFactors = FALSE)
script_dir <- getwd()
if (exists("sys.frame") && length(sys.frame()) > 0 && !is.null(sys.frame(1)$ofile)) {
  script_dir <- dirname(sys.frame(1)$ofile)
} else {
  args <- commandArgs(trailingOnly = FALSE)
  farg <- args[grepl("^--file=", args)]
  if (length(farg)) script_dir <- dirname(normalizePath(sub("^--file=", "", farg[1])))
}
ROOT <- normalizePath(file.path(script_dir, "..", ".."))

data_dir <- file.path(ROOT, "data")
out_dir <- file.path(ROOT, "data", "raw_loaded")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

train_files <- Sys.glob(file.path(data_dir, "train-*-of-*.parquet"))
if (length(train_files) == 0) {
  stop("No train parquet files in ", data_dir, ". Expected train-00000-of-00003.parquet etc.")
}

python_script <- file.path(script_dir, "decode_parquet_to_wav.py")
if (!file.exists(python_script)) stop("Helper not found: ", python_script)

# Prefer venv or conda Python that has datasets, soundfile
python <- Sys.which("python")
if (nzchar(Sys.getenv("VIRTUAL_ENV"))) {
  venv_python <- file.path(Sys.getenv("VIRTUAL_ENV"), "Scripts", "python.exe")
  if (!file.exists(venv_python)) venv_python <- file.path(Sys.getenv("VIRTUAL_ENV"), "bin", "python")
  if (file.exists(venv_python)) python <- venv_python
}
result <- system2(python, c(shQuote(python_script), shQuote(data_dir), shQuote(out_dir)), stdout = TRUE, stderr = TRUE)
if (!is.null(attr(result, "status")) && attr(result, "status") != 0) {
  stop("decode_parquet_to_wav.py failed: ", paste(result, collapse = "\n"))
}
message(paste(result, collapse = "\n"))

meta_path <- file.path(out_dir, "metadata.csv")
if (!file.exists(meta_path)) stop("Expected metadata.csv at ", meta_path)
meta <- utils::read.csv(meta_path)
message("Loaded dataset: ", nrow(meta), " examples. WAVs in ", file.path(out_dir, "wav"))
for (s in unique(meta$split)) {
  message("  ", s, ": ", sum(meta$split == s))
}
