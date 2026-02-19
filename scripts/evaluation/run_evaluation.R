# Run TTS evaluation: generate WAVs for 5 fixed sentences, measure latency, model size; write reports/eval_report.json.
# Pure R where possible; calls inference API (local) or Python script to synthesize and measure.

options(stringsAsFactors = FALSE)
script_dir <- getwd()
if (exists("sys.frame") && length(sys.frame()) > 0) {
  ofile <- sys.frame(1)$ofile
  if (!is.null(ofile) && length(ofile) > 0 && is.character(ofile)) {
    script_dir <- dirname(ofile)
  }
}
ROOT <- normalizePath(file.path(script_dir, "..", ".."))

sentences <- c(
  "Muraho, nagufasha gute uyu munsi?",
  "Niba ufite ibibazo bijyanye n'ubuzima bwawe, twagufasha.",
  "Ni ngombwa ko ubonana umuganga vuba.",
  "Twabanye nawe kandi tuzakomeza kukwitaho.",
  "Ushobora kuduhamagara igihe cyose ukeneye ubufasha."
)

model_dir <- file.path(ROOT, "artifacts", "final_model")
eval_out_dir <- file.path(ROOT, "reports", "eval_wavs")
dir.create(eval_out_dir, recursive = TRUE, showWarnings = FALSE)

# Option A: call local inference API (must be running)
inference_url <- Sys.getenv("TTS_INFERENCE_URL", "http://localhost:8000")
use_api <- TRUE

latencies_ms <- numeric(length(sentences))
if (use_api && requireNamespace("httr", quietly = TRUE)) {
  for (i in seq_along(sentences)) {
    body <- list(text = sentences[i])
    t0 <- as.numeric(Sys.time()) * 1000
    r <- tryCatch(
      httr::POST(paste0(inference_url, "/synthesize"), body = body, encode = "json", httr::timeout(10)),
      error = function(e) NULL
    )
    t1 <- as.numeric(Sys.time()) * 1000
    latencies_ms[i] <- t1 - t0
    if (!is.null(r) && httr::status_code(r) == 200) {
      cont <- httr::content(r)
      wav_b64 <- cont$wav_base64
      if (!is.null(wav_b64)) {
        raw_wav <- base64enc::base64decode(wav_b64)
        out_path <- file.path(eval_out_dir, sprintf("eval_%02d.wav", i))
        writeBin(raw_wav, out_path)
      }
      if (!is.null(cont$latency_ms)) latencies_ms[i] <- cont$latency_ms
    }
  }
} else {
  # Option B: no API — write placeholder report and WAV paths as NA
  latencies_ms[] <- NA_real_
}

# Model size (dir size in MB)
model_size_mb <- if (dir.exists(model_dir)) {
  sum(file.info(list.files(model_dir, recursive = TRUE, full.names = TRUE))$size, na.rm = TRUE) / (1024 * 1024)
} else {
  NA_real_
}

eval_report <- list(
  timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ"),
  sentences = as.list(sentences),
  mean_latency_ms = mean(latencies_ms, na.rm = TRUE),
  p95_latency_ms = if (all(is.na(latencies_ms))) NA else quantile(latencies_ms, 0.95, na.rm = TRUE),
  latencies_ms = as.list(latencies_ms),
  model_size_mb = round(model_size_mb, 2),
  model_dir = model_dir,
  eval_wav_dir = eval_out_dir
)

report_path <- file.path(ROOT, "reports", "eval_report.json")
dir.create(dirname(report_path), recursive = TRUE, showWarnings = FALSE)
jsonlite::write_json(eval_report, report_path, auto_unbox = TRUE, pretty = TRUE)
message("Wrote ", report_path)
