# Plumber API: load TTS model once, expose POST /synthesize (text, speaker_id) -> WAV bytes + latency_ms.
# Run: Rscript run_plumber.R or in R: plumber::pr_run(plumber::pr("api_plumber.R"), port = 8000)

library(plumber)
library(jsonlite)
library(base64enc)

# Load TTS engine via reticulate (one-time). Run with setwd(inference_dir) so getwd() is inference/
if (requireNamespace("reticulate", quietly = TRUE)) {
  infer_path <- normalizePath(getwd())
  if (!nzchar(Sys.getenv("TTS_PROJECT_ROOT"))) Sys.setenv(TTS_PROJECT_ROOT = dirname(infer_path))
  reticulate::source_python(file.path(infer_path, "tts_engine.py"))
} else {
  stop("Install reticulate: install.packages('reticulate')")
}

#* Health check
#* @get /health
function() {
  list(status = "ok", service = "kinyarwanda-tts")
}

# Helper: parse JSON body safely (req$body can be raw, list, or char vector)
parse_body <- function(raw) {
  if (is.raw(raw)) raw <- rawToChar(raw)
  if (is.list(raw)) raw <- raw[[1]]
  if (length(raw) != 1) raw <- paste(as.character(raw), collapse = "")
  if (is.null(raw) || !nzchar(trimws(paste(raw, collapse = "")))) raw <- "{}"
  tryCatch(jsonlite::fromJSON(raw), error = function(e) list())
}

#* Synthesize speech from text
#* @param text Text to synthesize (Kinyarwanda)
#* @param speaker_id Optional speaker ID (integer)
#* @post /synthesize
function(req, res, text = NULL, speaker_id = NULL) {
  body <- parse_body(req$body)
  text <- if (!is.null(body$text)) body$text else if (!is.null(text)) text else ""
  speaker_id <- if (!is.null(body$speaker_id)) as.integer(body$speaker_id) else speaker_id
  if (!nzchar(trimws(text))) {
    res$status <- 400
    return(list(error = "Missing or empty text"))
  }
  out <- tryCatch({
    if (is.null(speaker_id)) {
      result <- synthesize(text)
    } else {
      result <- synthesize(text, speaker_id = as.integer(speaker_id)[1])
    }
    list(
      wav_base64 = base64enc::base64encode(result[[1]]),
      latency_ms = result[[2]]
    )
  }, error = function(e) {
    res$status <- 500
    list(error = conditionMessage(e))
  })
  res$setHeader("Content-Type", "application/json")
  return(out)
}

#* Return WAV binary directly (for download)
#* @param text Text to synthesize
#* @post /synthesize_wav
function(req, res, text = NULL) {
  body <- parse_body(req$body)
  text <- if (!is.null(body$text)) body$text else if (!is.null(text)) text else ""
  if (!nzchar(trimws(text))) {
    res$status <- 400
    return("Missing text")
  }
  result <- synthesize(text, speaker_id = NULL)
  res$setHeader("Content-Type", "audio/wav")
  res$body <- result[[1]]
  return(res$body)
}
