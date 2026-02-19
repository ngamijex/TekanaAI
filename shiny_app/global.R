# Load TTS model directly via reticulate (no API required).
# Run from project root: setwd("path/to/TekanaAI"); shiny::runApp("shiny_app")

# Ensure synthesize() always exists so the app never fails with "could not find function 'synthesize'"
synthesize <- function(text, speaker_id = NULL) {
  stop("TTS engine did not load. Check the R console for errors from global.R.")
}

if (!requireNamespace("reticulate", quietly = TRUE)) {
  message("TTS: Install reticulate: install.packages('reticulate')")
} else if (!requireNamespace("base64enc", quietly = TRUE)) {
  message("TTS: Install base64enc: install.packages('base64enc')")
} else {
  tryCatch({
    # Find project root (where inference/tts_engine.py lives)
    app_dir <- getwd()
    candidates <- if (basename(app_dir) == "shiny_app") {
      list(normalizePath(file.path(app_dir, ".."), mustWork = FALSE), app_dir)
    } else {
      list(app_dir, normalizePath(file.path(app_dir, ".."), mustWork = FALSE))
    }
    ROOT <- NULL
    for (r in candidates) {
      p <- file.path(r, "inference", "tts_engine.py")
      if (file.exists(p)) {
        ROOT <- normalizePath(r)
        break
      }
    }
    if (is.null(ROOT)) stop("TTS engine not found. Run from project root: setwd('path/to/TekanaAI'); runApp('shiny_app')")
    tts_engine_path <- file.path(ROOT, "inference", "tts_engine.py")

    Sys.setenv(TTS_PROJECT_ROOT = ROOT)
    reticulate::source_python(tts_engine_path)

    py_synthesize <- synthesize
    synthesize <<- function(text, speaker_id = NULL) {
      res <- py_synthesize(text, speaker_id)
      wav_bytes <- res[[1]]
      if (inherits(wav_bytes, "python.builtin.bytes")) {
        wav_bytes <- reticulate::py_to_r(wav_bytes)
      }
      list(
        wav_base64 = base64enc::base64encode(wav_bytes),
        latency_ms = as.numeric(res[[2]])
      )
    }
    message("TTS: Model loaded from ", ROOT)
  }, error = function(e) {
    message("TTS load failed: ", conditionMessage(e))
    synthesize <<- function(text, speaker_id = NULL) {
      stop("TTS engine failed to load: ", conditionMessage(e))
    }
  })
}
