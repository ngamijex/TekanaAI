# Clean and export: silence trimming, loudness normalization, clip length 2–12 s, optional segmentation.
# Reads data/raw_loaded/ (metadata + WAVs); writes data/processed/wav/*.wav and data/processed/metadata.csv.
# Pure R (tuneR + seewave or av for trim/normalize).

options(stringsAsFactors = FALSE)
script_dir <- getwd()
if (exists("sys.frame") && length(sys.frame()) > 0) {
  ofile <- sys.frame(1)$ofile
  if (!is.null(ofile) && length(ofile) > 0 && is.character(ofile)) {
    script_dir <- dirname(ofile)
  }
}
ROOT <- normalizePath(file.path(script_dir, "..", ".."))

# Dependencies: tuneR (WAV read/write), jsonlite (optional, for config)
if (!requireNamespace("tuneR", quietly = TRUE)) {
  stop("Install tuneR for 03_clean_export. Run: install.packages('tuneR')")
}

raw_meta_path <- file.path(ROOT, "data", "raw_loaded", "metadata.csv")
if (!file.exists(raw_meta_path)) stop("Run 01_load_dataset.R and 02_audit_report.R first. Missing: ", raw_meta_path)

meta <- utils::read.csv(raw_meta_path)
meta$duration_sec <- as.numeric(meta$duration_sec)

# Parameters
min_dur <- 2
max_dur <- 12
target_lufs <- -23   # loudness normalization target (dB LUFS)
silence_thresh <- 0.01  # threshold for silence trim

process_one <- function(wav_path, out_path, trim_silence = TRUE, normalize_loudness = TRUE) {
  w <- tuneR::readWave(wav_path)
  sr <- w@samp.rate
  left <- as.numeric(w@left)
  if (w@stereo) {
    right <- as.numeric(w@right)
    mono <- (left + right) / 2
  } else {
    mono <- left
  }
  mono <- mono / (2^15)
  # Silence trim: find first/last above threshold
  if (trim_silence && length(mono) > 0) {
    above <- which(abs(mono) > silence_thresh)
    if (length(above) > 0) {
      mono <- mono[seq(min(above), max(above))]
    }
  }
  if (length(mono) == 0) return(NA)
  # Loudness normalize (RMS-based simple normalization to approximate LUFS)
  if (normalize_loudness) {
    rms <- sqrt(mean(mono^2, na.rm = TRUE))
    if (rms > 1e-10) {
      # Target RMS ~ 10^(target_lufs/20) * small factor
      target_rms <- 10^(target_lufs / 20) * 0.1
      mono <- mono * (target_rms / rms)
      mono <- pmin(pmax(mono, -1), 1)
    }
  }
  w_out <- tuneR::Wave(round(mono * 32767), samp.rate = sr, bit = 16)
  tuneR::writeWave(w_out, out_path)
  length(mono) / sr
}

processed_dir <- file.path(ROOT, "data", "processed")
wav_out_dir <- file.path(processed_dir, "wav")
dir.create(wav_out_dir, recursive = TRUE, showWarnings = FALSE)

out_rows <- list()
idx <- 0
for (i in seq_len(nrow(meta))) {
  path_in <- meta$path[i]
  if (!file.exists(path_in)) next
  dur_orig <- meta$duration_sec[i]
  if (dur_orig < min_dur || dur_orig > max_dur) next
  fname <- basename(path_in)
  path_out <- file.path(wav_out_dir, fname)
  dur_new <- tryCatch(
    process_one(path_in, path_out, trim_silence = TRUE, normalize_loudness = TRUE),
    error = function(e) NA
  )
  if (is.na(dur_new)) next
  if (dur_new < min_dur || dur_new > max_dur) next
  idx <- idx + 1
  out_rows[[idx]] <- data.frame(
    path = path_out,
    text = meta$text[i],
    speaker_id = meta$speaker_id[i],
    duration_sec = round(dur_new, 4),
    split = meta$split[i]
  )
}

if (length(out_rows) == 0) stop("No clips passed filters. Check raw_loaded WAVs and duration range.")
processed_meta <- do.call(rbind, out_rows)
processed_meta_path <- file.path(processed_dir, "metadata.csv")
# Write to temp dir first to avoid 'Permission denied' (OneDrive lock or file open in Excel)
tmp_path <- file.path(tempdir(), "tekana_metadata.csv")
err_msg <- character(0)
tryCatch({
  utils::write.csv(processed_meta, tmp_path, row.names = FALSE)
  ok <- file.copy(tmp_path, processed_meta_path, overwrite = TRUE)
  if (!ok) err_msg <- paste0("file.copy to ", processed_meta_path, " returned FALSE.")
}, error = function(e) err_msg <<- conditionMessage(e))
if (file.exists(tmp_path)) tryCatch(file.remove(tmp_path), error = function(e) NULL)
if (length(err_msg) > 0) {
  stop("Failed to write ", processed_meta_path, ": ", err_msg,
       ". If 'Permission denied': close the file in Excel/editor, or right-click OneDrive icon -> Pause syncing, then retry.")
}
message("Wrote ", nrow(processed_meta), " clips to ", processed_meta_path, " and WAVs to ", wav_out_dir)
