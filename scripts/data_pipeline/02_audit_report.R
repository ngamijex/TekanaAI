# Data audit: total duration, speaker distribution, clip duration distribution, text length distribution.
# Reads data/raw_loaded/metadata.csv; writes reports/dataset_report.json.
# Pure R.

options(stringsAsFactors = FALSE)
script_dir <- getwd()
if (exists("sys.frame") && length(sys.frame()) > 0) {
  ofile <- sys.frame(1)$ofile
  if (!is.null(ofile) && length(ofile) > 0 && is.character(ofile)) {
    script_dir <- dirname(ofile)
  }
}
ROOT <- normalizePath(file.path(script_dir, "..", ".."))

meta_path <- file.path(ROOT, "data", "raw_loaded", "metadata.csv")
if (!file.exists(meta_path)) stop("Run 01_load_dataset.R first. Missing: ", meta_path)

meta <- utils::read.csv(meta_path)
meta$text_length <- nchar(meta$text)
meta$duration_sec <- as.numeric(meta$duration_sec)

total_duration_sec <- sum(meta$duration_sec, na.rm = TRUE)
total_duration_min <- total_duration_sec / 60
n_clips <- nrow(meta)

speaker_counts <- as.list(table(meta$speaker_id))
names(speaker_counts) <- paste0("speaker_", names(speaker_counts))
speaker_distribution <- as.list(prop.table(table(meta$speaker_id)) * 100)
names(speaker_distribution) <- paste0("speaker_", names(speaker_distribution))

duration_summary <- summary(meta$duration_sec)
clip_duration_distribution <- list(
  min_sec = as.numeric(duration_summary["Min."]),
  q1_sec = as.numeric(duration_summary["1st Qu."]),
  median_sec = as.numeric(duration_summary["Median"]),
  mean_sec = as.numeric(duration_summary["Mean"]),
  q3_sec = as.numeric(duration_summary["3rd Qu."]),
  max_sec = as.numeric(duration_summary["Max."])
)

text_len_summary <- summary(meta$text_length)
text_length_distribution <- list(
  min_chars = as.numeric(text_len_summary["Min."]),
  q1_chars = as.numeric(text_len_summary["1st Qu."]),
  median_chars = as.numeric(text_len_summary["Median"]),
  mean_chars = as.numeric(text_len_summary["Mean"]),
  q3_chars = as.numeric(text_len_summary["3rd Qu."]),
  max_chars = as.numeric(text_len_summary["Max."])
)

by_split <- lapply(split(meta, meta$split), function(d) {
  list(
    n = nrow(d),
    total_duration_sec = sum(d$duration_sec, na.rm = TRUE),
    total_duration_min = sum(d$duration_sec, na.rm = TRUE) / 60
  )
})

report <- list(
  audit_timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ"),
  total_clips = n_clips,
  total_duration_sec = round(total_duration_sec, 2),
  total_duration_min = round(total_duration_min, 2),
  speaker_counts = speaker_counts,
  speaker_distribution_pct = lapply(speaker_distribution, round, 2),
  clip_duration_sec = clip_duration_distribution,
  text_length_chars = text_length_distribution,
  by_split = by_split
)

reports_dir <- file.path(ROOT, "reports")
dir.create(reports_dir, recursive = TRUE, showWarnings = FALSE)
report_path <- file.path(reports_dir, "dataset_report.json")
jsonlite::write_json(report, report_path, auto_unbox = TRUE, pretty = TRUE)
message("Wrote ", report_path)
