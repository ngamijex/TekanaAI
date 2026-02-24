# Kinyarwanda TTS — Tekana Project
# Two tabs: Writings (detailed technical write-up) and Model demonstration (TTS demo).
# Model runs directly in the app via reticulate (no API). Run from project root: runApp("shiny_app")

library(shiny)
library(base64enc)
source("writings_content.R", local = TRUE)

QUICK_SENTENCES <- c(
  "Muraho, nagufasha gute uyu munsi?",
  "Niba ufite ibibazo bijyanye n'ubuzima bwawe, twagufasha.",
  "Ni ngombwa ko ubonana umuganga vuba.",
  "Twabanye nawe kandi tuzakomeza kukwitaho.",
  "Ushobora kuduhamagara igihe cyose ukeneye ubufasha."
)

# ----- TTS: load model and define synthesize() in .GlobalEnv -----
.GlobalEnv$tts_load_error <- ""
.GlobalEnv$synthesize <- function(text, speaker_id = NULL) {
  msg <- get("tts_load_error", envir = .GlobalEnv)
  stop(if (nzchar(msg)) msg else "TTS engine failed to load. Check the R console for details.")
}

.tts_init <- function() {
  # 1. Check reticulate
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    .GlobalEnv$tts_load_error <- "R package 'reticulate' is not installed. Run: install.packages('reticulate')"
    return(invisible(NULL))
  }

  # 2. Check Python
  if (!reticulate::py_available(initialize = TRUE)) {
    .GlobalEnv$tts_load_error <- paste0(
      "Python not found. Install Python 3.10+ and run: ",
      "pip install transformers torch soundfile scipy. ",
      "Then restart R and re-run the app."
    )
    return(invisible(NULL))
  }

  # 3. Find tts_engine.py
  wd <- getwd()
  candidates <- unique(c(
    if (basename(wd) == "shiny_app") normalizePath(file.path(wd, ".."), mustWork = FALSE) else character(0),
    wd,
    normalizePath(file.path(wd, ".."), mustWork = FALSE)
  ))
  tts_path <- NULL
  ROOT     <- NULL
  for (r in candidates) {
    if (!nzchar(r) || is.na(r)) next
    p <- file.path(r, "inference", "tts_engine.py")
    if (file.exists(p)) { tts_path <- normalizePath(p); ROOT <- normalizePath(r); break }
  }
  if (is.null(tts_path)) {
    .GlobalEnv$tts_load_error <- paste0(
      "inference/tts_engine.py not found. ",
      "Run the app from the project root: setwd('path/to/TekanaAI'); shiny::runApp('shiny_app'). ",
      "Searched in: ", paste(candidates, collapse = ", ")
    )
    return(invisible(NULL))
  }

  # 4. Source the Python engine — source_python puts functions in the LOCAL env
  tryCatch({
    Sys.setenv(TTS_PROJECT_ROOT = ROOT)
    # Use a dedicated environment so source_python has a clean namespace to write into
    tts_env <- new.env(parent = emptyenv())
    reticulate::source_python(tts_path, envir = tts_env)

    if (!exists("synthesize", envir = tts_env, inherits = FALSE)) {
      stop("synthesize() not found in tts_engine.py after source_python.")
    }
    py_synth <- tts_env$synthesize

    .GlobalEnv$synthesize <- function(text, speaker_id = NULL) {
      res <- py_synth(text, speaker_id)
      wav <- res[[1]]
      if (inherits(wav, "python.builtin.bytes")) wav <- reticulate::py_to_r(wav)
      list(wav_base64 = base64enc::base64encode(wav), latency_ms = as.numeric(res[[2]]))
    }
    .GlobalEnv$tts_load_error <- ""
    message("TTS engine ready. Project root: ", ROOT)
  }, error = function(e) {
    msg <- conditionMessage(e)
    .GlobalEnv$tts_load_error <- paste0(
      "Python engine failed to load: ", msg, ". ",
      "Make sure these packages are installed in your Python env: ",
      "transformers torch soundfile scipy"
    )
    .GlobalEnv$synthesize <- function(text, speaker_id = NULL) {
      stop(get("tts_load_error", envir = .GlobalEnv))
    }
  })
}

.tts_init()

# Writings tab: detailed content from writings_content.R (methodology, training, evaluation, formulas)
writings_html <- function() writings_html_content()

# ----- UI -----
ui <- fluidPage(
  tags$head(
    tags$link(rel = "stylesheet", href = "style.css"),
    tags$meta(name = "viewport", content = "width=device-width, initial-scale=1")
  ),
  class = "app-root",

  div(class = "about-me",
    div(class = "about-me-inner",
      img(src = "didier.jfif", alt = "Didier", class = "about-photo", onerror = "this.style.display='none'"),
      div(class = "about-text",
        h1("Didier"),
        p("Kinyarwanda TTS — Tekana Project"),
        p(class = "about-desc", "Passionate and innovative Data Scientist with extensive experience in designing and implementing large-scale data solutions. Proven expertise in building end-to-end data pipelines, cloud-native architectures, and real-time analytics platforms.")
      )
    )
  ),

  navbarPage(
    title = NULL,
    id = "main_tabs",
    tabPanel("Writings", value = "writings",
      div(class = "tab-content writeup-tab",
        writings_html_content()
      )
    ),
    tabPanel("Model demonstration", value = "demo",
      div(class = "tab-content demo-tab",
        p(class = "demo-intro", "Generate speech from Kinyarwanda text. The fine-tuned model runs directly in the app (no separate API). First synthesis may take a moment while the model loads."),
        div(class = "card quick-sentences",
          span(class = "label", "Required evaluation sentences"),
          div(class = "quick-btns",
            lapply(seq_along(QUICK_SENTENCES), function(i) {
              actionButton(
                inputId = paste0("quick_", i),
                label = substring(QUICK_SENTENCES[i], 1, 42),
                class = "btn btn-quick"
              )
            })
          )
        ),
        uiOutput("tts_status_ui"),
        div(class = "card",
          span(class = "label", "Text to synthesize"),
          textAreaInput(
            inputId = "text",
            label = NULL,
            value = "",
            placeholder = "Andika umugambo wa Kinyarwanda...",
            rows = 4
          ),
          div(class = "demo-options",
            span(class = "label", "Speaker"),
            selectInput(
              inputId = "speaker_id",
              label = NULL,
              choices = c("Speaker 1" = "0", "Speaker 2" = "1", "Speaker 3" = "2"),
              selected = "0"
            )
          ),
          actionButton("generate", "Generate speech", class = "btn btn-primary"),
          uiOutput("error_msg_ui"),
          uiOutput("latency_badge_ui"),
          div(class = "audio-row",
            uiOutput("audio_ui"),
            uiOutput("download_ui")
          )
        )
      )
    )
  )
)

# ----- Server -----
server <- function(input, output, session) {
  last_result <- reactiveVal(NULL)
  error_msg <- reactiveVal(NULL)
  latency_ms <- reactiveVal(NULL)

  observeEvent(input$generate, {
    txt <- trimws(input$text)
    if (!nzchar(txt)) {
      error_msg("Please enter text.")
      return()
    }
    error_msg(NULL)
    run_synthesis(txt)
  })

  for (i in seq_along(QUICK_SENTENCES)) {
    local({
      ii <- i
      observeEvent(input[[paste0("quick_", ii)]], {
        updateTextAreaInput(session, "text", value = QUICK_SENTENCES[ii])
        error_msg(NULL)
        run_synthesis(QUICK_SENTENCES[ii])
      })
    })
  }

  run_synthesis <- function(text) {
    error_msg(NULL)
    latency_ms(NULL)
    updateActionButton(session, "generate", label = "Generating...", icon = NULL)
    speaker <- input$speaker_id
    speaker_id <- if (nzchar(speaker)) as.integer(speaker) else NULL
    result <- tryCatch(
      get("synthesize", envir = .GlobalEnv)(text, speaker_id = speaker_id),
      error = function(e) {
        error_msg(paste("Inference error:", conditionMessage(e)))
        NULL
      }
    )
    updateActionButton(session, "generate", label = "Generate speech", icon = NULL)
    if (!is.null(result)) {
      last_result(result)
      latency_ms(result$latency_ms)
    }
  }

  output$tts_status_ui <- renderUI({
    err <- get("tts_load_error", envir = .GlobalEnv)
    if (!nzchar(err)) return(NULL)
    div(class = "error-msg",
        style = "margin-bottom:12px;",
        tags$strong("TTS setup issue: "),
        err
    )
  })

  output$error_msg_ui <- renderUI({
    msg <- error_msg()
    if (is.null(msg)) return(div(class = "hidden"))
    div(class = "error-msg", msg)
  })

  output$latency_badge_ui <- renderUI({
    ms <- latency_ms()
    if (is.null(ms)) return(div(class = "hidden"))
    ms_num <- suppressWarnings(as.numeric(ms))
    if (is.na(ms_num)) return(div(class = "hidden"))
    div(class = "badge success", paste0(round(ms_num, 0), " ms latency"))
  })

  output$audio_ui <- renderUI({
    res <- last_result()
    if (is.null(res) || is.null(res$wav_base64)) return(NULL)
    tags$audio(
      src = paste0("data:audio/wav;base64,", res$wav_base64),
      controls = TRUE,
      style = "max-width: 100%;"
    )
  })

  output$download_ui <- renderUI({
    res <- last_result()
    if (is.null(res) || is.null(res$wav_base64)) return(NULL)
    downloadLink("download_wav", "Download WAV", class = "btn btn-quick")
  })

  output$download_wav <- downloadHandler(
    filename = function() paste0("kinyarwanda_tts_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".wav"),
    content = function(file) {
      res <- last_result()
      if (!is.null(res$wav_base64)) writeBin(base64decode(res$wav_base64), file)
    }
  )
}

shinyApp(ui = ui, server = server)
