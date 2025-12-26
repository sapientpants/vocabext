# VocabExt

German vocabulary extraction tool with Anki sync.

## Features

- Upload German documents (PDF, PPTX, audio, text)
- Extract vocabulary using spaCy
- Enrich with grammar via local LLM (Ollama)
- Review and accept/reject words
- Sync to Anki for spaced repetition

## Setup

```bash
# Install dependencies (includes spaCy model)
uv sync

# Run the app
uv run uvicorn app.main:app --reload
```

## Configuration

Copy `.env.sample` to `.env` and adjust as needed. All settings have sensible defaults.

### Logging

By default, logs are written to the console only. To enable file logging:

```bash
LOG_FILE_ENABLED=true
LOG_FILE_PATH=data/vocabext.log  # optional, defaults to data/vocabext.log
LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

Log files are automatically rotated at 10MB with 5 backups kept.

## Requirements

- Ollama running locally for LLM enrichment
- Anki with AnkiConnect plugin for sync
