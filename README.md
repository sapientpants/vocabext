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

## Requirements

- Ollama running locally for LLM enrichment
- Anki with AnkiConnect plugin for sync
