# LLM Pipeline Documentation

This document provides Mermaid sequence diagrams for all processing pipelines in the vocabext application, distinguishing between **local processing** (spaCy) and **LLM API calls** (OpenAI).

## Overview

The application uses a hybrid approach with a **unified pipeline** for all word processing:
- **Local processing**: spaCy for tokenization, POS detection, and lemma normalization (fast, free, offline)
- **LLM API**: OpenAI for translations and grammar details only (single call per word)

### Key Design Principles

1. **Single LLM call per word** - All enrichment data is fetched in one API call using a unified schema
2. **spaCy for all POS detection** - Both file processing and manual word addition use spaCy
3. **Alphabetic-only words** - Only words containing purely alphabetic characters are processed

### When Each Is Used

| Operation | Local (spaCy) | LLM (OpenAI) |
|-----------|---------------|--------------|
| Tokenize document text | Yes | No |
| Extract POS from document | Yes | No |
| Detect POS (manual word add) | Yes | No |
| Normalize lemma | Yes | No |
| Validate lemma exists | Yes | No |
| Get translations | No | Yes |
| Get noun gender | No | Yes |
| Get noun plural | No | Yes |
| Get verb conjugations | No | Yes |

---

## Unified Processing Pipeline

Both `vocab add` and `process file` commands now use the **same pipeline**:

```mermaid
sequenceDiagram
    participant Input as User Input
    participant spaCy as spaCy (LOCAL)
    participant Enricher
    participant API as OpenAI API
    participant DB as Database

    Input->>spaCy: Word + optional context

    Note over spaCy: LOCAL - POS Detection & Lemma Normalization
    spaCy->>spaCy: Analyze word with nlp()
    spaCy->>spaCy: Determine POS (NOUN, VERB, ADJ, etc.)
    spaCy->>spaCy: Normalize lemma based on POS
    Note over spaCy: NOUN: capitalize, strip diminutive<br/>VERB: convert participle to infinitive<br/>ADJ: convert to base form
    spaCy-->>Enricher: TokenInfo(pos, lemma, context)

    Note over Enricher,API: SINGLE API CALL - Unified Enrichment
    Enricher->>API: Prompt + UNIFIED_SCHEMA
    Note over API: Returns ALL fields in one call:<br/>translations, gender, plural,<br/>preterite, past_participle, auxiliary<br/>(null for non-applicable fields)
    API-->>Enricher: EnrichmentResult

    Enricher->>DB: Create Word record
    Note over DB: lemma_source = "spacy"
```

---

## Pipeline 1: Document Processing

When processing files, spaCy tokenizes the entire document and extracts all relevant words.

Located in `app/services/tokenizer.py` and `app/cli/commands/process.py`.

```mermaid
sequenceDiagram
    participant User
    participant CLI as process file
    participant Extractor as Text Extractor
    participant spaCy as spaCy (LOCAL)
    participant Tokenizer as Tokenizer (LOCAL)
    participant DB as Database
    participant Enricher
    participant API as OpenAI API

    User->>CLI: process file document.pdf

    Note over CLI,Tokenizer: LOCAL PROCESSING - No API calls
    CLI->>Extractor: Extract text
    Note over Extractor: PDF, PPTX, TXT, MD, Audio (Whisper)
    Extractor-->>CLI: Raw text

    CLI->>Tokenizer: tokenize(text)
    Tokenizer->>spaCy: nlp(text)
    Note over spaCy: Sentence segmentation, POS tagging, Lemmatization
    spaCy-->>Tokenizer: Doc with tokens

    loop For each token
        Tokenizer->>Tokenizer: Filter by POS (NOUN, VERB, ADJ, ADV, ADP)
        Tokenizer->>Tokenizer: Skip non-alphabetic tokens
        Tokenizer->>Tokenizer: Skip participles, ordinals, stopwords
        Tokenizer->>Tokenizer: Normalize lemma (diminutives, contractions)
        Tokenizer->>Tokenizer: Deduplicate by (lemma, pos)
    end

    Tokenizer-->>CLI: List[TokenInfo]

    CLI->>DB: Check for duplicates
    DB-->>CLI: Existing lemmas
    CLI->>CLI: Filter to new words only

    Note over CLI,API: SINGLE API CALL per word
    par Parallel enrichment (semaphore limited)
        loop For each new word
            CLI->>Enricher: enrich(lemma, pos)
            Enricher->>API: Unified schema request
            API-->>Enricher: EnrichmentResult
            Enricher-->>CLI: EnrichmentResult
        end
    end

    CLI->>DB: Batch insert Word records
    Note over DB: lemma_source = "spacy"
    CLI-->>User: Summary (X new words added)
```

---

## Pipeline 2: Manual Word Addition

When adding a word via `vocab add`, spaCy's `analyze_word()` method determines POS.

Located in `app/cli/commands/vocabulary.py` and `app/services/tokenizer.py`.

```mermaid
sequenceDiagram
    participant User
    participant CLI as vocab add
    participant Validator
    participant Tokenizer as Tokenizer (LOCAL)
    participant spaCy as spaCy (LOCAL)
    participant Enricher
    participant API as OpenAI API
    participant DB as Database

    User->>CLI: vocab add "Hund" --context "Der Hund bellt"

    CLI->>Validator: Validate input
    Validator->>Validator: Check non-empty
    Validator->>Validator: Check alphabetic only
    Note over Validator: Rejects: "Wort123", "Baden-Württemberg"<br/>Accepts: "Größe", "über", "Hund"

    Note over CLI,spaCy: LOCAL - POS Detection (same as document processing)
    CLI->>Tokenizer: analyze_word("Hund", context)
    Tokenizer->>spaCy: nlp(context or word)
    spaCy-->>Tokenizer: Doc with tokens
    Tokenizer->>Tokenizer: Find matching token
    Tokenizer->>Tokenizer: Extract POS and normalize lemma
    Tokenizer-->>CLI: TokenInfo(pos="NOUN", lemma="Hund")

    CLI->>DB: Check for duplicate (lemma, pos, gender)
    DB-->>CLI: No duplicate found

    Note over CLI,API: SINGLE API CALL - Unified Enrichment
    CLI->>Enricher: enrich("Hund", "NOUN")
    Enricher->>API: Unified schema request
    API-->>Enricher: {gender: "der", plural: "Hunde", translations: ["dog"], ...}
    Enricher-->>CLI: EnrichmentResult

    CLI->>DB: Create Word record
    Note over DB: lemma_source = "spacy"
    CLI-->>User: Display enriched word
```

---

## Pipeline 3: Core LLM Service

The foundational LLM service with retry logic and concurrency control.

Located in `app/services/llm.py`.

```mermaid
sequenceDiagram
    participant Caller
    participant llm as llm.chat_completion()
    participant Semaphore
    participant Client as AsyncOpenAI
    participant API as OpenAI API

    Note over Caller,API: SINGLE API CALL

    Caller->>llm: chat_completion(prompt, schema, schema_name)

    llm->>llm: get_client() [lazy init]
    llm->>Semaphore: acquire (max 100 concurrent)

    loop Max 3 retries
        llm->>Client: responses.create()

        Note over Client,API: model: gpt-5-mini, input: prompt, text.format: json_schema

        alt Success
            API-->>Client: Response with output_text
            Client-->>llm: response
            llm->>llm: Parse JSON from output_text
            llm->>Semaphore: release
            llm-->>Caller: dict[str, Any]
        else Retryable Error (429, 5xx, timeout, connection)
            API-->>Client: Error
            llm->>llm: Backoff: min(1.0 * 2^attempt + jitter, 30)
            llm->>llm: await asyncio.sleep(delay)
        else Non-Retryable Error (4xx except 429)
            API-->>Client: Error
            llm->>Semaphore: release
            llm-->>Caller: Raise APIStatusError
        end
    end
```

---

## Pipeline 4: Word Enrichment (POS-Specific Schemas)

Single LLM call per word using POS-specific schemas for tighter validation.

Located in `app/services/enricher.py`.

```mermaid
sequenceDiagram
    participant Caller
    participant Enricher as enricher.enrich()
    participant llm as llm.chat_completion()
    participant API as OpenAI API

    Note over Caller,API: SINGLE API CALL per word

    Caller->>Enricher: enrich(lemma, pos)

    Enricher->>Enricher: Select schema for POS
    Note over Enricher: NOUN → NOUN_SCHEMA<br/>VERB → VERB_SCHEMA<br/>ADP → PREPOSITION_SCHEMA<br/>OTHER → WORD_SCHEMA

    Enricher->>llm: chat_completion(prompt, POS_SCHEMA)
    llm->>API: Structured JSON request
    API-->>llm: POS-specific fields
    llm-->>Enricher: Parsed dict

    Enricher->>Enricher: Build EnrichmentResult
    Note over Enricher: Fields populated based on POS:<br/>- lemma, translations (always)<br/>- gender, plural (nouns only)<br/>- preterite, past_participle, aux (verbs only)

    Enricher-->>Caller: EnrichmentResult
```

### POS-Specific Schemas

**NOUN_SCHEMA** - For nouns:
```json
{
  "properties": {
    "lemma": {"type": "string"},
    "gender": {"type": "string", "enum": ["der", "die", "das"]},
    "plural": {"type": "string"},
    "translations": {"type": "array", "items": {"type": "string"}}
  },
  "required": ["lemma", "gender", "plural", "translations"]
}
```

**VERB_SCHEMA** - For verbs:
```json
{
  "properties": {
    "lemma": {"type": "string"},
    "preterite": {"type": "string"},
    "past_participle": {"type": "string"},
    "auxiliary": {"type": "string", "enum": ["haben", "sein"]},
    "translations": {"type": "array", "items": {"type": "string"}}
  },
  "required": ["lemma", "preterite", "past_participle", "auxiliary", "translations"]
}
```

**PREPOSITION_SCHEMA** - For prepositions (ADP):
```json
{
  "properties": {
    "lemma": {"type": "string"},
    "cases": {"type": "array", "items": {"type": "string", "enum": ["akkusativ", "dativ", "genitiv"]}},
    "translations": {"type": "array", "items": {"type": "string"}}
  },
  "required": ["lemma", "cases", "translations"]
}
```

**WORD_SCHEMA** - For adjectives, adverbs:
```json
{
  "properties": {
    "lemma": {"type": "string"},
    "translations": {"type": "array", "items": {"type": "string"}}
  },
  "required": ["lemma", "translations"]
}
```

---

## Vocab Validate Command

Re-enriches existing words using POS-specific schemas.

```mermaid
sequenceDiagram
    participant User
    participant CLI as vocab validate
    participant DB as Database
    participant Enricher
    participant API as OpenAI API

    User->>CLI: vocab validate

    CLI->>DB: Fetch all words
    DB-->>CLI: List[Word]

    loop For each word (parallel with semaphore)
        Note over CLI,API: SINGLE API CALL per word
        CLI->>Enricher: enrich(word.lemma, word.pos)
        Enricher->>API: POS-specific schema request
        API-->>Enricher: EnrichmentResult

        Enricher-->>CLI: EnrichmentResult

        alt Fields changed
            CLI->>DB: Update word record
        end
    end

    CLI->>DB: Commit batch
    CLI-->>User: Summary of changes
```

---

## Summary: What Goes Where

### Local Processing (spaCy)

```mermaid
flowchart LR
    subgraph LOCAL["LOCAL PROCESSING (No API)"]
        A[Document Text] --> B[spaCy Tokenizer]
        B --> C[POS Tagging]
        B --> D[Lemmatization]
        B --> E[Sentence Segmentation]

        F[Manual Word Input] --> G[spaCy analyze_word]
        G --> C
        G --> D
    end
```

### LLM Processing (OpenAI)

```mermaid
flowchart LR
    subgraph LLM["LLM API (Single Call)"]
        A[Word + POS + Context] --> B[Unified Enrichment]
        B --> C[Translations]
        B --> D[Gender if NOUN]
        B --> E[Plural if NOUN]
        B --> F[Conjugations if VERB]
    end
```

---

## Input Validation

Words must pass validation before processing:

```mermaid
flowchart TD
    A[Input Word] --> B{Empty?}
    B -->|Yes| C[Reject]
    B -->|No| D{Alphabetic only?}
    D -->|No| E[Reject]
    D -->|Yes| F[Process]
```

**Accepted**: `Hund`, `Größe`, `über`, `Straße`
**Rejected**: `Wort123`, `Baden-Württemberg`, `test@email`, `word.`

---

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `openai_api_key` | (required) | OpenAI API key from environment |
| `openai_model` | `gpt-5-mini` | Model to use for all LLM calls |
| `spacy_model` | `de_core_news_lg` | German spaCy model for tokenization |

---

## API Call Comparison

### Before (Multiple Calls)
```
vocab add "Hund":
  1. detect_pos() -> LLM call
  2. validate_lemma() -> LLM call (sometimes)
  3. enrich() -> LLM call
  = 2-3 API calls per word
```

### After (Single Call)
```
vocab add "Hund":
  1. analyze_word() -> spaCy (local)
  2. enrich() -> LLM call
  = 1 API call per word
```

This reduces API costs by 60-70% and improves response time.
