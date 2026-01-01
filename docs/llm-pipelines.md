# LLM Pipeline Documentation

This document provides Mermaid sequence diagrams for all processing pipelines in the vocabext application, clearly distinguishing between **local processing** (spaCy, dictionary) and **LLM API calls** (OpenAI).

## Overview

The application uses a hybrid approach:
- **Local processing**: spaCy for tokenization and lemma validation (fast, free, offline)
- **LLM API**: OpenAI for translations, grammar details, and POS detection when context is unavailable

### When Each Is Used

| Operation | Local (spaCy) | LLM (OpenAI) |
|-----------|---------------|--------------|
| Tokenize document text | Yes | No |
| Extract POS from document | Yes | No |
| Extract lemma from document | Yes | No |
| Detect POS (manual input) | No | Yes |
| Validate lemma exists | Yes | No |
| Get translations | No | Yes |
| Get noun gender | Partial | Fallback |
| Get noun plural | No | Yes |
| Get verb conjugations | No | Yes |

---

## Pipeline 1: Document Processing (Mostly Local)

When processing files, most work is done locally by spaCy. OpenAI is only called for enrichment.

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

    rect rgb(230, 255, 230)
        Note over CLI,Tokenizer: LOCAL PROCESSING - No API calls
        CLI->>Extractor: Extract text
        Note over Extractor: PDF, PPTX, TXT, MD,<br/>Audio (Whisper)
        Extractor-->>CLI: Raw text

        CLI->>Tokenizer: tokenize(text)
        Tokenizer->>spaCy: nlp(text)

        Note over spaCy: Sentence segmentation<br/>POS tagging<br/>Lemmatization

        spaCy-->>Tokenizer: Doc with tokens

        loop For each token
            Tokenizer->>Tokenizer: Filter by POS (NOUN, VERB, ADJ, ADV, ADP)
            Tokenizer->>Tokenizer: Skip participles, ordinals, stopwords
            Tokenizer->>Tokenizer: Normalize lemma (diminutives, contractions)
            Tokenizer->>Tokenizer: Deduplicate by (lemma, pos)
        end

        Tokenizer-->>CLI: List[TokenInfo]
    end

    CLI->>DB: Check for duplicates
    DB-->>CLI: Existing lemmas
    CLI->>CLI: Filter to new words only

    rect rgb(255, 240, 230)
        Note over CLI,API: API CALLS - Only for enrichment
        par Parallel enrichment (semaphore limited)
            loop For each new word
                CLI->>Enricher: enrich(lemma, pos, context)
                Enricher->>API: Get translations + grammar
                API-->>Enricher: EnrichmentResult
                Enricher-->>CLI: EnrichmentResult
            end
        end
    end

    CLI->>DB: Batch insert Word records
    CLI-->>User: Summary (X new words added)
```

---

## Pipeline 2: Manual Word Add (Mixed Local + LLM)

When adding a word manually via `vocab add`, we need LLM for POS detection (no document context), but use local dictionary for validation.

Located in `app/cli/commands/vocabulary.py` and `app/services/enricher.py`.

```mermaid
sequenceDiagram
    participant User
    participant CLI as vocab add
    participant Enricher as enrich_word()
    participant LLM_POS as detect_pos()
    participant Dict as Dictionary (LOCAL)
    participant spaCy as spaCy Backend (LOCAL)
    participant LLM_Enrich as enrich()
    participant API as OpenAI API
    participant DB as Database

    User->>CLI: vocab add "Hund" --context "Der Hund bellt"
    CLI->>Enricher: enrich_word("Hund", context, session)

    rect rgb(255, 240, 230)
        Note over Enricher,API: API CALL - POS Detection<br/>(no document context available)
        Enricher->>LLM_POS: detect_pos("Hund", context)
        LLM_POS->>API: Determine POS and normalize lemma
        API-->>LLM_POS: {pos: "NOUN", lemma: "Hund"}
        LLM_POS-->>Enricher: ("NOUN", "Hund")
    end

    Enricher->>Dict: enrich_with_dictionary("Hund", "NOUN", context)

    rect rgb(230, 255, 230)
        Note over Dict,spaCy: LOCAL - Dictionary Validation
        Dict->>spaCy: validate_and_ground_lemma("Hund", "NOUN")
        spaCy->>spaCy: Check vocabulary: nlp.vocab["Hund"].is_oov?
        spaCy-->>Dict: (lemma="Hund", is_grounded=True, source="spacy")

        Dict->>spaCy: get_enrichment_data("Hund", "NOUN")
        Note over spaCy: spaCy only validates existence,<br/>no gender/definitions/frequency
        spaCy-->>Dict: DictionaryEntry(lemma_validated=True)
    end

    rect rgb(255, 240, 230)
        Note over Dict,API: API CALL - Enrichment<br/>(translations, grammar not in dictionary)
        Dict->>LLM_Enrich: enrich("Hund", "NOUN", context)
        LLM_Enrich->>API: Get gender, plural, translations
        API-->>LLM_Enrich: {gender: "der", plural: "Hunde", translations: ["dog"]}
        LLM_Enrich-->>Dict: EnrichmentResult
    end

    Dict->>Dict: Merge dictionary + LLM results
    Dict-->>Enricher: EnrichmentResult

    Enricher-->>CLI: ("NOUN", EnrichmentResult)

    CLI->>DB: Create Word record
    CLI-->>User: Display enriched word
```

---

## Pipeline 3: Core Chat Completion (LLM Wrapper)

The foundational LLM service with retry logic and concurrency control. Located in `app/services/llm.py`.

```mermaid
sequenceDiagram
    participant Caller
    participant llm as llm.chat_completion()
    participant Semaphore
    participant Client as AsyncOpenAI
    participant API as OpenAI API

    rect rgb(255, 240, 230)
        Note over Caller,API: API CALL

        Caller->>llm: chat_completion(prompt, schema, schema_name)

        llm->>llm: get_client() [lazy init]
        llm->>Semaphore: acquire (max 100 concurrent)

        loop Max 3 retries
            llm->>Client: responses.create()

            Note over Client,API: Request Structure
            Client->>API: model: gpt-5-mini<br/>input: [{role: user, content: prompt}]<br/>text.format: json_schema

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
                llm-->>Caller: Raise LLMError
            end
        end
    end
```

---

## Pipeline 4: Word Enrichment (LLM Only)

Enriches German words with grammatical information based on part of speech. Located in `app/services/enricher.py`.

```mermaid
sequenceDiagram
    participant Caller
    participant Enricher as enricher.enrich()
    participant llm as llm.chat_completion()
    participant API as OpenAI API

    rect rgb(255, 240, 230)
        Note over Caller,API: API CALL

        Caller->>Enricher: enrich(lemma, pos, context)

        alt POS == NOUN
            Enricher->>Enricher: Build noun prompt
            Note over Enricher: Request: gender, plural, translations
            Enricher->>Enricher: Select NOUN_SCHEMA
        else POS == VERB
            Enricher->>Enricher: Build verb prompt
            Note over Enricher: Request: preterite, past_participle,<br/>auxiliary, translations
            Enricher->>Enricher: Select VERB_SCHEMA
        else POS == ADJ/ADV/ADP
            Enricher->>Enricher: Build word prompt
            Note over Enricher: Request: lemma, translations
            Enricher->>Enricher: Select WORD_SCHEMA
        end

        Enricher->>llm: chat_completion(prompt, schema)
        llm->>API: Structured JSON request
        API-->>llm: JSON response
        llm-->>Enricher: Parsed dict

        Enricher->>Enricher: Strip German articles
        Enricher->>Enricher: Build EnrichmentResult
        Enricher-->>Caller: EnrichmentResult
    end
```

---

## Pipeline 5: POS Detection (LLM Only)

Determines part of speech for words without document context. Located in `app/services/enricher.py`.

```mermaid
sequenceDiagram
    participant Caller
    participant Enricher as enricher.detect_pos()
    participant llm as llm.chat_completion()
    participant API as OpenAI API

    rect rgb(255, 240, 230)
        Note over Caller,API: API CALL<br/>(used when no document context)

        Caller->>Enricher: detect_pos(word, context)

        Enricher->>Enricher: Build POS detection prompt
        Note over Enricher: Categories: NOUN, VERB, ADJ, ADV, ADP<br/>+ rules for dictionary form

        Enricher->>llm: chat_completion(prompt, POS_DETECTION_SCHEMA)
        llm->>API: Structured JSON request
        API-->>llm: {pos: "VERB", lemma: "arbeiten"}
        llm-->>Enricher: Parsed dict

        Enricher->>Enricher: Strip articles from lemma
        Enricher-->>Caller: (pos, lemma)
    end
```

---

## Pipeline 6: Dictionary-Grounded Enrichment (Hybrid)

The main enrichment flow that combines local dictionary with LLM. Located in `app/services/enricher.py`.

```mermaid
sequenceDiagram
    participant Caller
    participant Enricher as enrich_with_dictionary()
    participant Dict as DictionaryService (LOCAL)
    participant spaCy as SpacyBackend (LOCAL)
    participant LLM as enrich()
    participant API as OpenAI API

    Caller->>Enricher: enrich_with_dictionary(lemma, pos, context, session)

    rect rgb(230, 255, 230)
        Note over Enricher,spaCy: STEP 1: LOCAL - Lemma Validation
        Enricher->>Dict: validate_and_ground_lemma(lemma, pos)
        Dict->>spaCy: validate_lemma(lemma, pos)

        spaCy->>spaCy: Check nlp.vocab[lemma].is_oov
        alt Word in vocabulary
            spaCy-->>Dict: (True, None)
        else Try lowercase/titlecase
            spaCy->>spaCy: Check variations
            spaCy-->>Dict: (True, corrected_lemma)
        else Unknown word
            Note over spaCy: Assume valid (spaCy vocab<br/>doesn't have all German words)
            spaCy-->>Dict: (True, None)
        end

        Dict-->>Enricher: (final_lemma, is_grounded, source)
    end

    rect rgb(230, 255, 230)
        Note over Enricher,spaCy: STEP 2: LOCAL - Dictionary Metadata
        Enricher->>Dict: get_enrichment_data(lemma, pos)
        Dict->>spaCy: lookup(lemma, pos)
        Note over spaCy: spaCy only provides:<br/>- lemma validation<br/>- source="spacy"<br/><br/>Does NOT provide:<br/>- gender, definitions<br/>- frequency, IPA, URL
        spaCy-->>Dict: DictionaryEntry(lemma_validated=True)
        Dict-->>Enricher: DictionaryEntry
    end

    rect rgb(255, 240, 230)
        Note over Enricher,API: STEP 3: API CALL - LLM Enrichment
        Enricher->>LLM: enrich(lemma, pos, context)
        LLM->>API: Get translations + grammar
        API-->>LLM: {translations, gender, plural, ...}
        LLM-->>Enricher: EnrichmentResult
    end

    Enricher->>Enricher: Merge results
    Note over Enricher: Dictionary: lemma validation, source<br/>LLM: translations, plural, conjugations<br/><br/>Note: Does NOT use LLM for lemma<br/>correction (spaCy is source of truth)

    Enricher-->>Caller: EnrichmentResult
```

---

## Pipeline 7: Lemma Validation (LLM - Rarely Used)

This LLM-based validation exists but is NOT used in the main flow. The `enrich_with_dictionary` intentionally uses local spaCy instead.

Located in `app/services/enricher.py`.

```mermaid
sequenceDiagram
    participant Caller
    participant Enricher as enricher.validate_lemma()
    participant llm as llm.chat_completion()
    participant API as OpenAI API

    rect rgb(255, 240, 230)
        Note over Caller,API: API CALL (rarely used)<br/>Main flow uses local spaCy instead

        Caller->>Enricher: validate_lemma(lemma, pos, context)

        Enricher->>Enricher: Build validation prompt
        Note over Enricher: POS-specific rules:<br/>NOUN: singular nominative<br/>VERB: infinitive form<br/>ADJ: base form<br/>+ German prefix warning

        Enricher->>llm: chat_completion(prompt, VALIDATION_SCHEMA)
        llm->>API: Is "{lemma}" a valid German {pos}?
        API-->>llm: {valid, corrected_lemma, reason}
        llm-->>Enricher: Parsed dict

        Enricher-->>Caller: {valid, corrected_lemma, reason}
    end
```

---

## Vocab Validate Command

Re-enriches existing words using the hybrid local+LLM flow.

```mermaid
sequenceDiagram
    participant User
    participant CLI as vocab validate
    participant DB as Database
    participant Enricher as enrich_with_dictionary()
    participant spaCy as spaCy (LOCAL)
    participant API as OpenAI API

    User->>CLI: vocab validate

    CLI->>DB: Fetch all words
    DB-->>CLI: List[Word]

    loop For each word (parallel with semaphore)
        CLI->>Enricher: enrich_with_dictionary(word.lemma, word.pos)

        rect rgb(230, 255, 230)
            Note over Enricher,spaCy: LOCAL
            Enricher->>spaCy: Validate lemma
            spaCy-->>Enricher: Validation result
        end

        rect rgb(255, 240, 230)
            Note over Enricher,API: API CALL
            Enricher->>API: Get translations + grammar
            API-->>Enricher: Enrichment data
        end

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

### Local Processing (spaCy) - FREE, FAST, OFFLINE

```mermaid
flowchart LR
    subgraph LOCAL["LOCAL PROCESSING"]
        A[Document Text] --> B[spaCy Tokenizer]
        B --> C[POS Tagging]
        B --> D[Lemmatization]
        B --> E[Sentence Segmentation]

        F[Lemma] --> G[spaCy Vocabulary Check]
        G --> H[Is word known?]
    end

    style LOCAL fill:#e6ffe6
```

### LLM Processing (OpenAI) - COSTS MONEY, REQUIRES NETWORK

```mermaid
flowchart LR
    subgraph LLM["LLM API CALLS"]
        A[Manual Word Input] --> B[detect_pos]

        C[Any Word] --> D[enrich]
        D --> E[Translations]
        D --> F[Gender]
        D --> G[Plural]
        D --> H[Verb Conjugations]
    end

    style LLM fill:#fff0e6
```

---

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `openai_api_key` | (required) | OpenAI API key from environment |
| `openai_model` | `gpt-5-mini` | Model to use for all LLM calls |
| `dictionary_enabled` | `True` | Enable local dictionary lookup |

---

## JSON Schemas (Used by LLM)

### NOUN_SCHEMA
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

### VERB_SCHEMA
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

### WORD_SCHEMA
```json
{
  "properties": {
    "lemma": {"type": "string"},
    "translations": {"type": "array", "items": {"type": "string"}}
  },
  "required": ["lemma", "translations"]
}
```

### POS_DETECTION_SCHEMA
```json
{
  "properties": {
    "pos": {"type": "string", "enum": ["NOUN", "VERB", "ADJ", "ADV", "ADP"]},
    "lemma": {"type": "string"}
  },
  "required": ["pos", "lemma"]
}
```
