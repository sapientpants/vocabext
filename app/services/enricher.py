"""LLM enrichment service using OpenAI API for German vocabulary."""

import json
import logging
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

from openai import APIConnectionError, APIStatusError, APITimeoutError
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.services.dictionary import DictionaryService
from app.services.llm import chat_completion

logger = logging.getLogger(__name__)

# German articles to strip from lemmas
GERMAN_ARTICLES = ("der ", "die ", "das ", "ein ", "eine ", "einen ", "einem ", "einer ")


def strip_article(lemma: str) -> str:
    """Remove leading German article from a lemma if present."""
    lemma_lower = lemma.lower()
    for article in GERMAN_ARTICLES:
        if lemma_lower.startswith(article):
            return lemma[len(article) :]
    return lemma


# ============================================================================
# LLM Response Models (Pydantic)
# ============================================================================
# These models define the structure of LLM responses. They are used to:
# 1. Generate JSON schemas for OpenAI's structured output mode
# 2. Parse and validate LLM responses
# 3. Provide type safety throughout the codebase


class NounResponse(BaseModel):
    """LLM response schema for German nouns."""

    model_config = ConfigDict(extra="forbid")

    lemma: str = Field(description="Correct base form (singular, nominative)")
    gender: Literal["der", "die", "das"] = Field(description="Article for singular form")
    plural: str = Field(description="Plural form without article")
    translations: list[str] = Field(description="1-3 English translations")


class VerbResponse(BaseModel):
    """LLM response schema for German verbs."""

    model_config = ConfigDict(extra="forbid")

    lemma: str = Field(description="Correct infinitive form")
    preterite: str = Field(description="3rd person singular preterite")
    past_participle: str = Field(description="Past participle without auxiliary")
    auxiliary: Literal["haben", "sein"] = Field(description="Auxiliary verb")
    translations: list[str] = Field(description="1-3 English translations")


class PrepositionResponse(BaseModel):
    """LLM response schema for German prepositions."""

    model_config = ConfigDict(extra="forbid")

    lemma: str = Field(description="Correct preposition form")
    cases: list[Literal["akkusativ", "dativ", "genitiv"]] = Field(
        description="Grammatical cases this preposition governs"
    )
    translations: list[str] = Field(description="1-3 English translations")


class WordResponse(BaseModel):
    """LLM response schema for other German words (adjectives, adverbs)."""

    model_config = ConfigDict(extra="forbid")

    lemma: str = Field(description="Correct base/dictionary form")
    translations: list[str] = Field(description="1-3 English translations")


class ValidationResponse(BaseModel):
    """LLM response schema for lemma validation."""

    model_config = ConfigDict(extra="forbid")

    valid: bool = Field(description="True if word is correct as-is")
    corrected_lemma: str = Field(description="Correct form (same as input if valid)")
    reason: str = Field(description="Explanation if invalid or corrected, empty if valid")


class POSDetectionResponse(BaseModel):
    """LLM response schema for part-of-speech detection."""

    model_config = ConfigDict(extra="forbid")

    pos: Literal["NOUN", "VERB", "ADJ", "ADV", "ADP"] = Field(description="Part of speech")
    lemma: str = Field(description="Correct base/dictionary form of the word")


class EnrichmentResult(BaseModel):
    """Result of LLM enrichment for a word."""

    lemma: str | None = None
    gender: str | None = None  # der/die/das
    plural: str | None = None
    preterite: str | None = None
    past_participle: str | None = None
    auxiliary: str | None = None  # haben/sein
    cases: list[str] = Field(default_factory=list)  # akkusativ/dativ/genitiv for prepositions
    translations: list[str] = Field(default_factory=list)
    error: str | None = None  # Error message if enrichment failed

    # Dictionary-grounded fields
    definition_de: str | None = None  # German definition
    synonyms: list[str] = Field(default_factory=list)
    frequency: float | None = None  # DWDS 0-6 scale
    ipa: str | None = None  # Pronunciation
    lemma_source: str | None = None  # "spacy", "dwds", etc.
    dictionary_url: str | None = None  # Link to entry
    dictionary_error: str | None = None  # Error from dictionary lookup (if any)


class NonGermanWordsResponse(BaseModel):
    """LLM response for non-German word detection."""

    model_config = ConfigDict(extra="forbid")

    non_german_words: list[str] = Field(
        description="List of words that are NOT German (foreign words, gibberish, etc.)"
    )


async def _check_batch_for_non_german(batch: list[str]) -> set[str]:
    """Check a single batch of words for non-German words.

    Args:
        batch: List of words to check (should be ~50 words)

    Returns:
        Set of non-German words from this batch
    """
    word_list = ", ".join(batch)

    prompt = f"""Which of these words are NOT German? Include foreign words (English, Turkish, etc.) and gibberish, but NOT valid German compound words.

Words: {word_list}

Reply with ONLY the non-German words. If all words are German, reply with an empty list."""

    try:
        result = await chat_completion(
            prompt,
            NonGermanWordsResponse.model_json_schema(),
            "non_german_words",
        )
        non_german = result.get("non_german_words", [])
        # Only include words that were in the original batch (case-insensitive match)
        # Map lowercase to original casing for lookup
        lower_to_orig = {w.lower(): w for w in batch}
        found: set[str] = set()
        for word in non_german:
            orig = lower_to_orig.get(word.lower())
            if orig:
                found.add(orig)
        return found
    except (APIStatusError, APIConnectionError, APITimeoutError) as e:
        logger.warning(f"Failed to check batch for non-German words: {e}")
        # On error, don't filter anything from this batch
        return set()


async def filter_non_german_words(
    words: list[str],
    on_progress: "Callable[[int, int], None] | None" = None,
) -> set[str]:
    """Use LLM to identify non-German words from a list.

    This is more reliable than vocabulary lookup because:
    - German compound words are valid even if not in a dictionary
    - LLM understands language patterns, not just vocabulary

    Args:
        words: List of words to check
        on_progress: Optional callback(completed, total) for progress updates

    Returns:
        Set of words that are NOT German
    """
    import asyncio

    if not words:
        return set()

    # Batch words into chunks to avoid token limits
    # ~50 words per batch should be safe
    batch_size = 50
    batches = [words[i : i + batch_size] for i in range(0, len(words), batch_size)]
    total_batches = len(batches)

    # Process all batches in parallel (semaphore in chat_completion limits concurrency)
    tasks = [asyncio.create_task(_check_batch_for_non_german(batch)) for batch in batches]

    all_non_german: set[str] = set()
    completed = 0

    for coro in asyncio.as_completed(tasks):
        result = await coro
        all_non_german.update(result)
        completed += 1
        if on_progress:
            on_progress(completed, total_batches)

    return all_non_german


class Enricher:
    """Enrich vocabulary with grammatical information using OpenAI API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        dictionary_service: DictionaryService | None = None,
    ) -> None:
        # Keep for backwards compatibility with tests
        self.api_key = api_key
        self.model = model
        self._dictionary = dictionary_service
        self._dictionary_enabled = settings.dictionary_enabled

    @property
    def dictionary(self) -> DictionaryService:
        """Get or create the dictionary service."""
        if self._dictionary is None:
            self._dictionary = DictionaryService()
        return self._dictionary

    def _build_prompt(self, lemma: str, pos: str) -> str:
        """Build the prompt for LLM enrichment based on part of speech."""
        if pos == "NOUN":
            return f"""Analyze this German noun and provide grammatical information.

Word: {lemma}

Provide:
- Gender (der/die/das)
- Plural form
- 1-3 English translations"""

        elif pos == "VERB":
            return f"""Analyze this German verb and provide grammatical information.

Word: {lemma}

Provide:
- Preterite (3rd person singular)
- Past participle
- Auxiliary (haben or sein)
- 1-3 English translations"""

        elif pos == "ADP":
            return f"""Analyze this German preposition and provide grammatical information.

Word: {lemma}

Provide:
- Cases: which grammatical cases this preposition governs (akkusativ, dativ, and/or genitiv)
- 1-3 English translations"""

        else:  # ADJ, ADV
            return f"""Analyze this German word and provide translations.

Word: {lemma}
Part of speech: {pos}

Provide 1-3 English translations."""

    def _get_schema_for_pos(self, pos: str) -> tuple[dict[str, Any], str]:
        """Get the JSON schema and name for a part of speech."""
        if pos == "NOUN":
            return NounResponse.model_json_schema(), "noun_enrichment"
        elif pos == "VERB":
            return VerbResponse.model_json_schema(), "verb_enrichment"
        elif pos == "ADP":
            return PrepositionResponse.model_json_schema(), "preposition_enrichment"
        else:
            return WordResponse.model_json_schema(), "word_enrichment"

    async def _call_chat_api(
        self, prompt: str, schema: dict[str, Any], schema_name: str
    ) -> dict[str, Any]:
        """Make a request to OpenAI's Chat Completions API with structured output."""
        return await chat_completion(prompt, schema, schema_name, model=self.model)

    async def enrich(self, lemma: str, pos: str) -> EnrichmentResult:
        """
        Enrich a word with grammatical information via OpenAI API.

        Makes a single LLM call per word using POS-specific schemas for
        tighter validation and smaller responses.

        Returns EnrichmentResult with available fields populated.

        Uses a semaphore to limit concurrent LLM requests,
        balancing throughput with API rate limits.
        """
        prompt = self._build_prompt(lemma, pos)
        schema, schema_name = self._get_schema_for_pos(pos)

        try:
            data = await self._call_chat_api(prompt, schema, schema_name)
            return self._build_result(data, pos)

        except APITimeoutError:
            error_msg = "OpenAI request timed out"
            logger.warning(f"OpenAI timeout for '{lemma}', skipping enrichment")
            return EnrichmentResult(error=error_msg)
        except APIStatusError as e:
            if e.status_code == 401:
                error_msg = "Invalid OpenAI API key. Check OPENAI_API_KEY setting."
            elif e.status_code == 404:
                error_msg = f"Model '{self.model}' not found. Check OPENAI_MODEL setting."
            elif e.status_code == 429:
                error_msg = "OpenAI rate limit exceeded. Please try again later."
            else:
                error_msg = f"OpenAI returned HTTP {e.status_code}"
            logger.error(f"OpenAI API error for '{lemma}': {e}")
            return EnrichmentResult(error=error_msg)
        except APIConnectionError:
            error_msg = "Cannot connect to OpenAI API."
            logger.error(f"OpenAI connection error for '{lemma}'")
            return EnrichmentResult(error=error_msg)
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON response: {e}"
            logger.error(f"JSON decode error for '{lemma}': {e}")
            return EnrichmentResult(error=error_msg)
        except Exception as e:
            error_msg = f"Enrichment error: {e}"
            logger.error(f"Enrichment error for '{lemma}': {e}")
            return EnrichmentResult(error=error_msg)

    def _build_result(self, data: dict[str, Any], pos: str) -> EnrichmentResult:
        """Build EnrichmentResult from parsed JSON data."""
        lemma = data.get("lemma")
        if lemma:
            lemma = strip_article(lemma)

        result = EnrichmentResult(
            lemma=lemma,
            translations=data.get("translations", []),
        )

        if pos == "NOUN":
            result.gender = data.get("gender")
            result.plural = data.get("plural")
        elif pos == "VERB":
            result.preterite = data.get("preterite")
            result.past_participle = data.get("past_participle")
            result.auxiliary = data.get("auxiliary")
        elif pos == "ADP":
            result.cases = data.get("cases", [])

        return result

    async def validate_lemma(self, lemma: str, pos: str, context: str) -> dict[str, Any]:
        """
        Validate a lemma and get the correct base form.

        Returns a dict with:
        - valid: bool - whether the input is a valid German word
        - corrected_lemma: str - the correct base form (may be same as input)
        - error: str | None - error message if validation failed
        """
        pos_info = {
            "NOUN": (
                "noun",
                "singular nominative form, non-declined (e.g., 'Arbeit' not 'Arbeiten' or 'Arbeits')",
            ),
            "VERB": (
                "verb",
                "infinitive form, non-conjugated (e.g., 'arbeiten' not 'arbeitete' or 'gearbeitet')",
            ),
            "ADJ": (
                "adjective",
                "base/predicative form, non-declined, non-inflected (e.g., 'schnell' not 'schneller' or 'schnellen')",
            ),
            "ADV": ("adverb", "base form, non-inflected"),
            "ADP": ("preposition", "standard form"),
        }.get(pos, ("word", "base/dictionary form"))
        pos_name, form_desc = pos_info

        prompt = f"""Is "{lemma}" a valid, complete German {pos_name} in its correct dictionary form?

The correct form should be: {form_desc}

Context: "{context}"

IMPORTANT: Check carefully if the word:
1. Is a COMPLETE word (not truncated or missing letters at the end)
2. Is spelled correctly (no typos or missing characters)
3. Is a real German word that exists in dictionaries
4. Is in the correct base/dictionary form (not inflected, conjugated, or declined)

CRITICAL: Do NOT remove prefixes! German compound words and prefixed words are valid distinct words:
- "Abbedingung" is valid (waiver) - do NOT change to "Bedingung"
- "Abfahrt" is valid (departure) - do NOT change to "Fahrt"
- "Einleitung" is valid (introduction) - do NOT change to "Leitung"
Keep all prefixes (Ab-, An-, Auf-, Aus-, Be-, Ein-, Er-, Ent-, Ver-, Vor-, Zer-, etc.)"""

        try:
            data = await self._call_chat_api(
                prompt, ValidationResponse.model_json_schema(), "lemma_validation"
            )
            corrected = data.get("corrected_lemma") or lemma
            corrected = strip_article(corrected)

            return {
                "valid": data.get("valid", True),
                "corrected_lemma": corrected,
                "reason": data.get("reason"),
                "error": None,
            }

        except APIStatusError as e:
            if e.status_code == 401:
                error_msg = "Invalid OpenAI API key. Check OPENAI_API_KEY setting."
            elif e.status_code == 404:
                error_msg = f"Model '{self.model}' not found. Check OPENAI_MODEL setting."
            else:
                error_msg = f"OpenAI returned HTTP {e.status_code}"
            logger.error(f"Lemma validation API error: {e}")
            return {"valid": True, "corrected_lemma": lemma, "error": error_msg}
        except APIConnectionError:
            error_msg = "Cannot connect to OpenAI API."
            logger.error("Lemma validation connection error")
            return {"valid": True, "corrected_lemma": lemma, "error": error_msg}
        except Exception as e:
            logger.error(f"Lemma validation error for '{lemma}': {e}")
            return {"valid": True, "corrected_lemma": lemma, "error": str(e)}

    async def validate_and_enrich(
        self, lemma: str, pos: str, context: str = ""
    ) -> EnrichmentResult:
        """
        Validate lemma and get enrichment in one operation.

        First validates the lemma is correct, then enriches with grammatical info.
        If lemma needs correction, uses the corrected lemma for enrichment.

        Returns EnrichmentResult with all available fields populated.
        """
        # First validate lemma
        validation = await self.validate_lemma(lemma, pos, context)

        if validation.get("error"):
            return EnrichmentResult(
                lemma=lemma,
                error=validation["error"],
            )

        # Use corrected lemma if different
        lemma_to_use = validation.get("corrected_lemma", lemma)

        # Get enrichment with validated/corrected lemma
        result = await self.enrich(lemma_to_use, pos)

        # If lemma was corrected, include the correction in result
        if lemma_to_use != lemma:
            result.lemma = lemma_to_use
            logger.info(f"Lemma corrected: '{lemma}' -> '{lemma_to_use}'")

        return result

    async def enrich_with_dictionary(
        self,
        lemma: str,
        pos: str,
        context: str,
        session: AsyncSession | None = None,
    ) -> EnrichmentResult:
        """
        Enrich a word using dictionary lookup first, then LLM for remaining data.

        This method:
        1. Validates the lemma using local dictionary (spaCy vocabulary)
        2. Uses LLM for translations, plural forms, verb conjugations
        3. Merges dictionary and LLM data

        Args:
            lemma: The word lemma to enrich
            pos: Part of speech (NOUN, VERB, ADJ, ADV, ADP)
            context: Context sentence (currently unused, reserved for future use)
            session: Database session for dictionary caching

        Returns:
            EnrichmentResult with combined dictionary and LLM data
        """
        result = EnrichmentResult(lemma=lemma, lemma_source="spacy")

        # Step 1: Validate and potentially correct lemma using dictionary
        validated_lemma = lemma
        if self._dictionary_enabled:
            try:
                (
                    validated_lemma,
                    is_grounded,
                    source,
                ) = await self.dictionary.validate_and_ground_lemma(lemma, pos, session)

                if is_grounded:
                    result.lemma = validated_lemma
                    result.lemma_source = source or "dictionary"
                    if validated_lemma != lemma:
                        logger.info(
                            f"Lemma corrected by {source}: '{lemma}' -> '{validated_lemma}'"
                        )

            except Exception as e:
                logger.warning(f"Dictionary validation failed for '{lemma}': {e}")
                result.dictionary_error = f"Validation failed: {e}"
                validated_lemma = lemma

        # Step 2: Get additional dictionary data (frequency, IPA, URL)
        if self._dictionary_enabled:
            try:
                dict_entry = await self.dictionary.get_enrichment_data(
                    validated_lemma, pos, session
                )

                if dict_entry:
                    # Use dictionary gender if available (nouns)
                    if dict_entry.gender and pos == "NOUN":
                        result.gender = dict_entry.gender

                    # Copy dictionary metadata
                    if dict_entry.definitions:
                        result.definition_de = "; ".join(dict_entry.definitions)
                    if dict_entry.synonyms:
                        result.synonyms = dict_entry.synonyms
                    result.frequency = dict_entry.frequency
                    result.ipa = dict_entry.ipa
                    result.dictionary_url = dict_entry.url

                    logger.debug(
                        f"Dictionary enrichment for '{validated_lemma}': source={dict_entry.source}"
                    )

            except Exception as e:
                logger.warning(f"Dictionary lookup failed for '{validated_lemma}': {e}")
                result.dictionary_error = f"Lookup failed: {e}"
                # Continue with LLM-only enrichment

        # Step 3: Use LLM for remaining data
        # - Translations (always from LLM)
        # - Plural forms (not in local dictionary)
        # - Verb conjugations (preterite, past_participle, auxiliary)
        # - Gender (if dictionary didn't provide it)
        try:
            llm_result = await self.enrich(result.lemma or lemma, pos)

            # Merge LLM results (prefer dictionary data where available)
            result.translations = llm_result.translations

            # Use LLM for data not available from dictionary
            if pos == "NOUN":
                if not result.gender:
                    result.gender = llm_result.gender
                result.plural = llm_result.plural

            elif pos == "VERB":
                result.preterite = llm_result.preterite
                result.past_participle = llm_result.past_participle
                result.auxiliary = llm_result.auxiliary

            elif pos == "ADP":
                # Copy preposition cases from LLM result
                # Note: cases are not currently persisted to the database (Word model
                # doesn't have a cases column). They are included in EnrichmentResult
                # for potential future use or display purposes.
                result.cases = llm_result.cases

            # Note: We intentionally do NOT use LLM lemma corrections here.
            # Local dictionary (spaCy) is the source of truth for lemma validation.
            # LLM corrections were non-deterministic and caused idempotency issues.
            # If dictionary didn't ground the lemma, we keep the original.

        except Exception as e:
            logger.warning(f"LLM enrichment failed for '{lemma}': {e}")
            result.error = str(e)

        return result

    async def detect_pos(self, word: str, context: str = "") -> tuple[str, str]:
        """
        Detect part of speech and normalize lemma for a German word using LLM.

        Note: This is a legacy/alternative method that uses an LLM call for POS
        detection. The preferred approach is to use Tokenizer.analyze_word()
        which performs POS detection locally via spaCy (faster, free, offline).

        This method is kept for cases where LLM-based detection might be preferred,
        such as when spaCy's accuracy is insufficient for certain edge cases.

        Args:
            word: The German word to analyze
            context: Optional context sentence for better accuracy

        Returns:
            Tuple of (pos, normalized_lemma)

        Raises:
            Exception if LLM call fails
        """
        context_info = f'\nContext: "{context}"' if context else ""
        prompt = f"""Analyze this German word and determine its part of speech.

Word: {word}{context_info}

Determine the most likely part of speech for this word:
- NOUN: A noun (Substantiv) - names a person, place, thing, or concept
- VERB: A verb (Verb) - describes an action or state
- ADJ: An adjective (Adjektiv) - describes a noun
- ADV: An adverb (Adverb) - modifies a verb, adjective, or other adverb
- ADP: A preposition (Präposition) - shows relationship between words

Also provide the correct base/dictionary form (lemma):
- For nouns: singular nominative form (e.g., "Arbeit" not "Arbeiten")
- For verbs: infinitive form (e.g., "arbeiten" not "arbeitete")
- For adjectives: base form (e.g., "schnell" not "schneller")"""

        data = await self._call_chat_api(
            prompt, POSDetectionResponse.model_json_schema(), "pos_detection"
        )

        pos = data.get("pos", "NOUN")
        lemma = data.get("lemma", word)
        lemma = strip_article(lemma)

        logger.info(f"POS detection for '{word}': {pos}, lemma='{lemma}'")
        return pos, lemma

    async def enrich_word(
        self,
        word: str,
        context: str = "",
        session: AsyncSession | None = None,
    ) -> tuple[str, EnrichmentResult]:
        """
        Full enrichment pipeline: detect POS, then enrich.

        This is the unified entry point for adding words, whether from CLI input
        or other sources. It automatically determines the part of speech and
        then enriches the word with grammatical information.

        Args:
            word: The German word to enrich
            context: Optional context sentence for better POS detection and enrichment
            session: Database session for dictionary caching

        Returns:
            Tuple of (detected_pos, enrichment_result)
        """
        # Step 1: Detect POS and normalize lemma
        pos, lemma = await self.detect_pos(word, context)

        # Step 2: Enrich with detected POS using the full dictionary+LLM pipeline
        result = await self.enrich_with_dictionary(lemma, pos, context, session)

        return pos, result
