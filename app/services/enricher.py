"""LLM enrichment service using OpenAI API for German vocabulary."""

import json
import logging
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError
from pydantic import BaseModel, Field
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


# OpenAI-compatible JSON schemas for structured output
# These are manually defined to ensure compatibility with strict mode
NOUN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "lemma": {"type": "string", "description": "Correct base form (singular, nominative)"},
        "gender": {
            "type": "string",
            "enum": ["der", "die", "das"],
            "description": "Article for singular form",
        },
        "plural": {"type": "string", "description": "Plural form without article"},
        "translations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "1-3 English translations",
        },
    },
    "required": ["lemma", "gender", "plural", "translations"],
    "additionalProperties": False,
}

VERB_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "lemma": {"type": "string", "description": "Correct infinitive form"},
        "preterite": {"type": "string", "description": "3rd person singular preterite"},
        "past_participle": {"type": "string", "description": "Past participle without auxiliary"},
        "auxiliary": {"type": "string", "enum": ["haben", "sein"], "description": "Auxiliary verb"},
        "translations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "1-3 English translations",
        },
    },
    "required": ["lemma", "preterite", "past_participle", "auxiliary", "translations"],
    "additionalProperties": False,
}

WORD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "lemma": {"type": "string", "description": "Correct base/dictionary form"},
        "translations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "1-3 English translations",
        },
    },
    "required": ["lemma", "translations"],
    "additionalProperties": False,
}

VALIDATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "valid": {"type": "boolean", "description": "True if word is correct as-is"},
        "corrected_lemma": {
            "type": "string",
            "description": "Correct form (same as input if valid)",
        },
        "reason": {
            "type": "string",
            "description": "Explanation if invalid or corrected, empty if valid",
        },
    },
    "required": ["valid", "corrected_lemma", "reason"],
    "additionalProperties": False,
}

POS_DETECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pos": {
            "type": "string",
            "enum": ["NOUN", "VERB", "ADJ", "ADV", "ADP"],
            "description": "Part of speech",
        },
        "lemma": {
            "type": "string",
            "description": "Correct base/dictionary form of the word",
        },
    },
    "required": ["pos", "lemma"],
    "additionalProperties": False,
}


class EnrichmentResult(BaseModel):
    """Result of LLM enrichment for a word."""

    lemma: str | None = None
    gender: str | None = None  # der/die/das
    plural: str | None = None
    preterite: str | None = None
    past_participle: str | None = None
    auxiliary: str | None = None  # haben/sein
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

        else:  # ADJ, ADV, ADP
            return f"""Analyze this German word and provide translations.

Word: {lemma}
Part of speech: {pos}

Provide 1-3 English translations."""

    def _get_schema_for_pos(self, pos: str) -> tuple[dict[str, Any], str]:
        """Get the JSON schema and name for a part of speech."""
        if pos == "NOUN":
            return NOUN_SCHEMA, "noun_enrichment"
        elif pos == "VERB":
            return VERB_SCHEMA, "verb_enrichment"
        else:
            return WORD_SCHEMA, "word_enrichment"

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
            data = await self._call_chat_api(prompt, VALIDATION_SCHEMA, "lemma_validation")
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
            context: Context sentence for better LLM results
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
        Detect part of speech and normalize lemma for a German word.

        Uses LLM to determine the most likely POS for the word, which is more
        reliable than spaCy for isolated words without context.

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

        data = await self._call_chat_api(prompt, POS_DETECTION_SCHEMA, "pos_detection")

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
