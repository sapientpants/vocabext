"""LLM enrichment service using Ollama for German vocabulary."""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# Global semaphore to ensure only one LLM request runs at a time
# This prevents overwhelming the LLM server and ensures predictable resource usage
_llm_semaphore = asyncio.Semaphore(1)


@dataclass
class EnrichmentResult:
    """Result of LLM enrichment for a word."""

    lemma: str | None = None  # Corrected lemma/base form
    gender: str | None = None  # der/die/das
    plural: str | None = None
    preterite: str | None = None
    past_participle: str | None = None
    auxiliary: str | None = None  # haben/sein
    translations: list[str] | None = None
    error: str | None = None  # Error message if enrichment failed

    def __post_init__(self) -> None:
        if self.translations is None:
            self.translations = []


class Enricher:
    """Enrich vocabulary with grammatical information using Ollama."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.ollama_model

    def _build_prompt(self, lemma: str, pos: str, context: str) -> str:
        """Build the prompt for LLM enrichment."""
        if pos == "NOUN":
            return f"""Analyze this German noun and provide grammatical information.

Word: {lemma}
Part of speech: Noun
Context: "{context}"

Respond with ONLY a JSON object (no markdown, no explanation) with these fields:
- "lemma": the correct base form/dictionary form of the noun (singular, nominative)
- "gender": the article for the SINGULAR form only (der, die, or das) - just ONE article
- "plural": the plural form of the noun (e.g., "Arbeiten" without article)
- "translations": array of 1-3 English translations

Example response:
{{"lemma": "Arbeit", "gender": "die", "plural": "Arbeiten", "translations": ["work", "labor", "job"]}}"""

        elif pos == "VERB":
            return f"""Analyze this German verb and provide grammatical information.

Word: {lemma}
Part of speech: Verb
Context: "{context}"

Respond with ONLY a JSON object (no markdown, no explanation) with these fields:
- "lemma": the correct infinitive form of the verb
- "preterite": the 3rd person singular preterite form
- "past_participle": the past participle (without auxiliary)
- "auxiliary": "haben" or "sein"
- "translations": array of 1-3 English translations

Example response:
{{"lemma": "arbeiten", "preterite": "arbeitete", "past_participle": "gearbeitet", "auxiliary": "haben", "translations": ["to work", "to labor"]}}"""

        else:  # ADJ, ADV, ADP
            return f"""Analyze this German word and provide English translations.

Word: {lemma}
Part of speech: {pos}
Context: "{context}"

Respond with ONLY a JSON object (no markdown, no explanation) with these fields:
- "lemma": the correct base form/dictionary form of the word
- "translations": array of 1-3 English translations

Example response:
{{"lemma": "schnell", "translations": ["quickly", "fast", "rapidly"]}}"""

    async def enrich(self, lemma: str, pos: str, context: str) -> EnrichmentResult:
        """
        Enrich a word with grammatical information via Ollama.

        Returns EnrichmentResult with available fields populated.

        Uses a semaphore to ensure only one LLM request runs at a time,
        preventing resource contention and ensuring predictable behavior.
        """
        prompt = self._build_prompt(lemma, pos, context)

        # Acquire semaphore to ensure only one LLM request at a time
        async with _llm_semaphore:
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(
                        f"{self.base_url}/api/generate",
                        json={
                            "model": self.model,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0.1,  # Low temperature for consistent output
                            },
                        },
                    )
                    response.raise_for_status()

                result_text = response.json().get("response", "")
                return self._parse_response(result_text, pos)

            except httpx.TimeoutException:
                error_msg = f"Ollama request timed out after 120s (URL: {self.base_url})"
                logger.warning(f"Ollama timeout for '{lemma}', skipping enrichment")
                return EnrichmentResult(translations=[], error=error_msg)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    error_msg = f"Model '{self.model}' not found at {self.base_url}. Check OLLAMA_MODEL setting."
                else:
                    error_msg = (
                        f"Ollama returned HTTP {e.response.status_code} (URL: {self.base_url})"
                    )
                logger.error(f"Ollama HTTP error for '{lemma}': {e}")
                return EnrichmentResult(translations=[], error=error_msg)
            except httpx.ConnectError:
                error_msg = f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?"
                logger.error(f"Ollama connection error for '{lemma}': {error_msg}")
                return EnrichmentResult(translations=[], error=error_msg)
            except httpx.HTTPError as e:
                error_msg = f"Ollama request failed: {e}"
                logger.error(f"Ollama HTTP error for '{lemma}': {e}")
                return EnrichmentResult(translations=[], error=error_msg)
            except Exception as e:
                error_msg = f"Enrichment error: {e}"
                logger.error(f"Enrichment error for '{lemma}': {e}")
                return EnrichmentResult(translations=[], error=error_msg)

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

Respond with ONLY a JSON object (no markdown, no explanation):
- "valid": true ONLY if the word is 100% correct as-is
- "corrected_lemma": the correct complete base/dictionary form (same as input if valid, null if not a real word)
- "reason": brief explanation if invalid or needs correction

JSON format:
{{"valid": <boolean>, "corrected_lemma": "<word or null>", "reason": "<explanation or omit if valid>"}}"""

        async with _llm_semaphore:
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(
                        f"{self.base_url}/api/generate",
                        json={
                            "model": self.model,
                            "prompt": prompt,
                            "stream": False,
                            "options": {"temperature": 0.1},
                        },
                    )
                    response.raise_for_status()

                result_text = response.json().get("response", "")
                json_str = self._extract_json_object(result_text)

                if not json_str:
                    logger.warning(f"No JSON in lemma validation response: {result_text[:100]}")
                    return {"valid": True, "corrected_lemma": lemma, "error": None}

                json_str = self._clean_json(json_str)
                data = json.loads(json_str)

                return {
                    "valid": data.get("valid", True),
                    "corrected_lemma": data.get("corrected_lemma") or lemma,
                    "reason": data.get("reason"),
                    "error": None,
                }

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    error_msg = f"Model '{self.model}' not found at {self.base_url}. Check OLLAMA_MODEL setting."
                else:
                    error_msg = f"Ollama returned HTTP {e.response.status_code}"
                logger.error(f"Lemma validation HTTP error: {e}")
                return {"valid": True, "corrected_lemma": lemma, "error": error_msg}
            except httpx.ConnectError:
                error_msg = f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?"
                logger.error(f"Lemma validation connection error: {error_msg}")
                return {"valid": True, "corrected_lemma": lemma, "error": error_msg}
            except Exception as e:
                logger.error(f"Lemma validation error for '{lemma}': {e}")
                return {"valid": True, "corrected_lemma": lemma, "error": str(e)}

    def _clean_json(self, json_str: str) -> str:
        """Clean LLM output to get valid JSON."""
        # Remove C-style comments
        json_str = re.sub(r"/\*.*?\*/", "", json_str, flags=re.DOTALL)
        json_str = re.sub(r"//.*?$", "", json_str, flags=re.MULTILINE)

        # Remove trailing content after string values (e.g., "value" (extra stuff))
        # Match: "key": "value" followed by parenthetical or extra text before comma/brace
        json_str = re.sub(r'(":\s*"[^"]*")\s*\([^)]*\)', r"\1", json_str)

        # Remove trailing commas before closing braces
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)

        return json_str.strip()

    def _extract_json_object(self, text: str) -> str | None:
        """Extract a JSON object from text by matching balanced braces."""
        start = text.find("{")
        if start == -1:
            return None

        # Find matching closing brace by counting
        depth = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

        return None

    def _parse_response(self, response_text: str, pos: str) -> EnrichmentResult:
        """Parse the LLM JSON response into EnrichmentResult."""
        # Try to extract JSON from response
        text = response_text.strip()

        # Handle markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        # Find JSON object with balanced braces
        json_str = self._extract_json_object(text)
        if not json_str:
            logger.warning(f"No JSON found in response: {text[:100]}")
            return EnrichmentResult(translations=[])

        json_str = self._clean_json(json_str)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}, text: {json_str[:150]}")
            return EnrichmentResult(translations=[])

        # Build result based on POS
        result = EnrichmentResult(
            lemma=self._normalize_to_string(data.get("lemma")),
            translations=data.get("translations", []),
        )

        if pos == "NOUN":
            result.gender = self._normalize_to_string(data.get("gender"))
            result.plural = self._normalize_to_string(data.get("plural"))
        elif pos == "VERB":
            result.preterite = self._normalize_to_string(data.get("preterite"))
            result.past_participle = self._normalize_to_string(data.get("past_participle"))
            result.auxiliary = self._normalize_to_string(data.get("auxiliary"))

        return result

    def _normalize_to_string(self, value: str | list[str] | None) -> str | None:
        """Normalize a value to a string, handling lists from LLM responses."""
        if value is None:
            return None
        if isinstance(value, list):
            # LLM sometimes returns lists - take the first value (singular form)
            return str(value[0]) if value else None
        return str(value)
