"""LLM enrichment service using Ollama for German vocabulary."""

import json
import logging
import re
from dataclasses import dataclass

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class EnrichmentResult:
    """Result of LLM enrichment for a word."""

    gender: str | None = None  # der/die/das
    plural: str | None = None
    preterite: str | None = None
    past_participle: str | None = None
    auxiliary: str | None = None  # haben/sein
    translations: list[str] | None = None

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
- "gender": the article (der, die, or das)
- "plural": the plural form of the noun
- "translations": array of 1-3 English translations

Example response:
{{"gender": "die", "plural": "Arbeiten", "translations": ["work", "labor", "job"]}}"""

        elif pos == "VERB":
            return f"""Analyze this German verb and provide grammatical information.

Word: {lemma}
Part of speech: Verb
Context: "{context}"

Respond with ONLY a JSON object (no markdown, no explanation) with these fields:
- "preterite": the 3rd person singular preterite form
- "past_participle": the past participle (without auxiliary)
- "auxiliary": "haben" or "sein"
- "translations": array of 1-3 English translations

Example response:
{{"preterite": "arbeitete", "past_participle": "gearbeitet", "auxiliary": "haben", "translations": ["to work", "to labor"]}}"""

        else:  # ADJ, ADV, ADP
            return f"""Analyze this German word and provide English translations.

Word: {lemma}
Part of speech: {pos}
Context: "{context}"

Respond with ONLY a JSON object (no markdown, no explanation) with this field:
- "translations": array of 1-3 English translations

Example response:
{{"translations": ["quickly", "fast", "rapidly"]}}"""

    async def enrich(self, lemma: str, pos: str, context: str) -> EnrichmentResult:
        """
        Enrich a word with grammatical information via Ollama.

        Returns EnrichmentResult with available fields populated.
        """
        prompt = self._build_prompt(lemma, pos, context)

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
            logger.warning(f"Ollama timeout for '{lemma}', skipping enrichment")
            return EnrichmentResult(translations=[])
        except httpx.HTTPError as e:
            logger.warning(f"Ollama HTTP error for '{lemma}': {e}")
            return EnrichmentResult(translations=[])
        except Exception as e:
            logger.warning(f"Enrichment error for '{lemma}': {e}")
            return EnrichmentResult(translations=[])

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

    def _parse_response(self, response_text: str, pos: str) -> EnrichmentResult:
        """Parse the LLM JSON response into EnrichmentResult."""
        # Try to extract JSON from response
        text = response_text.strip()

        # Handle markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        # Find JSON object in text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            logger.warning(f"No JSON found in response: {text[:100]}")
            return EnrichmentResult(translations=[])

        json_str = text[start:end]
        json_str = self._clean_json(json_str)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}, text: {json_str[:100]}")
            return EnrichmentResult(translations=[])

        # Build result based on POS
        result = EnrichmentResult(
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
