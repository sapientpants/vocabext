"""Tests for enricher module."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

from app.services.enricher import (
    NOUN_SCHEMA,
    VERB_SCHEMA,
    WORD_SCHEMA,
    Enricher,
    EnrichmentResult,
    strip_article,
)


class TestStripArticle:
    """Tests for strip_article function."""

    def test_strips_der(self):
        """Should strip 'der ' prefix."""
        assert strip_article("der Hund") == "Hund"

    def test_strips_die(self):
        """Should strip 'die ' prefix."""
        assert strip_article("die Katze") == "Katze"

    def test_strips_das(self):
        """Should strip 'das ' prefix."""
        assert strip_article("das Haus") == "Haus"

    def test_strips_ein(self):
        """Should strip 'ein ' prefix."""
        assert strip_article("ein Buch") == "Buch"

    def test_strips_eine(self):
        """Should strip 'eine ' prefix."""
        assert strip_article("eine Frau") == "Frau"

    def test_strips_einen(self):
        """Should strip 'einen ' prefix."""
        assert strip_article("einen Mann") == "Mann"

    def test_strips_einem(self):
        """Should strip 'einem ' prefix."""
        assert strip_article("einem Kind") == "Kind"

    def test_strips_einer(self):
        """Should strip 'einer ' prefix."""
        assert strip_article("einer Sache") == "Sache"

    def test_case_insensitive(self):
        """Should be case insensitive."""
        assert strip_article("Der Hund") == "Hund"
        assert strip_article("DIE Katze") == "Katze"

    def test_no_article(self):
        """Should return unchanged if no article."""
        assert strip_article("Hund") == "Hund"

    def test_article_in_middle(self):
        """Should not strip article in middle of word."""
        assert strip_article("Kindergarten") == "Kindergarten"


class TestEnrichmentResult:
    """Tests for EnrichmentResult model."""

    def test_default_values(self):
        """Should have correct defaults."""
        result = EnrichmentResult()
        assert result.lemma is None
        assert result.gender is None
        assert result.plural is None
        assert result.preterite is None
        assert result.past_participle is None
        assert result.auxiliary is None
        assert result.translations == []
        assert result.error is None
        assert result.synonyms == []

    def test_with_noun_fields(self):
        """Should accept noun fields."""
        result = EnrichmentResult(
            lemma="Hund",
            gender="der",
            plural="Hunde",
            translations=["dog"],
        )
        assert result.lemma == "Hund"
        assert result.gender == "der"
        assert result.plural == "Hunde"
        assert result.translations == ["dog"]

    def test_with_verb_fields(self):
        """Should accept verb fields."""
        result = EnrichmentResult(
            lemma="gehen",
            preterite="ging",
            past_participle="gegangen",
            auxiliary="sein",
            translations=["to go"],
        )
        assert result.lemma == "gehen"
        assert result.preterite == "ging"
        assert result.past_participle == "gegangen"
        assert result.auxiliary == "sein"

    def test_with_dictionary_fields(self):
        """Should accept dictionary-grounded fields."""
        result = EnrichmentResult(
            lemma="Arbeit",
            definition_de="Tätigkeit",
            synonyms=["Job", "Werk"],
            frequency=5.2,
            ipa="/ˈaʁbaɪ̯t/",
            lemma_source="dwds",
            dictionary_url="https://dwds.de/wb/Arbeit",
        )
        assert result.definition_de == "Tätigkeit"
        assert result.synonyms == ["Job", "Werk"]
        assert result.frequency == 5.2
        assert result.ipa == "/ˈaʁbaɪ̯t/"
        assert result.lemma_source == "dwds"
        assert result.dictionary_url == "https://dwds.de/wb/Arbeit"


class TestEnricherInit:
    """Tests for Enricher initialization."""

    def test_default_init(self):
        """Should initialize with defaults."""
        enricher = Enricher()
        assert enricher.api_key is None
        assert enricher.model is None
        assert enricher._dictionary is None

    def test_custom_init(self):
        """Should accept custom parameters."""
        mock_dict = MagicMock()
        enricher = Enricher(
            api_key="test-key",
            model="gpt-4",
            dictionary_service=mock_dict,
        )
        assert enricher.api_key == "test-key"
        assert enricher.model == "gpt-4"
        assert enricher._dictionary is mock_dict


class TestEnricherDictionaryProperty:
    """Tests for Enricher.dictionary property."""

    def test_lazy_creates_dictionary_service(self):
        """Should lazily create DictionaryService if None."""
        enricher = Enricher()
        with patch("app.services.enricher.DictionaryService") as mock_cls:
            mock_service = MagicMock()
            mock_cls.return_value = mock_service
            result = enricher.dictionary
            assert result is mock_service
            mock_cls.assert_called_once()

    def test_returns_existing_dictionary(self):
        """Should return existing dictionary if set."""
        mock_dict = MagicMock()
        enricher = Enricher(dictionary_service=mock_dict)
        assert enricher.dictionary is mock_dict


class TestEnricherBuildPrompt:
    """Tests for Enricher._build_prompt method."""

    def test_noun_prompt(self):
        """Should build noun-specific prompt."""
        enricher = Enricher()
        prompt = enricher._build_prompt("Hund", "NOUN")
        assert "German noun" in prompt
        assert "Hund" in prompt
        assert "Gender" in prompt
        assert "Plural" in prompt

    def test_verb_prompt(self):
        """Should build verb-specific prompt."""
        enricher = Enricher()
        prompt = enricher._build_prompt("gehen", "VERB")
        assert "German verb" in prompt
        assert "gehen" in prompt
        assert "Preterite" in prompt
        assert "Past participle" in prompt
        assert "Auxiliary" in prompt

    def test_adjective_prompt(self):
        """Should build generic prompt for adjectives."""
        enricher = Enricher()
        prompt = enricher._build_prompt("schnell", "ADJ")
        assert "German word" in prompt
        assert "schnell" in prompt
        assert "ADJ" in prompt

    def test_adverb_prompt(self):
        """Should build generic prompt for adverbs."""
        enricher = Enricher()
        prompt = enricher._build_prompt("oft", "ADV")
        assert "German word" in prompt
        assert "ADV" in prompt


class TestEnricherGetSchemaForPos:
    """Tests for Enricher._get_schema_for_pos method."""

    def test_noun_schema(self):
        """Should return NOUN_SCHEMA for nouns."""
        enricher = Enricher()
        schema, name = enricher._get_schema_for_pos("NOUN")
        assert schema is NOUN_SCHEMA
        assert name == "noun_enrichment"

    def test_verb_schema(self):
        """Should return VERB_SCHEMA for verbs."""
        enricher = Enricher()
        schema, name = enricher._get_schema_for_pos("VERB")
        assert schema is VERB_SCHEMA
        assert name == "verb_enrichment"

    def test_other_schema(self):
        """Should return WORD_SCHEMA for other POS."""
        enricher = Enricher()
        for pos in ["ADJ", "ADV", "ADP"]:
            schema, name = enricher._get_schema_for_pos(pos)
            assert schema is WORD_SCHEMA
            assert name == "word_enrichment"


class TestEnricherEnrich:
    """Tests for Enricher.enrich method."""

    @pytest.mark.asyncio
    async def test_enrich_noun_success(self):
        """Should enrich noun successfully."""
        enricher = Enricher()
        mock_data = {
            "lemma": "der Hund",
            "gender": "der",
            "plural": "Hunde",
            "translations": ["dog", "hound"],
        }
        with patch.object(enricher, "_call_chat_api", new_callable=AsyncMock) as mock:
            mock.return_value = mock_data
            result = await enricher.enrich("Hund", "NOUN")

        assert result.lemma == "Hund"  # Article stripped
        assert result.gender == "der"
        assert result.plural == "Hunde"
        assert result.translations == ["dog", "hound"]
        assert result.error is None

    @pytest.mark.asyncio
    async def test_enrich_verb_success(self):
        """Should enrich verb successfully."""
        enricher = Enricher()
        mock_data = {
            "lemma": "gehen",
            "preterite": "ging",
            "past_participle": "gegangen",
            "auxiliary": "sein",
            "translations": ["to go", "to walk"],
        }
        with patch.object(enricher, "_call_chat_api", new_callable=AsyncMock) as mock:
            mock.return_value = mock_data
            result = await enricher.enrich("gehen", "VERB")

        assert result.lemma == "gehen"
        assert result.preterite == "ging"
        assert result.past_participle == "gegangen"
        assert result.auxiliary == "sein"
        assert result.translations == ["to go", "to walk"]

    @pytest.mark.asyncio
    async def test_enrich_adjective_success(self):
        """Should enrich adjective successfully."""
        enricher = Enricher()
        mock_data = {
            "lemma": "schnell",
            "translations": ["fast", "quick"],
        }
        with patch.object(enricher, "_call_chat_api", new_callable=AsyncMock) as mock:
            mock.return_value = mock_data
            result = await enricher.enrich("schnell", "ADJ")

        assert result.lemma == "schnell"
        assert result.translations == ["fast", "quick"]
        assert result.gender is None  # Not a noun

    @pytest.mark.asyncio
    async def test_enrich_timeout_error(self):
        """Should handle timeout error."""
        enricher = Enricher()
        with patch.object(enricher, "_call_chat_api", new_callable=AsyncMock) as mock:
            mock.side_effect = APITimeoutError(request=MagicMock())
            result = await enricher.enrich("Hund", "NOUN")

        assert result.error == "OpenAI request timed out"

    @pytest.mark.asyncio
    async def test_enrich_401_error(self):
        """Should handle 401 auth error."""
        enricher = Enricher()
        response = MagicMock()
        response.status_code = 401
        with patch.object(enricher, "_call_chat_api", new_callable=AsyncMock) as mock:
            mock.side_effect = APIStatusError(message="Unauthorized", response=response, body=None)
            result = await enricher.enrich("Hund", "NOUN")

        assert "Invalid OpenAI API key" in result.error

    @pytest.mark.asyncio
    async def test_enrich_404_error(self):
        """Should handle 404 model not found error."""
        enricher = Enricher(model="invalid-model")
        response = MagicMock()
        response.status_code = 404
        with patch.object(enricher, "_call_chat_api", new_callable=AsyncMock) as mock:
            mock.side_effect = APIStatusError(message="Not found", response=response, body=None)
            result = await enricher.enrich("Hund", "NOUN")

        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_enrich_429_rate_limit(self):
        """Should handle 429 rate limit error."""
        enricher = Enricher()
        response = MagicMock()
        response.status_code = 429
        with patch.object(enricher, "_call_chat_api", new_callable=AsyncMock) as mock:
            mock.side_effect = APIStatusError(message="Rate limited", response=response, body=None)
            result = await enricher.enrich("Hund", "NOUN")

        assert "rate limit exceeded" in result.error

    @pytest.mark.asyncio
    async def test_enrich_other_status_error(self):
        """Should handle other HTTP status errors."""
        enricher = Enricher()
        response = MagicMock()
        response.status_code = 500
        with patch.object(enricher, "_call_chat_api", new_callable=AsyncMock) as mock:
            mock.side_effect = APIStatusError(message="Server error", response=response, body=None)
            result = await enricher.enrich("Hund", "NOUN")

        assert "HTTP 500" in result.error

    @pytest.mark.asyncio
    async def test_enrich_connection_error(self):
        """Should handle connection error."""
        enricher = Enricher()
        with patch.object(enricher, "_call_chat_api", new_callable=AsyncMock) as mock:
            mock.side_effect = APIConnectionError(request=MagicMock())
            result = await enricher.enrich("Hund", "NOUN")

        assert "Cannot connect to OpenAI API" in result.error

    @pytest.mark.asyncio
    async def test_enrich_json_decode_error(self):
        """Should handle JSON decode error."""
        enricher = Enricher()
        with patch.object(enricher, "_call_chat_api", new_callable=AsyncMock) as mock:
            mock.side_effect = json.JSONDecodeError("Invalid", "doc", 0)
            result = await enricher.enrich("Hund", "NOUN")

        assert "Invalid JSON response" in result.error

    @pytest.mark.asyncio
    async def test_enrich_generic_error(self):
        """Should handle generic exception."""
        enricher = Enricher()
        with patch.object(enricher, "_call_chat_api", new_callable=AsyncMock) as mock:
            mock.side_effect = RuntimeError("Something went wrong")
            result = await enricher.enrich("Hund", "NOUN")

        assert "Enrichment error" in result.error


class TestEnricherBuildResult:
    """Tests for Enricher._build_result method."""

    def test_build_result_strips_article(self):
        """Should strip article from lemma."""
        enricher = Enricher()
        data = {"lemma": "der Hund", "translations": ["dog"]}
        result = enricher._build_result(data, "NOUN")
        assert result.lemma == "Hund"

    def test_build_result_noun_fields(self):
        """Should set noun-specific fields."""
        enricher = Enricher()
        data = {
            "lemma": "Katze",
            "gender": "die",
            "plural": "Katzen",
            "translations": ["cat"],
        }
        result = enricher._build_result(data, "NOUN")
        assert result.gender == "die"
        assert result.plural == "Katzen"

    def test_build_result_verb_fields(self):
        """Should set verb-specific fields."""
        enricher = Enricher()
        data = {
            "lemma": "laufen",
            "preterite": "lief",
            "past_participle": "gelaufen",
            "auxiliary": "sein",
            "translations": ["to run"],
        }
        result = enricher._build_result(data, "VERB")
        assert result.preterite == "lief"
        assert result.past_participle == "gelaufen"
        assert result.auxiliary == "sein"

    def test_build_result_no_lemma(self):
        """Should handle missing lemma."""
        enricher = Enricher()
        data = {"translations": ["word"]}
        result = enricher._build_result(data, "ADJ")
        assert result.lemma is None


class TestEnricherValidateLemma:
    """Tests for Enricher.validate_lemma method."""

    @pytest.mark.asyncio
    async def test_validate_lemma_success(self):
        """Should return validation result on success."""
        enricher = Enricher()
        mock_data = {
            "valid": True,
            "corrected_lemma": "Hund",
            "reason": "",
        }
        with patch.object(enricher, "_call_chat_api", new_callable=AsyncMock) as mock:
            mock.return_value = mock_data
            result = await enricher.validate_lemma("Hund", "NOUN", "context")

        assert result["valid"] is True
        assert result["corrected_lemma"] == "Hund"
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_validate_lemma_with_correction(self):
        """Should return corrected lemma."""
        enricher = Enricher()
        mock_data = {
            "valid": False,
            "corrected_lemma": "der Hund",  # Has article
            "reason": "Added article",
        }
        with patch.object(enricher, "_call_chat_api", new_callable=AsyncMock) as mock:
            mock.return_value = mock_data
            result = await enricher.validate_lemma("Hunde", "NOUN", "context")

        assert result["corrected_lemma"] == "Hund"  # Article stripped

    @pytest.mark.asyncio
    async def test_validate_lemma_pos_info_noun(self):
        """Should include noun-specific info in prompt."""
        enricher = Enricher()
        mock_data = {"valid": True, "corrected_lemma": "Arbeit", "reason": ""}
        with patch.object(enricher, "_call_chat_api", new_callable=AsyncMock) as mock:
            mock.return_value = mock_data
            await enricher.validate_lemma("Arbeit", "NOUN", "context")
            prompt = mock.call_args[0][0]
            assert "noun" in prompt
            assert "singular nominative" in prompt

    @pytest.mark.asyncio
    async def test_validate_lemma_pos_info_verb(self):
        """Should include verb-specific info in prompt."""
        enricher = Enricher()
        mock_data = {"valid": True, "corrected_lemma": "arbeiten", "reason": ""}
        with patch.object(enricher, "_call_chat_api", new_callable=AsyncMock) as mock:
            mock.return_value = mock_data
            await enricher.validate_lemma("arbeiten", "VERB", "context")
            prompt = mock.call_args[0][0]
            assert "verb" in prompt
            assert "infinitive" in prompt

    @pytest.mark.asyncio
    async def test_validate_lemma_pos_info_adj(self):
        """Should include adjective-specific info in prompt."""
        enricher = Enricher()
        mock_data = {"valid": True, "corrected_lemma": "schnell", "reason": ""}
        with patch.object(enricher, "_call_chat_api", new_callable=AsyncMock) as mock:
            mock.return_value = mock_data
            await enricher.validate_lemma("schnell", "ADJ", "context")
            prompt = mock.call_args[0][0]
            assert "adjective" in prompt

    @pytest.mark.asyncio
    async def test_validate_lemma_pos_info_adv(self):
        """Should include adverb-specific info in prompt."""
        enricher = Enricher()
        mock_data = {"valid": True, "corrected_lemma": "oft", "reason": ""}
        with patch.object(enricher, "_call_chat_api", new_callable=AsyncMock) as mock:
            mock.return_value = mock_data
            await enricher.validate_lemma("oft", "ADV", "context")
            prompt = mock.call_args[0][0]
            assert "adverb" in prompt

    @pytest.mark.asyncio
    async def test_validate_lemma_pos_info_adp(self):
        """Should include preposition-specific info in prompt."""
        enricher = Enricher()
        mock_data = {"valid": True, "corrected_lemma": "auf", "reason": ""}
        with patch.object(enricher, "_call_chat_api", new_callable=AsyncMock) as mock:
            mock.return_value = mock_data
            await enricher.validate_lemma("auf", "ADP", "context")
            prompt = mock.call_args[0][0]
            assert "preposition" in prompt

    @pytest.mark.asyncio
    async def test_validate_lemma_unknown_pos(self):
        """Should handle unknown POS gracefully."""
        enricher = Enricher()
        mock_data = {"valid": True, "corrected_lemma": "foo", "reason": ""}
        with patch.object(enricher, "_call_chat_api", new_callable=AsyncMock) as mock:
            mock.return_value = mock_data
            await enricher.validate_lemma("foo", "UNKNOWN", "context")
            prompt = mock.call_args[0][0]
            assert "word" in prompt

    @pytest.mark.asyncio
    async def test_validate_lemma_401_error(self):
        """Should handle 401 auth error."""
        enricher = Enricher()
        response = MagicMock()
        response.status_code = 401
        with patch.object(enricher, "_call_chat_api", new_callable=AsyncMock) as mock:
            mock.side_effect = APIStatusError(message="Unauthorized", response=response, body=None)
            result = await enricher.validate_lemma("Hund", "NOUN", "context")

        assert "Invalid OpenAI API key" in result["error"]
        assert result["valid"] is True  # Default to valid on error
        assert result["corrected_lemma"] == "Hund"

    @pytest.mark.asyncio
    async def test_validate_lemma_404_error(self):
        """Should handle 404 model not found error."""
        enricher = Enricher(model="bad-model")
        response = MagicMock()
        response.status_code = 404
        with patch.object(enricher, "_call_chat_api", new_callable=AsyncMock) as mock:
            mock.side_effect = APIStatusError(message="Not found", response=response, body=None)
            result = await enricher.validate_lemma("Hund", "NOUN", "context")

        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_validate_lemma_other_status_error(self):
        """Should handle other HTTP errors."""
        enricher = Enricher()
        response = MagicMock()
        response.status_code = 503
        with patch.object(enricher, "_call_chat_api", new_callable=AsyncMock) as mock:
            mock.side_effect = APIStatusError(message="Unavailable", response=response, body=None)
            result = await enricher.validate_lemma("Hund", "NOUN", "context")

        assert "HTTP 503" in result["error"]

    @pytest.mark.asyncio
    async def test_validate_lemma_connection_error(self):
        """Should handle connection error."""
        enricher = Enricher()
        with patch.object(enricher, "_call_chat_api", new_callable=AsyncMock) as mock:
            mock.side_effect = APIConnectionError(request=MagicMock())
            result = await enricher.validate_lemma("Hund", "NOUN", "context")

        assert "Cannot connect" in result["error"]

    @pytest.mark.asyncio
    async def test_validate_lemma_generic_error(self):
        """Should handle generic exception."""
        enricher = Enricher()
        with patch.object(enricher, "_call_chat_api", new_callable=AsyncMock) as mock:
            mock.side_effect = RuntimeError("Something broke")
            result = await enricher.validate_lemma("Hund", "NOUN", "context")

        assert "Something broke" in result["error"]


class TestEnricherValidateAndEnrich:
    """Tests for Enricher.validate_and_enrich method."""

    @pytest.mark.asyncio
    async def test_validate_and_enrich_success(self):
        """Should validate and enrich successfully."""
        enricher = Enricher()
        with patch.object(enricher, "validate_lemma", new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = {
                "valid": True,
                "corrected_lemma": "Hund",
                "error": None,
            }
            with patch.object(enricher, "enrich", new_callable=AsyncMock) as mock_enrich:
                mock_enrich.return_value = EnrichmentResult(
                    lemma="Hund",
                    gender="der",
                    translations=["dog"],
                )
                result = await enricher.validate_and_enrich("Hund", "NOUN", "context")

        assert result.lemma == "Hund"
        assert result.gender == "der"
        mock_enrich.assert_called_once_with("Hund", "NOUN")

    @pytest.mark.asyncio
    async def test_validate_and_enrich_with_correction(self):
        """Should use corrected lemma for enrichment."""
        enricher = Enricher()
        with patch.object(enricher, "validate_lemma", new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = {
                "valid": False,
                "corrected_lemma": "Hund",
                "error": None,
            }
            with patch.object(enricher, "enrich", new_callable=AsyncMock) as mock_enrich:
                mock_enrich.return_value = EnrichmentResult(translations=["dog"])
                result = await enricher.validate_and_enrich("Hunde", "NOUN", "context")

        assert result.lemma == "Hund"
        mock_enrich.assert_called_once_with("Hund", "NOUN")

    @pytest.mark.asyncio
    async def test_validate_and_enrich_validation_error(self):
        """Should return error if validation fails."""
        enricher = Enricher()
        with patch.object(enricher, "validate_lemma", new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = {
                "valid": True,
                "corrected_lemma": "Hund",
                "error": "API error",
            }
            result = await enricher.validate_and_enrich("Hund", "NOUN", "context")

        assert result.error == "API error"
        assert result.lemma == "Hund"


class TestEnricherEnrichWithDictionary:
    """Tests for Enricher.enrich_with_dictionary method."""

    @pytest.mark.asyncio
    async def test_enrich_with_dictionary_full_flow(self):
        """Should combine dictionary and LLM data."""
        mock_dict = MagicMock()
        mock_dict.validate_and_ground_lemma = AsyncMock(return_value=("Hund", True, "spacy"))
        mock_dict.get_enrichment_data = AsyncMock(
            return_value=MagicMock(
                gender="der",
                definitions=["Haustier"],
                synonyms=["Köter"],
                frequency=5.0,
                ipa="/hʊnt/",
                url="https://dwds.de/wb/Hund",
                source="dwds",
            )
        )

        enricher = Enricher(dictionary_service=mock_dict)
        enricher._dictionary_enabled = True

        with patch.object(enricher, "enrich", new_callable=AsyncMock) as mock_enrich:
            mock_enrich.return_value = EnrichmentResult(
                lemma="Hund",
                gender="der",
                plural="Hunde",
                translations=["dog"],
            )
            result = await enricher.enrich_with_dictionary("Hund", "NOUN", "context")

        assert result.lemma == "Hund"
        assert result.lemma_source == "spacy"
        assert result.gender == "der"  # From dictionary
        assert result.plural == "Hunde"  # From LLM
        assert result.translations == ["dog"]
        assert result.definition_de == "Haustier"
        assert result.synonyms == ["Köter"]
        assert result.frequency == 5.0

    @pytest.mark.asyncio
    async def test_enrich_with_dictionary_validation_failure(self):
        """Should handle dictionary validation failure."""
        mock_dict = MagicMock()
        mock_dict.validate_and_ground_lemma = AsyncMock(
            side_effect=RuntimeError("Dictionary error")
        )
        mock_dict.get_enrichment_data = AsyncMock(return_value=None)

        enricher = Enricher(dictionary_service=mock_dict)
        enricher._dictionary_enabled = True

        with patch.object(enricher, "enrich", new_callable=AsyncMock) as mock_enrich:
            mock_enrich.return_value = EnrichmentResult(
                lemma="Hund",
                translations=["dog"],
            )
            result = await enricher.enrich_with_dictionary("Hund", "NOUN", "context")

        assert "Validation failed" in result.dictionary_error

    @pytest.mark.asyncio
    async def test_enrich_with_dictionary_lookup_failure(self):
        """Should handle dictionary lookup failure."""
        mock_dict = MagicMock()
        mock_dict.validate_and_ground_lemma = AsyncMock(return_value=("Hund", True, "spacy"))
        mock_dict.get_enrichment_data = AsyncMock(side_effect=RuntimeError("Lookup error"))

        enricher = Enricher(dictionary_service=mock_dict)
        enricher._dictionary_enabled = True

        with patch.object(enricher, "enrich", new_callable=AsyncMock) as mock_enrich:
            mock_enrich.return_value = EnrichmentResult(
                lemma="Hund",
                translations=["dog"],
            )
            result = await enricher.enrich_with_dictionary("Hund", "NOUN", "context")

        assert "Lookup failed" in result.dictionary_error

    @pytest.mark.asyncio
    async def test_enrich_with_dictionary_llm_failure(self):
        """Should handle LLM enrichment failure."""
        mock_dict = MagicMock()
        mock_dict.validate_and_ground_lemma = AsyncMock(return_value=("Hund", True, "spacy"))
        mock_dict.get_enrichment_data = AsyncMock(return_value=None)

        enricher = Enricher(dictionary_service=mock_dict)
        enricher._dictionary_enabled = True

        with patch.object(enricher, "enrich", new_callable=AsyncMock) as mock_enrich:
            mock_enrich.side_effect = RuntimeError("LLM error")
            result = await enricher.enrich_with_dictionary("Hund", "NOUN", "context")

        assert "LLM error" in result.error

    @pytest.mark.asyncio
    async def test_enrich_with_dictionary_disabled(self):
        """Should skip dictionary when disabled."""
        enricher = Enricher()
        enricher._dictionary_enabled = False

        with patch.object(enricher, "enrich", new_callable=AsyncMock) as mock_enrich:
            mock_enrich.return_value = EnrichmentResult(
                lemma="Hund",
                gender="der",
                plural="Hunde",
                translations=["dog"],
            )
            result = await enricher.enrich_with_dictionary("Hund", "NOUN", "context")

        assert result.lemma == "Hund"
        assert result.lemma_source == "spacy"  # Default source

    @pytest.mark.asyncio
    async def test_enrich_with_dictionary_verb(self):
        """Should handle verb enrichment correctly."""
        mock_dict = MagicMock()
        mock_dict.validate_and_ground_lemma = AsyncMock(return_value=("gehen", True, "spacy"))
        mock_dict.get_enrichment_data = AsyncMock(return_value=None)

        enricher = Enricher(dictionary_service=mock_dict)
        enricher._dictionary_enabled = True

        with patch.object(enricher, "enrich", new_callable=AsyncMock) as mock_enrich:
            mock_enrich.return_value = EnrichmentResult(
                lemma="gehen",
                preterite="ging",
                past_participle="gegangen",
                auxiliary="sein",
                translations=["to go"],
            )
            result = await enricher.enrich_with_dictionary("gehen", "VERB", "context")

        assert result.preterite == "ging"
        assert result.past_participle == "gegangen"
        assert result.auxiliary == "sein"

    @pytest.mark.asyncio
    async def test_enrich_with_dictionary_noun_gender_from_llm(self):
        """Should use LLM gender if dictionary doesn't provide it."""
        mock_dict = MagicMock()
        mock_dict.validate_and_ground_lemma = AsyncMock(return_value=("Hund", True, "spacy"))
        mock_dict.get_enrichment_data = AsyncMock(
            return_value=MagicMock(
                gender=None,  # No gender from dictionary
                definitions=None,
                synonyms=None,
                frequency=None,
                ipa=None,
                url=None,
                source="spacy",
            )
        )

        enricher = Enricher(dictionary_service=mock_dict)
        enricher._dictionary_enabled = True

        with patch.object(enricher, "enrich", new_callable=AsyncMock) as mock_enrich:
            mock_enrich.return_value = EnrichmentResult(
                lemma="Hund",
                gender="der",  # Gender from LLM
                plural="Hunde",
                translations=["dog"],
            )
            result = await enricher.enrich_with_dictionary("Hund", "NOUN", "context")

        assert result.gender == "der"  # From LLM fallback

    @pytest.mark.asyncio
    async def test_enrich_with_dictionary_lemma_correction(self):
        """Should log when lemma is corrected."""
        mock_dict = MagicMock()
        mock_dict.validate_and_ground_lemma = AsyncMock(
            return_value=("Hund", True, "spacy")  # Corrected from "hund"
        )
        mock_dict.get_enrichment_data = AsyncMock(return_value=None)

        enricher = Enricher(dictionary_service=mock_dict)
        enricher._dictionary_enabled = True

        with patch.object(enricher, "enrich", new_callable=AsyncMock) as mock_enrich:
            mock_enrich.return_value = EnrichmentResult(translations=["dog"])
            result = await enricher.enrich_with_dictionary("hund", "NOUN", "context")

        assert result.lemma == "Hund"

    @pytest.mark.asyncio
    async def test_enrich_with_dictionary_not_grounded(self):
        """Should continue with LLM when dictionary doesn't ground lemma."""
        mock_dict = MagicMock()
        mock_dict.validate_and_ground_lemma = AsyncMock(
            return_value=("Hund", False, None)  # Not grounded
        )
        mock_dict.get_enrichment_data = AsyncMock(return_value=None)

        enricher = Enricher(dictionary_service=mock_dict)
        enricher._dictionary_enabled = True

        with patch.object(enricher, "enrich", new_callable=AsyncMock) as mock_enrich:
            mock_enrich.return_value = EnrichmentResult(
                lemma="Hund",
                gender="der",
                translations=["dog"],
            )
            result = await enricher.enrich_with_dictionary("Hund", "NOUN", "context")

        # Should still have LLM data
        assert result.translations == ["dog"]
        assert result.gender == "der"

    @pytest.mark.asyncio
    async def test_enrich_with_dictionary_adjective(self):
        """Should handle adjective enrichment (no special fields)."""
        mock_dict = MagicMock()
        mock_dict.validate_and_ground_lemma = AsyncMock(return_value=("schnell", True, "spacy"))
        mock_dict.get_enrichment_data = AsyncMock(return_value=None)

        enricher = Enricher(dictionary_service=mock_dict)
        enricher._dictionary_enabled = True

        with patch.object(enricher, "enrich", new_callable=AsyncMock) as mock_enrich:
            mock_enrich.return_value = EnrichmentResult(
                lemma="schnell",
                translations=["fast", "quick"],
            )
            result = await enricher.enrich_with_dictionary("schnell", "ADJ", "context")

        # ADJ has no special fields like gender/plural (NOUN) or preterite (VERB)
        assert result.lemma == "schnell"
        assert result.translations == ["fast", "quick"]
        assert result.gender is None
        assert result.preterite is None


class TestEnricherDetectPos:
    """Tests for Enricher.detect_pos method."""

    @pytest.mark.asyncio
    async def test_detect_pos_noun(self):
        """Should detect noun POS."""
        enricher = Enricher()
        with patch.object(enricher, "_call_chat_api", new_callable=AsyncMock) as mock:
            mock.return_value = {"pos": "NOUN", "lemma": "der Hund"}
            pos, lemma = await enricher.detect_pos("Hund")

        assert pos == "NOUN"
        assert lemma == "Hund"  # Article stripped

    @pytest.mark.asyncio
    async def test_detect_pos_verb(self):
        """Should detect verb POS."""
        enricher = Enricher()
        with patch.object(enricher, "_call_chat_api", new_callable=AsyncMock) as mock:
            mock.return_value = {"pos": "VERB", "lemma": "gehen"}
            pos, lemma = await enricher.detect_pos("gehe")

        assert pos == "VERB"
        assert lemma == "gehen"

    @pytest.mark.asyncio
    async def test_detect_pos_with_context(self):
        """Should include context in prompt."""
        enricher = Enricher()
        with patch.object(enricher, "_call_chat_api", new_callable=AsyncMock) as mock:
            mock.return_value = {"pos": "VERB", "lemma": "laufen"}
            await enricher.detect_pos("laufen", "Ich laufe schnell.")
            prompt = mock.call_args[0][0]
            assert "Ich laufe schnell." in prompt

    @pytest.mark.asyncio
    async def test_detect_pos_default_noun(self):
        """Should default to NOUN if missing."""
        enricher = Enricher()
        with patch.object(enricher, "_call_chat_api", new_callable=AsyncMock) as mock:
            mock.return_value = {"lemma": "Ding"}  # No pos key
            pos, lemma = await enricher.detect_pos("Ding")

        assert pos == "NOUN"

    @pytest.mark.asyncio
    async def test_detect_pos_default_lemma(self):
        """Should use original word if lemma missing."""
        enricher = Enricher()
        with patch.object(enricher, "_call_chat_api", new_callable=AsyncMock) as mock:
            mock.return_value = {"pos": "ADJ"}  # No lemma key
            pos, lemma = await enricher.detect_pos("schnell")

        assert lemma == "schnell"


class TestEnricherEnrichWord:
    """Tests for Enricher.enrich_word method."""

    @pytest.mark.asyncio
    async def test_enrich_word_full_pipeline(self):
        """Should run full pipeline: detect POS, then enrich."""
        enricher = Enricher()
        with patch.object(enricher, "detect_pos", new_callable=AsyncMock) as mock_detect:
            mock_detect.return_value = ("NOUN", "Hund")
            with patch.object(
                enricher, "enrich_with_dictionary", new_callable=AsyncMock
            ) as mock_enrich:
                mock_enrich.return_value = EnrichmentResult(
                    lemma="Hund",
                    gender="der",
                    translations=["dog"],
                )
                pos, result = await enricher.enrich_word("Hund", "Der Hund bellt.")

        assert pos == "NOUN"
        assert result.lemma == "Hund"
        mock_detect.assert_called_once_with("Hund", "Der Hund bellt.")
        mock_enrich.assert_called_once_with("Hund", "NOUN", "Der Hund bellt.", None)

    @pytest.mark.asyncio
    async def test_enrich_word_with_session(self):
        """Should pass session to enrich_with_dictionary."""
        enricher = Enricher()
        mock_session = MagicMock()
        with patch.object(enricher, "detect_pos", new_callable=AsyncMock) as mock_detect:
            mock_detect.return_value = ("VERB", "gehen")
            with patch.object(
                enricher, "enrich_with_dictionary", new_callable=AsyncMock
            ) as mock_enrich:
                mock_enrich.return_value = EnrichmentResult(lemma="gehen")
                await enricher.enrich_word("gehe", session=mock_session)

        mock_enrich.assert_called_once_with("gehen", "VERB", "", mock_session)
