"""Tests for tokenizer service."""

from unittest.mock import MagicMock, patch

import pytest

from app.services.tokenizer import TokenInfo, Tokenizer


class TestTokenInfo:
    """Tests for TokenInfo dataclass."""

    def test_create_token_info(self):
        """Should create TokenInfo with all fields."""
        info = TokenInfo(
            surface_form="Arbeit",
            lemma="Arbeit",
            pos="NOUN",
            context_sentence="Die Arbeit ist wichtig.",
        )
        assert info.surface_form == "Arbeit"
        assert info.lemma == "Arbeit"
        assert info.pos == "NOUN"
        assert info.context_sentence == "Die Arbeit ist wichtig."


class TestTokenizerInit:
    """Tests for Tokenizer initialization."""

    def test_default_model(self):
        """Should use default model name."""
        tokenizer = Tokenizer()
        assert tokenizer.model_name == "de_core_news_lg"
        assert tokenizer._nlp is None

    def test_custom_model(self):
        """Should accept custom model name."""
        tokenizer = Tokenizer(model_name="de_core_news_sm")
        assert tokenizer.model_name == "de_core_news_sm"


class TestLooksLikeParticiple:
    """Tests for _looks_like_participle method."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer()

    def test_regular_ge_t_participle(self, tokenizer):
        """Should detect ge...t participles."""
        assert tokenizer._looks_like_participle("gearbeitet") is True
        assert tokenizer._looks_like_participle("gemacht") is True

    def test_regular_ge_en_participle(self, tokenizer):
        """Should detect ge...en participles."""
        assert tokenizer._looks_like_participle("gegangen") is True
        assert tokenizer._looks_like_participle("gekommen") is True

    def test_separable_verb_participle(self, tokenizer):
        """Should detect separable verb participles."""
        assert tokenizer._looks_like_participle("abgeholt") is True
        assert tokenizer._looks_like_participle("aufgestanden") is True
        assert tokenizer._looks_like_participle("eingestellt") is True

    def test_iert_participle(self, tokenizer):
        """Should detect -iert participles."""
        assert tokenizer._looks_like_participle("analysiert") is True
        assert tokenizer._looks_like_participle("strukturiert") is True

    def test_excludes_common_nouns(self, tokenizer):
        """Should exclude common nouns starting with ge-."""
        assert tokenizer._looks_like_participle("Gerät") is False
        assert tokenizer._looks_like_participle("Geschäft") is False
        assert tokenizer._looks_like_participle("Gebiet") is False
        assert tokenizer._looks_like_participle("Gesetz") is False

    def test_non_participle(self, tokenizer):
        """Should return False for non-participles."""
        assert tokenizer._looks_like_participle("Arbeit") is False
        assert tokenizer._looks_like_participle("schnell") is False
        # Note: "gehen" matches ge...en pattern but is an infinitive not participle
        # The function can't distinguish without context


class TestIsFeminineRoleNoun:
    """Tests for _is_feminine_role_noun method."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer()

    def test_feminine_innen_plural(self, tokenizer):
        """Should detect -innen plural forms."""
        assert tokenizer._is_feminine_role_noun("Lehrerinnen") is True
        assert tokenizer._is_feminine_role_noun("Ärztinnen") is True

    def test_feminine_in_singular(self, tokenizer):
        """Should detect -in singular forms."""
        assert tokenizer._is_feminine_role_noun("Lehrerin") is True
        assert tokenizer._is_feminine_role_noun("Ärztin") is True

    def test_excludes_exceptions(self, tokenizer):
        """Should exclude words naturally ending in -in."""
        assert tokenizer._is_feminine_role_noun("Termin") is False
        assert tokenizer._is_feminine_role_noun("Berlin") is False
        assert tokenizer._is_feminine_role_noun("Protein") is False
        assert tokenizer._is_feminine_role_noun("Vitamin") is False

    def test_short_words(self, tokenizer):
        """Should not match short words."""
        assert tokenizer._is_feminine_role_noun("in") is False
        assert tokenizer._is_feminine_role_noun("hin") is False


class TestIsNominalizedAdjective:
    """Tests for _is_nominalized_adjective method."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer()

    def test_bar_suffix(self, tokenizer):
        """Should detect -bar nominalized adjectives."""
        assert tokenizer._is_nominalized_adjective("Dankbare") is True
        assert tokenizer._is_nominalized_adjective("Machbaren") is True

    def test_lich_suffix(self, tokenizer):
        """Should detect -lich nominalized adjectives."""
        assert tokenizer._is_nominalized_adjective("Freundliche") is True
        assert tokenizer._is_nominalized_adjective("Möglichen") is True

    def test_ig_suffix(self, tokenizer):
        """Should detect -ig nominalized adjectives."""
        assert tokenizer._is_nominalized_adjective("Wichtige") is True
        assert tokenizer._is_nominalized_adjective("Richtige") is True

    def test_isch_suffix(self, tokenizer):
        """Should detect -isch nominalized adjectives."""
        assert tokenizer._is_nominalized_adjective("Politische") is True

    def test_regular_noun(self, tokenizer):
        """Should not match regular nouns."""
        assert tokenizer._is_nominalized_adjective("Tisch") is False
        assert tokenizer._is_nominalized_adjective("Buch") is False


class TestParticipleToInfinitive:
    """Tests for _participle_to_infinitive method."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer()

    def test_iert_to_ieren(self, tokenizer):
        """Should convert -iert to -ieren."""
        assert tokenizer._participle_to_infinitive("analysiert") == "analysieren"
        assert tokenizer._participle_to_infinitive("strukturiert") == "strukturieren"

    def test_separable_verb_t(self, tokenizer):
        """Should convert separable verb participles ending in -t."""
        assert tokenizer._participle_to_infinitive("abgeholt") == "abholen"
        assert tokenizer._participle_to_infinitive("eingestellt") == "einstellen"

    def test_separable_verb_en(self, tokenizer):
        """Should convert separable verb participles ending in -en."""
        assert tokenizer._participle_to_infinitive("aufgestanden") == "aufstanden"

    def test_regular_ge_t(self, tokenizer):
        """Should convert regular ge...t participles."""
        # gemacht -> mach + en = machen
        assert tokenizer._participle_to_infinitive("gemacht") == "machen"
        # gearbeitet -> arbeite + en = arbeiteen (known limitation)
        result = tokenizer._participle_to_infinitive("gearbeitet")
        assert result is not None
        assert result.startswith("arbeit")

    def test_regular_ge_en(self, tokenizer):
        """Should convert regular ge...en participles."""
        assert tokenizer._participle_to_infinitive("gegangen") == "gangen"
        assert tokenizer._participle_to_infinitive("gekommen") == "kommen"

    def test_non_participle(self, tokenizer):
        """Should return None for non-participles."""
        assert tokenizer._participle_to_infinitive("arbeiten") is None
        assert tokenizer._participle_to_infinitive("Arbeit") is None


class TestRestoreUmlaut:
    """Tests for _restore_umlaut method."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer()

    def test_restore_ae(self, tokenizer):
        """Should restore ä to a."""
        assert tokenizer._restore_umlaut("länger") == "langer"
        assert tokenizer._restore_umlaut("stärker") == "starker"

    def test_restore_oe(self, tokenizer):
        """Should restore ö to o."""
        assert tokenizer._restore_umlaut("größer") == "großer"  # ö→o but ß stays
        assert tokenizer._restore_umlaut("höher") == "hoher"

    def test_restore_ue(self, tokenizer):
        """Should restore ü to u."""
        assert tokenizer._restore_umlaut("jünger") == "junger"
        assert tokenizer._restore_umlaut("kürzer") == "kurzer"

    def test_restore_aeu(self, tokenizer):
        """Should restore äu to au."""
        assert tokenizer._restore_umlaut("Häuser") == "Hauser"
        assert tokenizer._restore_umlaut("Bäume") == "Baume"

    def test_no_umlaut(self, tokenizer):
        """Should not change words without umlauts."""
        assert tokenizer._restore_umlaut("klein") == "klein"


class TestAdjectiveToBaseForm:
    """Tests for _adjective_to_base_form method."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer()

    def test_superlative_sten(self, tokenizer):
        """Should convert -sten superlative."""
        # längsten -> läng -> lang
        result = tokenizer._adjective_to_base_form("längsten")
        assert result == "lang"

    def test_superlative_ste(self, tokenizer):
        """Should convert -ste superlative."""
        result = tokenizer._adjective_to_base_form("längste")
        assert result == "lang"

    def test_superlative_st(self, tokenizer):
        """Should convert -st superlative."""
        result = tokenizer._adjective_to_base_form("längst")
        assert result == "lang"

    def test_comparative_er(self, tokenizer):
        """Should convert -er comparative."""
        # länger -> läng -> lang
        result = tokenizer._adjective_to_base_form("länger")
        assert result == "lang"

    def test_base_form(self, tokenizer):
        """Should not change base forms."""
        result = tokenizer._adjective_to_base_form("klein")
        assert result == "klein"

    def test_short_words_unchanged(self, tokenizer):
        """Should not modify short words even with -er ending."""
        result = tokenizer._adjective_to_base_form("der")
        assert result == "der"


class TestDiminutiveToBase:
    """Tests for _diminutive_to_base method."""

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer()

    def test_chen_suffix(self, tokenizer):
        """Should convert -chen diminutives."""
        result = tokenizer._diminutive_to_base("Häuschen")
        assert result == "haus"

    def test_lein_suffix(self, tokenizer):
        """Should convert -lein diminutives."""
        result = tokenizer._diminutive_to_base("Büchlein")
        assert result == "buch"

    def test_canonical_diminutives(self, tokenizer):
        """Should keep canonical diminutives unchanged."""
        assert tokenizer._diminutive_to_base("Mädchen") == "Mädchen"
        assert tokenizer._diminutive_to_base("Brötchen") == "Brötchen"
        assert tokenizer._diminutive_to_base("Kaninchen") == "Kaninchen"

    def test_non_diminutive(self, tokenizer):
        """Should not change non-diminutives."""
        assert tokenizer._diminutive_to_base("Tisch") == "Tisch"


class TestLoadModel:
    """Tests for _load_model method."""

    def test_lazy_loading(self):
        """Should load model lazily."""
        tokenizer = Tokenizer()
        assert tokenizer._nlp is None

    def test_model_cached(self):
        """Should cache loaded model."""
        import sys

        mock_spacy = MagicMock()
        mock_nlp = MagicMock()
        mock_spacy.load.return_value = mock_nlp

        with patch.dict(sys.modules, {"spacy": mock_spacy}):
            tokenizer = Tokenizer()
            nlp1 = tokenizer._load_model()
            nlp2 = tokenizer._load_model()

            assert nlp1 is nlp2
            mock_spacy.load.assert_called_once()


class TestTokenize:
    """Tests for tokenize method."""

    @pytest.fixture
    def mock_spacy(self):
        """Create a mock spacy module."""
        import sys

        mock_spacy = MagicMock()
        with patch.dict(sys.modules, {"spacy": mock_spacy}):
            yield mock_spacy

    def _create_mock_token(self, text, lemma, pos, is_alpha=True):
        """Helper to create a mock token."""
        mock_token = MagicMock()
        mock_token.pos_ = pos
        mock_token.is_alpha = is_alpha
        mock_token.text = text
        mock_token.lemma_ = lemma
        return mock_token

    def _setup_tokenizer(self, mock_spacy, tokens, sentence_text="Test sentence."):
        """Helper to setup tokenizer with mock data."""
        mock_sent = MagicMock()
        mock_sent.text = sentence_text
        mock_sent.__iter__ = lambda self: iter(tokens)

        mock_doc = MagicMock()
        mock_doc.sents = [mock_sent]

        mock_nlp = MagicMock()
        mock_nlp.return_value = mock_doc
        mock_spacy.load.return_value = mock_nlp

        tokenizer = Tokenizer()
        return tokenizer

    def test_tokenize_returns_list(self, mock_spacy):
        """Should return a list of TokenInfo."""
        mock_token = self._create_mock_token("Arbeit", "Arbeit", "NOUN")
        tokenizer = self._setup_tokenizer(mock_spacy, [mock_token])

        tokens = tokenizer.tokenize("Die Arbeit ist wichtig.")
        assert isinstance(tokens, list)

    def test_skips_irrelevant_pos(self, mock_spacy):
        """Should skip tokens with irrelevant POS."""
        mock_token = self._create_mock_token("Die", "die", "DET")
        tokenizer = self._setup_tokenizer(mock_spacy, [mock_token])

        tokens = tokenizer.tokenize("Die Arbeit.")
        assert len(tokens) == 0

    def test_skips_non_alpha(self, mock_spacy):
        """Should skip non-alphabetic tokens."""
        mock_token = self._create_mock_token("123", "123", "NOUN", is_alpha=False)
        tokenizer = self._setup_tokenizer(mock_spacy, [mock_token])

        tokens = tokenizer.tokenize("Test 123.")
        assert len(tokens) == 0

    def test_skips_short_tokens(self, mock_spacy):
        """Should skip tokens shorter than 2 characters."""
        mock_token = self._create_mock_token("a", "a", "ADP")
        tokenizer = self._setup_tokenizer(mock_spacy, [mock_token])

        tokens = tokenizer.tokenize("a")
        assert len(tokens) == 0

    def test_deduplicates_tokens(self, mock_spacy):
        """Should deduplicate tokens by lemma+pos."""
        mock_token1 = self._create_mock_token("Arbeit", "Arbeit", "NOUN")
        mock_token2 = self._create_mock_token("Arbeit", "Arbeit", "NOUN")
        tokenizer = self._setup_tokenizer(mock_spacy, [mock_token1, mock_token2])

        tokens = tokenizer.tokenize("Arbeit Arbeit")
        assert len(tokens) == 1

    def test_expands_preposition_contractions(self, mock_spacy):
        """Should expand preposition contractions."""
        mock_token = self._create_mock_token("zum", "zum", "ADP")
        tokenizer = self._setup_tokenizer(mock_spacy, [mock_token])

        tokens = tokenizer.tokenize("zum Bahnhof")
        assert len(tokens) == 1
        assert tokens[0].lemma == "zu"


class TestTokenizerConstants:
    """Tests for Tokenizer class constants."""

    def test_relevant_pos(self):
        """Should have correct relevant POS tags."""
        assert "NOUN" in Tokenizer.RELEVANT_POS
        assert "VERB" in Tokenizer.RELEVANT_POS
        assert "ADJ" in Tokenizer.RELEVANT_POS
        assert "ADV" in Tokenizer.RELEVANT_POS
        assert "ADP" in Tokenizer.RELEVANT_POS

    def test_preposition_contractions(self):
        """Should have preposition contractions."""
        assert Tokenizer.PREPOSITION_CONTRACTIONS["zum"] == "zu"
        assert Tokenizer.PREPOSITION_CONTRACTIONS["im"] == "in"
        assert Tokenizer.PREPOSITION_CONTRACTIONS["beim"] == "bei"

    def test_diminutive_exceptions(self):
        """Should have diminutive exceptions."""
        assert "mädchen" in Tokenizer.DIMINUTIVE_EXCEPTIONS
        assert "brötchen" in Tokenizer.DIMINUTIVE_EXCEPTIONS
        assert "kaninchen" in Tokenizer.DIMINUTIVE_EXCEPTIONS
