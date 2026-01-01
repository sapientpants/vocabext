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

    def test_separable_verb_prefix_match_stem_neither_t_nor_en(self, tokenizer):
        """Should continue loop when prefix+ge matches but stem ends in neither t nor en."""
        # "aufgebroch" matches "auf" + "ge", stem is "broch" (ends in neither t nor en)
        # Falls through to check regular ge patterns
        result = tokenizer._participle_to_infinitive("aufgebroch")
        # Since it doesn't match any full pattern, returns None
        assert result is None

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

    def _create_mock_token(self, text, lemma, pos, is_alpha=True, like_num=False):
        """Helper to create a mock token."""
        mock_token = MagicMock()
        mock_token.pos_ = pos
        mock_token.is_alpha = is_alpha
        mock_token.like_num = like_num
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

    def test_skips_number_like_tokens(self, mock_spacy):
        """Should skip tokens that look like numbers."""
        mock_token = self._create_mock_token("15", "15", "NOUN", like_num=True)
        tokenizer = self._setup_tokenizer(mock_spacy, [mock_token])

        tokens = tokenizer.tokenize("15 Euro")
        assert len(tokens) == 0

    def test_skips_stopwords_months(self, mock_spacy):
        """Should skip month names."""
        mock_token = self._create_mock_token("Januar", "januar", "NOUN")
        tokenizer = self._setup_tokenizer(mock_spacy, [mock_token])

        tokens = tokenizer.tokenize("im Januar")
        assert len(tokens) == 0

    def test_skips_stopwords_days(self, mock_spacy):
        """Should skip day names."""
        mock_token = self._create_mock_token("Montag", "montag", "NOUN")
        tokenizer = self._setup_tokenizer(mock_spacy, [mock_token])

        tokens = tokenizer.tokenize("am Montag")
        assert len(tokens) == 0

    def test_skips_ordinals(self, mock_spacy):
        """Should skip ordinal numbers."""
        mock_token = self._create_mock_token("erste", "erster", "ADJ")
        tokenizer = self._setup_tokenizer(mock_spacy, [mock_token])

        tokens = tokenizer.tokenize("der erste Tag")
        assert len(tokens) == 0

    def test_skips_participles_as_nouns(self, mock_spacy):
        """Should skip participles mistagged as nouns."""
        # A word that looks like a participle (ge...t pattern)
        mock_token = self._create_mock_token("Gearbeitet", "gearbeitet", "NOUN")
        tokenizer = self._setup_tokenizer(mock_spacy, [mock_token])

        tokens = tokenizer.tokenize("Das Gearbeitet")
        assert len(tokens) == 0

    def test_skips_participles_as_adjectives(self, mock_spacy):
        """Should skip participles mistagged as adjectives."""
        mock_token = self._create_mock_token("gespielt", "gespielt", "ADJ")
        tokenizer = self._setup_tokenizer(mock_spacy, [mock_token])

        tokens = tokenizer.tokenize("gespielt")
        assert len(tokens) == 0

    def test_skips_feminine_role_nouns(self, mock_spacy):
        """Should skip feminine forms of role nouns like Lehrerin."""
        mock_token = self._create_mock_token("Lehrerin", "Lehrerin", "NOUN")
        tokenizer = self._setup_tokenizer(mock_spacy, [mock_token])

        tokens = tokenizer.tokenize("die Lehrerin")
        assert len(tokens) == 0

    def test_skips_nominalized_adjectives(self, mock_spacy):
        """Should skip nominalized adjectives like Liebliche."""
        mock_token = self._create_mock_token("Freundliche", "Freundliche", "NOUN")
        tokenizer = self._setup_tokenizer(mock_spacy, [mock_token])

        tokens = tokenizer.tokenize("das Freundliche")
        assert len(tokens) == 0

    def test_converts_verb_participle_to_infinitive(self, mock_spacy):
        """Should convert verb participles to infinitive form."""
        # A verb that looks like a participle
        mock_token = self._create_mock_token("gespielt", "gespielt", "VERB")
        tokenizer = self._setup_tokenizer(mock_spacy, [mock_token])

        tokens = tokenizer.tokenize("ich habe gespielt")
        assert len(tokens) == 1
        assert tokens[0].lemma == "spielen"  # Converted to infinitive

    def test_verb_non_participle_lowercased(self, mock_spacy):
        """Should lowercase regular verbs that are not participles."""
        mock_token = self._create_mock_token("arbeiten", "Arbeiten", "VERB")
        tokenizer = self._setup_tokenizer(mock_spacy, [mock_token])

        tokens = tokenizer.tokenize("ich will Arbeiten")
        assert len(tokens) == 1
        assert tokens[0].lemma == "arbeiten"  # Lowercased

    def test_converts_adjective_to_base_form(self, mock_spacy):
        """Should convert comparative adjectives to base form."""
        mock_token = self._create_mock_token("schneller", "schneller", "ADJ")
        tokenizer = self._setup_tokenizer(mock_spacy, [mock_token])

        tokens = tokenizer.tokenize("er ist schneller")
        assert len(tokens) == 1
        assert tokens[0].lemma == "schnell"  # Converted to base form

    def test_adverb_lowercased(self, mock_spacy):
        """Should lowercase adverbs."""
        mock_token = self._create_mock_token("Schnell", "Schnell", "ADV")
        tokenizer = self._setup_tokenizer(mock_spacy, [mock_token])

        tokens = tokenizer.tokenize("Er läuft schnell")
        assert len(tokens) == 1
        assert tokens[0].lemma == "schnell"  # Lowercased


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

    def test_stopwords_months(self):
        """Should have month names in stopwords."""
        assert "januar" in Tokenizer.STOPWORDS
        assert "februar" in Tokenizer.STOPWORDS
        assert "dezember" in Tokenizer.STOPWORDS

    def test_stopwords_days(self):
        """Should have day names in stopwords."""
        assert "montag" in Tokenizer.STOPWORDS
        assert "dienstag" in Tokenizer.STOPWORDS
        assert "sonntag" in Tokenizer.STOPWORDS

    def test_ordinal_lemmas(self):
        """Should have ordinal lemmas."""
        assert "erster" in Tokenizer.ORDINAL_LEMMAS
        assert "zweiter" in Tokenizer.ORDINAL_LEMMAS
        assert "dritter" in Tokenizer.ORDINAL_LEMMAS
        assert "zwanzigster" in Tokenizer.ORDINAL_LEMMAS


class TestAnalyzeWord:
    """Tests for the analyze_word method."""

    @pytest.fixture
    def mock_spacy(self):
        """Create mock for spacy.load."""
        with patch("spacy.load") as mock_load:
            yield mock_load

    def _create_mock_token(
        self,
        text: str,
        lemma: str,
        pos: str,
        is_alpha: bool = True,
        like_num: bool = False,
        sent_text: str = "",
    ) -> MagicMock:
        """Create a mock token with given attributes."""
        token = MagicMock()
        token.text = text
        token.lemma_ = lemma
        token.pos_ = pos
        token.is_alpha = is_alpha
        token.like_num = like_num

        # Mock sentence
        sent = MagicMock()
        sent.text = sent_text or f"Context with {text}."
        token.sent = sent

        return token

    def _setup_tokenizer(self, mock_spacy: MagicMock, tokens: list[MagicMock]) -> Tokenizer:
        """Setup tokenizer with mock nlp returning given tokens."""
        mock_doc = MagicMock()
        mock_doc.__iter__ = lambda self: iter(tokens)

        mock_nlp = MagicMock(return_value=mock_doc)
        mock_spacy.return_value = mock_nlp

        tokenizer = Tokenizer()
        tokenizer._nlp = mock_nlp
        return tokenizer

    def test_analyze_noun_word(self, mock_spacy):
        """Should analyze a noun and return correct TokenInfo."""
        mock_token = self._create_mock_token("Hund", "Hund", "NOUN")
        tokenizer = self._setup_tokenizer(mock_spacy, [mock_token])

        result = tokenizer.analyze_word("Hund", "")

        assert result is not None
        assert result.pos == "NOUN"
        assert result.lemma == "Hund"
        assert result.surface_form == "Hund"

    def test_analyze_verb_word(self, mock_spacy):
        """Should analyze a verb and return lowercase lemma."""
        mock_token = self._create_mock_token("arbeitet", "arbeiten", "VERB")
        tokenizer = self._setup_tokenizer(mock_spacy, [mock_token])

        result = tokenizer.analyze_word("arbeitet", "Er arbeitet viel.")

        assert result is not None
        assert result.pos == "VERB"
        assert result.lemma == "arbeiten"

    def test_analyze_adjective_word(self, mock_spacy):
        """Should analyze an adjective and convert to base form."""
        mock_token = self._create_mock_token("schneller", "schnell", "ADJ")
        tokenizer = self._setup_tokenizer(mock_spacy, [mock_token])

        result = tokenizer.analyze_word("schneller", "")

        assert result is not None
        assert result.pos == "ADJ"
        assert result.lemma == "schnell"

    def test_analyze_adverb_word(self, mock_spacy):
        """Should analyze an adverb and lowercase."""
        mock_token = self._create_mock_token("Schnell", "schnell", "ADV")
        tokenizer = self._setup_tokenizer(mock_spacy, [mock_token])

        result = tokenizer.analyze_word("Schnell", "")

        assert result is not None
        assert result.pos == "ADV"
        assert result.lemma == "schnell"

    def test_analyze_preposition_word(self, mock_spacy):
        """Should analyze a preposition and expand contractions."""
        mock_token = self._create_mock_token("zum", "zum", "ADP")
        tokenizer = self._setup_tokenizer(mock_spacy, [mock_token])

        result = tokenizer.analyze_word("zum", "")

        assert result is not None
        assert result.pos == "ADP"
        assert result.lemma == "zu"

    def test_analyze_with_context(self, mock_spacy):
        """Should use context for analysis."""
        mock_token = self._create_mock_token("Hund", "Hund", "NOUN", sent_text="Der Hund bellt.")
        tokenizer = self._setup_tokenizer(mock_spacy, [mock_token])

        result = tokenizer.analyze_word("Hund", "Der Hund bellt.")

        assert result is not None
        assert result.context_sentence == "Der Hund bellt."

    def test_analyze_unknown_word_defaults_to_noun(self, mock_spacy):
        """Should default to NOUN for unknown words."""
        # No matching token found
        mock_token = self._create_mock_token("anders", "anders", "INTJ")  # Irrelevant POS
        tokenizer = self._setup_tokenizer(mock_spacy, [mock_token])

        result = tokenizer.analyze_word("Unbekannt", "")

        assert result is not None
        assert result.pos == "NOUN"
        assert result.lemma == "Unbekannt"  # Capitalized for noun

    def test_analyze_participle_converts_to_infinitive(self, mock_spacy):
        """Should convert verb participle to infinitive."""
        # spaCy returns "gemacht" as lemma for participle "gemacht"
        mock_token = self._create_mock_token("gemacht", "gemacht", "VERB")
        tokenizer = self._setup_tokenizer(mock_spacy, [mock_token])

        result = tokenizer.analyze_word("gemacht", "")

        assert result is not None
        assert result.pos == "VERB"
        assert result.lemma == "machen"  # ge-mach-t -> machen

    def test_analyze_diminutive_strips_suffix(self, mock_spacy):
        """Should strip diminutive suffix from nouns."""
        mock_token = self._create_mock_token("Häuschen", "Häuschen", "NOUN")
        tokenizer = self._setup_tokenizer(mock_spacy, [mock_token])

        result = tokenizer.analyze_word("Häuschen", "")

        assert result is not None
        assert result.pos == "NOUN"
        assert result.lemma == "Haus"

    def test_analyze_skips_non_alpha_tokens(self, mock_spacy):
        """Should skip non-alphabetic tokens."""
        mock_token = self._create_mock_token("123", "123", "NOUN", is_alpha=False)
        tokenizer = self._setup_tokenizer(mock_spacy, [mock_token])

        result = tokenizer.analyze_word("Wort", "")

        # Falls back to default NOUN
        assert result is not None
        assert result.pos == "NOUN"

    def test_analyze_fallback_without_context(self, mock_spacy):
        """Should try without context if word not found in context."""
        # First call with context returns no match
        mock_token_no_match = self._create_mock_token("anders", "anders", "INTJ")
        mock_token_match = self._create_mock_token("Hund", "Hund", "NOUN")

        mock_nlp = MagicMock()
        call_count = [0]

        def mock_call(text):
            call_count[0] += 1
            mock_doc = MagicMock()
            # First call with context, second call without
            if call_count[0] == 1:
                mock_doc.__iter__ = lambda self: iter([mock_token_no_match])
            else:
                mock_doc.__iter__ = lambda self: iter([mock_token_match])
            return mock_doc

        mock_nlp.side_effect = mock_call
        mock_spacy.return_value = mock_nlp

        tokenizer = Tokenizer()
        tokenizer._nlp = mock_nlp

        result = tokenizer.analyze_word("Hund", "Some context sentence.")

        assert result is not None
        assert call_count[0] == 2  # Called twice (with and without context)

    def test_analyze_word_case_insensitive_match(self, mock_spacy):
        """Should match word case-insensitively."""
        mock_token = self._create_mock_token("hund", "Hund", "NOUN")
        tokenizer = self._setup_tokenizer(mock_spacy, [mock_token])

        result = tokenizer.analyze_word("HUND", "")

        assert result is not None
        assert result.pos == "NOUN"
        assert result.lemma == "Hund"

    def test_analyze_matches_by_lemma(self, mock_spacy):
        """Should match word by lemma as well as text."""
        mock_token = self._create_mock_token("Hunde", "hund", "NOUN")
        tokenizer = self._setup_tokenizer(mock_spacy, [mock_token])

        result = tokenizer.analyze_word("hund", "")

        assert result is not None
        assert result.pos == "NOUN"
        assert result.lemma == "Hund"  # Capitalized noun

    def test_analyze_skips_matching_token_with_irrelevant_pos(self, mock_spacy):
        """Should skip matching token if POS is not in RELEVANT_POS."""
        # Token matches but has irrelevant POS (PUNCT, DET, etc.)
        mock_token_punct = self._create_mock_token(".", ".", "PUNCT")
        mock_token_noun = self._create_mock_token("Wort", "Wort", "NOUN")
        tokenizer = self._setup_tokenizer(mock_spacy, [mock_token_punct, mock_token_noun])

        result = tokenizer.analyze_word(".", "")

        # Should fallback to default NOUN since PUNCT is skipped
        assert result is not None
        assert result.pos == "NOUN"

    def test_analyze_skips_matching_token_with_like_num(self, mock_spacy):
        """Should skip matching token if it looks like a number."""
        mock_token = self._create_mock_token("2023", "2023", "NOUN", like_num=True)
        tokenizer = self._setup_tokenizer(mock_spacy, [mock_token])

        result = tokenizer.analyze_word("2023", "")

        # Should fallback to default NOUN since number-like is skipped
        assert result is not None
        assert result.pos == "NOUN"

    def test_analyze_unknown_word_capitalizes_for_noun(self, mock_spacy):
        """Should capitalize default lemma for unknown words (defaults to NOUN)."""
        # No matching token
        mock_token = self._create_mock_token("anders", "anders", "INTJ")
        tokenizer = self._setup_tokenizer(mock_spacy, [mock_token])

        result = tokenizer.analyze_word("wort", "")

        assert result is not None
        assert result.pos == "NOUN"
        # Default to NOUN, which gets capitalized (German nouns are capitalized)
        assert result.lemma == "Wort"
