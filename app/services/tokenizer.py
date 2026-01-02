"""Tokenization service using spaCy for German text."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class TokenInfo:
    """Information about an extracted token."""

    surface_form: str
    lemma: str
    pos: str
    context_sentence: str


class Tokenizer:
    """Tokenize German text using spaCy."""

    # Parts of speech we care about
    RELEVANT_POS = {"NOUN", "VERB", "ADJ", "ADV", "ADP"}

    # Past participle patterns (often mistagged as nouns)
    PARTICIPLE_PREFIXES = (
        "ge",
        "ab",
        "an",
        "auf",
        "aus",
        "be",
        "ein",
        "er",
        "ver",
        "vor",
        "zu",
        "ent",
        "emp",
        "miss",
        "zer",
    )

    # Preposition contractions to expand
    PREPOSITION_CONTRACTIONS = {
        "zum": "zu",
        "zur": "zu",
        "im": "in",
        "ins": "in",
        "am": "an",
        "ans": "an",
        "beim": "bei",
        "vom": "von",
        "aufs": "auf",
        "ums": "um",
        "durchs": "durch",
        "fürs": "für",
        "hinters": "hinter",
        "übers": "über",
        "unters": "unter",
        "vors": "vor",
    }

    # Diminutives that are canonical forms (no base word in modern German)
    DIMINUTIVE_EXCEPTIONS = {
        "mädchen",
        "brötchen",
        "kaninchen",
        "eichhörnchen",
        "märchen",
        "veilchen",
        "hörnchen",
        "würstchen",
        "teilchen",
        "bisschen",
        "plätzchen",
    }

    # Words to skip (months, days) - cardinal numbers filtered by POS=NUM
    STOPWORDS = {
        # Months
        "januar",
        "februar",
        "märz",
        "april",
        "mai",
        "juni",
        "juli",
        "august",
        "september",
        "oktober",
        "november",
        "dezember",
        # Days of the week
        "montag",
        "dienstag",
        "mittwoch",
        "donnerstag",
        "freitag",
        "samstag",
        "sonntag",
    }

    # German ordinal base forms (lemmas as returned by spaCy) - tagged as ADJ
    ORDINAL_LEMMAS = {
        "erster",
        "zweiter",
        "dritter",
        "vierter",
        "fünfter",
        "sechster",
        "siebter",
        "achter",
        "neunter",
        "zehnter",
        "elfter",
        "zwölfter",
        "dreizehnter",
        "vierzehnter",
        "fünfzehnter",
        "sechzehnter",
        "siebzehnter",
        "achtzehnter",
        "neunzehnter",
        "zwanzigster",
        "dreißigster",
        "vierzigster",
        "fünfzigster",
        "sechzigster",
        "siebzigster",
        "achtzigster",
        "neunzigster",
        "hundertster",
        "tausendster",
    }

    def __init__(self, model_name: str = "de_core_news_lg") -> None:
        self.model_name = model_name
        self._nlp: Any = None

    def _looks_like_participle(self, word: str) -> bool:
        """Check if word looks like a German past participle."""
        lower = word.lower()

        # Common participle pattern: ge...t or ge...en
        if lower.startswith("ge") and (lower.endswith("t") or lower.endswith("en")):
            # Exclude common nouns: Gerät, Geschäft, Gebiet, Gesetz, etc.
            if lower in (
                "gerät",
                "geschäft",
                "gebiet",
                "gesetz",
                "gebet",
                "gedicht",
                "geheimnis",
                "gehalt",
                "gelände",
                "gemüt",
                "gericht",
                "geschlecht",
                "gesicht",
                "gewicht",
            ):
                return False
            return True

        # Separable verb participles with -ge-: abgerechnet, eingestellt, etc.
        for prefix in (
            "ab",
            "an",
            "auf",
            "aus",
            "ein",
            "vor",
            "zu",
            "mit",
            "nach",
            "um",
            "weg",
            "her",
            "hin",
        ):
            if prefix + "ge" in lower and (lower.endswith("t") or lower.endswith("en")):
                return True

        # -iert endings: analysiert, strukturiert
        if lower.endswith("iert"):
            return True

        return False

    def _is_feminine_role_noun(self, word: str) -> bool:
        """Check if word is a feminine form of a role/profession noun."""
        lower = word.lower()
        # Feminine forms end in -in or -innen
        if lower.endswith("innen"):
            return True
        if lower.endswith("in") and len(lower) > 4:
            # Check it's not a word that naturally ends in -in
            exceptions = (
                "termin",
                "berlin",
                "protein",
                "vitamin",
                "kamin",
                "delfin",
                "pinguin",
                "rosmarin",
                "satin",
                "ruin",
                "cousin",
            )
            if lower not in exceptions:
                return True
        return False

    def _is_nominalized_adjective(self, word: str) -> bool:
        """Check if word is a nominalized adjective (adjective used as noun)."""
        lower = word.lower()
        # Common adjective suffixes that indicate nominalized adjectives
        adjective_suffixes = (
            "bare",
            "baren",
            "barer",
            "bares",  # -bar (abschaltbar)
            "liche",
            "lichen",
            "licher",
            "liches",  # -lich
            "ige",
            "igen",
            "iger",
            "iges",  # -ig
            "ische",
            "ischen",
            "ischer",
            "isches",  # -isch
            "lose",
            "losen",
            "loser",
            "loses",  # -los
            "same",
            "samen",
            "samer",
            "sames",  # -sam
            "hafte",
            "haften",
            "hafter",
            "haftes",  # -haft
            "artige",
            "artigen",
            "artiger",
            "artiges",  # -artig
        )
        for suffix in adjective_suffixes:
            if lower.endswith(suffix) and len(lower) > len(suffix) + 2:
                return True
        return False

    def _participle_to_infinitive(self, word: str) -> str | None:
        """Convert a German past participle to its infinitive form."""
        lower = word.lower()

        # -iert → -ieren (analysiert → analysieren)
        if lower.endswith("iert"):
            return lower[:-1] + "en"

        # Separable verbs: abge...t → ab...en, abge...en → ab...en
        for prefix in (
            "ab",
            "an",
            "auf",
            "aus",
            "ein",
            "vor",
            "zu",
            "mit",
            "nach",
            "um",
            "weg",
            "her",
            "hin",
        ):
            if lower.startswith(prefix + "ge"):
                stem = lower[len(prefix) + 2 :]  # Remove prefix + "ge"
                if stem.endswith("t"):
                    return prefix + stem[:-1] + "en"
                elif stem.endswith("en"):
                    return prefix + stem

        # Regular ge...t → ...en (gearbeitet → arbeiten)
        if lower.startswith("ge") and lower.endswith("t"):
            stem = lower[2:-1]  # Remove "ge" and "t"
            return stem + "en"

        # ge...en → ...en (gegangen → gehen) - keep the stem
        if lower.startswith("ge") and lower.endswith("en"):
            return lower[2:]  # Remove "ge"

        return None

    def _restore_umlaut(self, word: str) -> str:
        """Remove umlauts that were added for comparative/diminutive forms."""
        # Order matters: äu before ä
        return word.replace("äu", "au").replace("ä", "a").replace("ö", "o").replace("ü", "u")

    def _adjective_to_base_form(self, word: str) -> str:
        """Convert comparative/superlative adjective to base form."""
        lower = word.lower()

        # Superlative: -sten, -ste, -st
        for suffix in ("sten", "ste", "st"):
            if lower.endswith(suffix) and len(lower) > len(suffix) + 2:
                base = lower[: -len(suffix)]
                return self._restore_umlaut(base)

        # Comparative: -er (but not words naturally ending in -er)
        if lower.endswith("er") and len(lower) > 4:
            base = lower[:-2]
            # Note: len(base) >= 3 is always True here since len(lower) > 4
            # implies len(lower) >= 5, so len(base) = len(lower) - 2 >= 3
            if len(base) >= 3:  # pragma: no branch
                return self._restore_umlaut(base)

        return lower

    def _diminutive_to_base(self, word: str) -> str:
        """Convert diminutive noun to base form."""
        lower = word.lower()

        # Keep canonical diminutives
        if lower in self.DIMINUTIVE_EXCEPTIONS:
            return word

        # -chen suffix (most common)
        if lower.endswith("chen") and len(lower) > 5:
            base = lower[:-4]
            return self._restore_umlaut(base)

        # -lein suffix
        if lower.endswith("lein") and len(lower) > 5:
            base = lower[:-4]
            return self._restore_umlaut(base)

        return word

    def _load_model(self) -> Any:
        """Load spaCy model lazily."""
        if self._nlp is None:
            import spacy

            logger.info(f"Loading spaCy model: {self.model_name}")
            self._nlp = spacy.load(self.model_name)
        return self._nlp

    def tokenize(self, text: str) -> list[TokenInfo]:
        """
        Extract relevant tokens from text.

        Returns deduplicated tokens with context sentences.
        """
        nlp = self._load_model()
        doc = nlp(text)

        # Track seen lemma+pos combinations for deduplication
        seen: set[tuple[str, str]] = set()
        tokens: list[TokenInfo] = []

        for sent in doc.sents:
            sentence_text = sent.text.strip()

            for token in sent:
                # Skip irrelevant POS
                if token.pos_ not in self.RELEVANT_POS:
                    continue

                # Skip punctuation, spaces, numbers
                if not token.is_alpha:
                    continue

                # Skip number-like tokens
                if token.like_num:
                    continue

                # Skip stopwords (months, days of week)
                if token.lemma_.lower() in self.STOPWORDS:
                    continue

                # Skip ordinal numbers (erste, zweite, dritte, etc.)
                if token.lemma_.lower() in self.ORDINAL_LEMMAS:
                    continue

                # Skip very short tokens
                if len(token.text) < 2:
                    continue

                # Skip participles mistagged as nouns or adjectives
                if token.pos_ in ("NOUN", "ADJ") and self._looks_like_participle(token.lemma_):
                    continue

                # Skip feminine forms of role/profession nouns
                if token.pos_ == "NOUN" and self._is_feminine_role_noun(token.lemma_):
                    continue

                # Skip nominalized adjectives (adjectives used as nouns)
                if token.pos_ == "NOUN" and self._is_nominalized_adjective(token.lemma_):
                    continue

                # Get canonical lemma based on POS
                if token.pos_ == "NOUN":
                    # Capitalize and convert diminutives to base form
                    lemma = self._diminutive_to_base(token.lemma_).capitalize().strip()
                elif token.pos_ == "VERB":
                    # Convert participle to infinitive if needed
                    if self._looks_like_participle(token.lemma_):
                        infinitive = self._participle_to_infinitive(token.lemma_)
                        lemma = (infinitive if infinitive else token.lemma_.lower()).strip()
                    else:
                        lemma = token.lemma_.lower().strip()
                elif token.pos_ == "ADJ":
                    # Convert comparative/superlative to base form
                    lemma = self._adjective_to_base_form(token.lemma_).strip()
                elif token.pos_ == "ADP":
                    # Expand preposition contractions
                    lower = token.lemma_.lower().strip()
                    lemma = self.PREPOSITION_CONTRACTIONS.get(lower, lower)
                else:
                    lemma = token.lemma_.lower().strip()

                # Create dedup key
                key = (lemma, token.pos_)

                # Skip if already seen in this document
                if key in seen:
                    continue
                seen.add(key)

                tokens.append(
                    TokenInfo(
                        surface_form=token.text,
                        lemma=lemma,
                        pos=token.pos_,
                        context_sentence=sentence_text,
                    )
                )

        logger.info(f"Extracted {len(tokens)} unique tokens from text")
        return tokens

    def analyze_word(self, word: str, context: str = "") -> TokenInfo | None:
        """
        Analyze a single word to determine its POS and normalized lemma.

        Uses the same spaCy pipeline as tokenize() for consistency.
        This is the preferred method for single-word analysis (e.g., vocab add).

        Args:
            word: The German word to analyze
            context: Optional context sentence for better POS detection

        Returns:
            TokenInfo with pos and lemma, or None if word cannot be analyzed
            (e.g., proper nouns are rejected)
        """
        nlp = self._load_model()

        # Build text for analysis - context helps with POS disambiguation
        if context:
            # Include context for better POS detection
            text = context
        else:
            # Just the word alone
            text = word

        doc = nlp(text)

        # Find the token matching our word
        word_lower = word.lower()
        best_match: TokenInfo | None = None
        is_proper_noun = False

        for token in doc:
            # Match by text (case-insensitive)
            if token.text.lower() == word_lower or token.lemma_.lower() == word_lower:
                # Reject proper nouns (names, places, etc.)
                if token.pos_ == "PROPN":
                    is_proper_noun = True
                    logger.info(f"Word '{word}' detected as proper noun (PROPN), rejecting")
                    continue

                # Skip irrelevant POS
                if token.pos_ not in self.RELEVANT_POS:
                    continue

                # Skip punctuation, spaces, numbers
                if not token.is_alpha or token.like_num:
                    continue

                # Get canonical lemma based on POS (same logic as tokenize)
                if token.pos_ == "NOUN":
                    lemma = self._diminutive_to_base(token.lemma_).capitalize().strip()
                elif token.pos_ == "VERB":
                    if self._looks_like_participle(token.lemma_):
                        infinitive = self._participle_to_infinitive(token.lemma_)
                        lemma = (infinitive if infinitive else token.lemma_.lower()).strip()
                    else:
                        lemma = token.lemma_.lower().strip()
                elif token.pos_ == "ADJ":
                    lemma = self._adjective_to_base_form(token.lemma_).strip()
                elif token.pos_ == "ADP":
                    lower = token.lemma_.lower().strip()
                    lemma = self.PREPOSITION_CONTRACTIONS.get(lower, lower)
                else:
                    lemma = token.lemma_.lower().strip()

                # Get context sentence
                sent_text = token.sent.text.strip() if token.sent else context or word

                best_match = TokenInfo(
                    surface_form=token.text,
                    lemma=lemma,
                    pos=token.pos_,
                    context_sentence=sent_text,
                )
                break  # Found a match

        # If no match in context, try analyzing the word alone
        if best_match is None and context and not is_proper_noun:
            return self.analyze_word(word, "")

        # If detected as proper noun, return None (reject it)
        if is_proper_noun and best_match is None:
            return None

        # If still no match, default to NOUN (most common for unknown words)
        if best_match is None:
            # German nouns are capitalized - always capitalize for default NOUN
            # Note: When we reach here, context is always empty because we recurse
            # with empty context above when no match is found with context present
            best_match = TokenInfo(
                surface_form=word,
                lemma=word.capitalize(),
                pos="NOUN",  # Default assumption
                context_sentence=context or word,
            )
            logger.debug(f"Word '{word}' not found in spaCy analysis, defaulting to NOUN")

        logger.info(f"Analyzed word '{word}': pos={best_match.pos}, lemma='{best_match.lemma}'")
        return best_match
