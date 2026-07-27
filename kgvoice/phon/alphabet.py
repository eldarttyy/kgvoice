"""Kyrgyz Cyrillic inventory with phonological features.

Vowel features follow the standard three-way description of the Kyrgyz system
(backness x rounding x height). Height matters here for more than tidiness: the
rounding-harmony rule in :mod:`kgvoice.phon.harmony` is conditioned on the
height of both trigger and target, and cannot be stated without it.

Consonant IPA follows the Kyrgyz phonology description on Wikipedia, with
allophony applied separately in :mod:`kgvoice.phon.g2p` so that this table stays
a plain phoneme inventory.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------
# Vowels
# --------------------------------------------------------------------------

BACK = "back"
FRONT = "front"
HIGH = "high"
LOW = "low"


@dataclass(frozen=True)
class Vowel:
    letter: str
    ipa: str
    backness: str  # BACK | FRONT
    rounded: bool
    height: str  # HIGH | LOW  (two-way, which is all harmony needs)
    native: bool = True

    @property
    def is_back(self) -> bool:
        return self.backness == BACK

    @property
    def is_high(self) -> bool:
        return self.height == HIGH


#: The eight native Kyrgyz vowels.
VOWELS: dict[str, Vowel] = {
    "а": Vowel("а", "ɑ", BACK, False, LOW),
    "ы": Vowel("ы", "ɯ", BACK, False, HIGH),
    "о": Vowel("о", "o", BACK, True, LOW),
    "у": Vowel("у", "u", BACK, True, HIGH),
    "е": Vowel("е", "e", FRONT, False, LOW),
    "и": Vowel("и", "i", FRONT, False, HIGH),
    "ө": Vowel("ө", "ø", FRONT, True, LOW),
    "ү": Vowel("ү", "y", FRONT, True, HIGH),
    # Non-harmonic / Russian-orthography vowels. 'э' is native but restricted to
    # word-initial position and patterns with 'е'.
    "э": Vowel("э", "e", FRONT, False, LOW, native=True),
    "я": Vowel("я", "jɑ", BACK, False, LOW, native=False),
    "ю": Vowel("ю", "ju", BACK, True, HIGH, native=False),
    "ё": Vowel("ё", "jo", BACK, True, LOW, native=False),
}

#: Vowels that a *native stem* is expected to harmonise internally: the
#: eight-vowel core plus 'э'. Used to judge whether a stem is harmonic.
HARMONIC_VOWELS = {k: v for k, v in VOWELS.items() if v.native}

#: Vowels that can *trigger* harmony on a following suffix. This is a strictly
#: larger set: the iotated Russian letters я /jɑ/, ю /ju/, ё /jo/ carry ordinary
#: back vowels under the glide, and Kyrgyz suffixes harmonise to them normally.
#: Borrowings in -ия are the common case and are unambiguously back stems —
#: россия -> россия+нын, биология -> биология+лык — so omitting them here makes
#: the harmony engine mis-predict every Russian -ия loan in the corpus.
TRIGGER_VOWELS = dict(VOWELS)

VOWEL_LETTERS = frozenset(VOWELS)
NATIVE_VOWEL_LETTERS = frozenset(HARMONIC_VOWELS)

#: Long vowels are written as doubled graphemes. /i/ and /ɯ/ have no long
#: counterpart in native vocabulary, so 'ии'/'ыы' are not listed.
LONG_VOWELS: dict[str, str] = {
    "аа": "ɑː",
    "ээ": "eː",
    "ее": "eː",
    "оо": "oː",
    "өө": "øː",
    "уу": "uː",
    "үү": "yː",
}

# --------------------------------------------------------------------------
# Consonants
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Consonant:
    letter: str
    ipa: str
    voiced: bool
    native: bool = True
    #: Sonorants condition the voicing of following suffix-initial stops and
    #: block the desonorisation rule; kept as a feature rather than a set
    #: membership test so g2p and stress can both consult it.
    sonorant: bool = False


CONSONANTS: dict[str, Consonant] = {
    "б": Consonant("б", "b", True),
    "в": Consonant("в", "v", True, native=False),
    "г": Consonant("г", "ɡ", True),
    "д": Consonant("д", "d", True),
    "ж": Consonant("ж", "d͡ʒ", True),
    "з": Consonant("з", "z", True),
    "й": Consonant("й", "j", True, sonorant=True),
    "к": Consonant("к", "k", False),
    "л": Consonant("л", "l", True, sonorant=True),
    "м": Consonant("м", "m", True, sonorant=True),
    "н": Consonant("н", "n", True, sonorant=True),
    "ң": Consonant("ң", "ŋ", True, sonorant=True),
    "п": Consonant("п", "p", False),
    "р": Consonant("р", "r", True, sonorant=True),
    "с": Consonant("с", "s", False),
    "т": Consonant("т", "t", False),
    "ф": Consonant("ф", "f", False, native=False),
    "х": Consonant("х", "x", False, native=False),
    "ц": Consonant("ц", "t͡s", False, native=False),
    "ч": Consonant("ч", "t͡ʃ", False),
    "ш": Consonant("ш", "ʃ", False),
    "щ": Consonant("щ", "ʃt͡ʃ", False, native=False),
}

CONSONANT_LETTERS = frozenset(CONSONANTS)

#: Orthographic signs with no segmental value of their own.
SIGNS = frozenset({"ъ", "ь"})

#: Letters that occur only in Russian/international borrowings. Their presence
#: is the cheapest reliable loanword signal in Kyrgyz orthography.
NON_NATIVE_LETTERS = frozenset(
    {c.letter for c in CONSONANTS.values() if not c.native}
    | {v.letter for v in VOWELS.values() if not v.native}
    | SIGNS
)

ALPHABET = frozenset(VOWEL_LETTERS | CONSONANT_LETTERS | SIGNS)


def is_vowel(ch: str) -> bool:
    return ch.lower() in VOWEL_LETTERS


def is_consonant(ch: str) -> bool:
    return ch.lower() in CONSONANT_LETTERS


def is_kyrgyz_letter(ch: str) -> bool:
    return ch.lower() in ALPHABET


def vowels_of(word: str) -> list[Vowel]:
    """Every vowel in ``word``, in order."""
    return [VOWELS[ch] for ch in word.lower() if ch in VOWELS]


def harmonic_vowels_of(word: str) -> list[Vowel]:
    """Only the natively harmony-participating vowels of ``word``, in order."""
    return [VOWELS[ch] for ch in word.lower() if ch in HARMONIC_VOWELS]


def last_harmonic_vowel(word: str) -> Vowel | None:
    """The rightmost harmony *trigger* in ``word``.

    Uses :data:`TRIGGER_VOWELS`, so ``я``/``ю``/``ё`` count. See the note there.
    """
    for ch in reversed(word.lower()):
        if ch in TRIGGER_VOWELS:
            return TRIGGER_VOWELS[ch]
    return None
