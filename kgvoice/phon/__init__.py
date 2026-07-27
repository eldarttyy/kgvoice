"""Kyrgyz phonology: inventory, harmony, syllabification, stress, and G2P.

``alphabet`` is a plain feature table; every other module in this package is
stated over it.

* ``harmony``  — suffix realisation, plus an audit that re-derives the rules
  from a corpus so they stay falsifiable rather than asserted.
* ``syllable`` — syllabification, including the loanword complex-onset case.
* ``lexicon``  — hand-checked borrowed stems and their source stress, shared by
  ``loanword`` and ``stress`` so neither has to import the other.
* ``loanword`` — graded loanword detection, which ``g2p`` and ``stress`` consult.
* ``stress``   — word stress, including non-stress-bearing suffixes and an
  explicit refusal to guess at unknown borrowings.
* ``g2p``      — grapheme-to-phoneme with context-sensitive dorsal, lateral and
  length rules.
* ``profile``  — per-token and per-entity pronunciation records with a
  difficulty score, used to build recording and review queues.

Submodules are exposed as modules rather than having their contents flattened
into this namespace: ``stress.stress`` and ``loanword.analyze`` are names that
would shadow or read ambiguously at package level.

    >>> from kgvoice.phon import g2p, syllable
    >>> g2p.transcribe("көл")
    'køl'
    >>> syllable.hyphenate("кыргыз")
    'кыр-гыз'
"""

from kgvoice.phon import g2p, harmony, lexicon, loanword, profile, stress, syllable
from kgvoice.phon.alphabet import (
    ALPHABET,
    CONSONANTS,
    HARMONIC_VOWELS,
    LONG_VOWELS,
    NON_NATIVE_LETTERS,
    TRIGGER_VOWELS,
    VOWELS,
    Consonant,
    Vowel,
    harmonic_vowels_of,
    is_consonant,
    is_kyrgyz_letter,
    is_vowel,
    last_harmonic_vowel,
    vowels_of,
)
from kgvoice.phon.harmony import (
    SUFFIXES,
    AuditResult,
    HarmonyViolation,
    attach,
    audit_all,
    audit_archiphoneme,
    audit_consonant,
    check_harmony,
    harmonic_class,
    harmonize,
    is_harmonic,
    paradigm,
    realize,
    realize_consonant,
    realize_for_stem,
    spreads_rounding,
)
from kgvoice.phon.profile import (
    EntityProfile,
    WordProfile,
    profile_entities,
    profile_entity,
    profile_word,
    rank_by_difficulty,
    signal_histogram,
)

__all__ = [
    # submodules
    "g2p",
    "harmony",
    "lexicon",
    "loanword",
    "profile",
    "stress",
    "syllable",
    # alphabet
    "ALPHABET",
    "CONSONANTS",
    "HARMONIC_VOWELS",
    "LONG_VOWELS",
    "NON_NATIVE_LETTERS",
    "TRIGGER_VOWELS",
    "VOWELS",
    "Consonant",
    "Vowel",
    "harmonic_vowels_of",
    "is_consonant",
    "is_kyrgyz_letter",
    "is_vowel",
    "last_harmonic_vowel",
    "vowels_of",
    # harmony
    "SUFFIXES",
    "AuditResult",
    "HarmonyViolation",
    "attach",
    "audit_all",
    "audit_archiphoneme",
    "audit_consonant",
    "check_harmony",
    "harmonic_class",
    "harmonize",
    "is_harmonic",
    "paradigm",
    "realize",
    "realize_consonant",
    "realize_for_stem",
    "spreads_rounding",
    # profile
    "EntityProfile",
    "WordProfile",
    "profile_entities",
    "profile_entity",
    "profile_word",
    "rank_by_difficulty",
    "signal_histogram",
]
