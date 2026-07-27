"""Loanword detection for Kyrgyz orthography.

This matters for speech work well beyond etymology. A borrowed stem behaves
differently at three points a voice model will get wrong:

* **Stress.** Native Kyrgyz stress is word-final; Russian borrowings keep their
  source stress (``пре-зи-ДЕНТ`` but ``ми-НИСТР``, ``ТЕ-ле-фон`` vs ``те-ле-ФОН``
  depending on the source word). Applying the native rule to a loanword is the
  single most audible pronunciation error in Kyrgyz TTS.
* **Segments.** ``ж`` is /d͡ʒ/ in native words but /ʒ/ in Russian ones
  (``жыл`` /d͡ʒɯl/ vs ``журнал`` /ʒurnɑl/), and ``в ф х ц щ`` occur only in
  borrowings.
* **Harmony.** Borrowed stems are frequently disharmonic (``телевизор``), so a
  harmony-based check will fire on them spuriously unless they are identified.

Detection is deliberately evidence-based and graded rather than binary: each
signal contributes, and the result carries the reasons so an annotator can
overrule it. Nothing here needs a dictionary, which keeps it working on the open
vocabulary of news text — the corpus has 21k types and a long tail of names.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kgvoice.phon.alphabet import NON_NATIVE_LETTERS
from kgvoice.phon.harmony import check_harmony
from kgvoice.phon.lexicon import known_loan_stem
from kgvoice.phon.syllable import has_complex_onset, syllabify

#: Consonant clusters that native Kyrgyz phonotactics disallow word-initially.
#: Their presence is close to conclusive.
_FOREIGN_INITIAL_CLUSTERS = (
    "ст", "сп", "ск", "тр", "пр", "кр", "гр", "бр", "др", "фр", "пл", "кл",
    "гл", "бл", "фл", "шк", "шт", "шп", "зн", "зд", "св", "тв", "кв", "хр",
    "хл", "пс", "сх", "см", "сл", "сн", "вз", "вс", "вл", "гв", "дв",
)

#: Suffixes that mark an international/Russian borrowing regardless of the stem.
_FOREIGN_SUFFIXES = (
    "ция", "сия", "зия", "ика", "изм", "ист", "тор", "сор", "лог", "граф",
    "метр", "скоп", "нт", "ор", "ер", "ур", "аж", "ант", "ент",
)


@dataclass
class LoanwordVerdict:
    """Graded loanword judgement with its reasons."""

    word: str
    score: float
    reasons: list[str] = field(default_factory=list)

    #: Above this, treat the word as borrowed for stress and /ʒ/ purposes.
    THRESHOLD = 0.5

    @property
    def is_loanword(self) -> bool:
        return self.score >= self.THRESHOLD

    @property
    def confidence(self) -> str:
        if self.score >= 0.8:
            return "high"
        if self.score >= self.THRESHOLD:
            return "medium"
        if self.score > 0.0:
            return "low"
        return "none"

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"{self.word}: {self.score:.2f} ({self.confidence}) {'; '.join(self.reasons)}"


def analyze(word: str) -> LoanwordVerdict:
    """Grade how likely ``word`` is a borrowing."""
    w = word.lower().strip()
    reasons: list[str] = []
    score = 0.0

    if not w:
        return LoanwordVerdict(word, 0.0, [])

    # 0. Known borrowed stem. Conclusive, and the only signal that catches a
    #    borrowing which happens to be orthographically and harmonically
    #    well-formed — ``журнал`` is back-harmonic and uses only native letters,
    #    yet its ``ж`` is /ʒ/, not /d͡ʒ/.
    stem = known_loan_stem(w)
    if stem is not None:
        return LoanwordVerdict(word, 1.0, [f"known borrowed stem {stem!r}"])

    # 1. Letters that exist in Kyrgyz only for borrowings. Near-conclusive.
    foreign_letters = sorted({ch for ch in w if ch in NON_NATIVE_LETTERS})
    if foreign_letters:
        score += 0.6
        reasons.append(f"non-native letters: {' '.join(foreign_letters)}")

    # 2. Phonotactically impossible onsets.
    if any(w.startswith(c) for c in _FOREIGN_INITIAL_CLUSTERS):
        score += 0.5
        reasons.append(f"foreign initial cluster {w[:2]!r}")
    elif has_complex_onset(w):
        score += 0.3
        reasons.append("complex onset")

    # 3. Disharmonic stems. Weak on its own — compounds and some native stems
    #    are disharmonic too — so it is scored low and always reported.
    violations = check_harmony(w)
    if violations:
        score += 0.25 * min(len(violations), 2)
        reasons.append(
            "disharmonic: " + ", ".join(str(v) for v in violations[:3])
        )

    # 4. International derivational suffixes.
    for suf in _FOREIGN_SUFFIXES:
        if w.endswith(suf) and len(w) > len(suf) + 1:
            score += 0.3
            reasons.append(f"borrowed suffix -{suf}")
            break

    return LoanwordVerdict(word, min(score, 1.0), reasons)


def is_loanword(word: str) -> bool:
    """Convenience boolean over :func:`analyze`."""
    return analyze(word).is_loanword


def zh_is_fricative(word: str) -> bool:
    """Should ``ж`` in ``word`` be read /ʒ/ (Russian) rather than /d͡ʒ/ (native)?

    ``жыл`` /d͡ʒɯl/, ``жол`` /d͡ʒol/ — native, affricate.
    ``журнал`` /ʒurnɑl/, ``режим`` /reʒim/ — borrowed, fricative.
    """
    return analyze(word).is_loanword


def summarize(words: list[str]) -> dict[str, int]:
    """Count loanword confidence bands across a word list."""
    counts = {"high": 0, "medium": 0, "low": 0, "none": 0}
    for w in words:
        counts[analyze(w).confidence] += 1
    return counts


def syllable_profile(word: str) -> str:
    """Compact CV-shape signature, e.g. ``'CVC.CV'`` — useful for spotting
    non-native templates at a glance."""
    return ".".join(s.shape for s in syllabify(word))
