"""Kyrgyz word stress.

The default is simple — stress falls on the final syllable — and that default is
right often enough that it is easy to ship a TTS voice which is wrong in exactly
the two places listeners notice.

**Non-stress-bearing morphemes.** A closed set of suffixes and clitics never
carries stress, so a word ending in one is stressed on the syllable *before* it:
the negative ``-БА``, the interrogative ``-БЫ``, the predicative person endings,
and the emphatic clitics. ``ба-ра-СЫҢ`` "you go" but ``бар-БА-сың`` "you don't
go"; ``кел-ДИ`` "he came" but ``КЕЛ-ди-би`` "did he come?".

**Borrowings.** Russian loans keep source stress, which is lexical and cannot be
derived: ``ми-НИСТР``, ``пре-зи-ДЕНТ``, ``ко-МИ-тет``. Native-rule stress on
these is the most audible single error in Kyrgyz synthetic speech, so this module
refuses to guess: an unrecognised borrowing returns ``confidence='unknown'`` and
is surfaced for human annotation rather than silently given final stress.

That refusal is the design point. For a voice dataset, a token flagged
"I don't know where this is stressed" is useful; a token confidently mis-stressed
is a defect that ships.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kgvoice.phon import loanword as _loan
from kgvoice.phon.lexicon import LOANWORD_STRESS, known_loan_stem
from kgvoice.phon.syllable import Syllable, syllabify, syllable_count

# --------------------------------------------------------------------------
# Non-stress-bearing morphemes
# --------------------------------------------------------------------------


def _harmonic_set(template: str) -> tuple[str, ...]:
    """Expand an archiphoneme template into its four surface shapes."""
    lows = "аеоө"
    highs = "ыиуү"
    out = []
    for vowels in (lows, highs):
        if "A" in template:
            out.extend(template.replace("A", v) for v in lows)
            break
        if "I" in template:
            out.extend(template.replace("I", v) for v in highs)
            break
    return tuple(dict.fromkeys(out))


#: Suffixes that reject stress, longest-first at match time.
UNSTRESSABLE: dict[str, tuple[str, ...]] = {
    # Predicative / copular person endings.
    "pred_1sg": _harmonic_set("мIн"),
    "pred_2sg": _harmonic_set("сIң"),
    "pred_2sg_polite": _harmonic_set("сIз"),
    "pred_1pl": _harmonic_set("бIз"),
    # Negation: -ба / -па / -ма by preceding voicing.
    "negative": _harmonic_set("бA") + _harmonic_set("пA") + _harmonic_set("мA"),
    # Interrogative particle.
    "question": _harmonic_set("бI") + _harmonic_set("пI") + _harmonic_set("мI"),
    # Habitual / emphatic clitics.
    "habitual": ("чу", "чү"),
    "emphatic": _harmonic_set("дA") + ("го", "ко", "гой"),
}

#: Minimum syllables the remaining stem must have before a class may be stripped.
#:
#: The person endings and the interrogative are phonologically distinctive — a
#: word ending in ``-сың`` or ``-би`` is almost certainly inflected — so a
#: monosyllabic stem is fine (``бар-БА-сың``).
#:
#: The negative ``-мA``/``-бA`` and the emphatic ``-дA`` are not. They are
#: homophonous with the final syllable of a large number of ordinary nouns:
#: ``алма`` (apple), ``тема``, ``форма``, ``дайра``, ``жаңылык-та``. Requiring a
#: two-syllable stem trades a rare error (a bare monosyllabic negative imperative,
#: ``барба``, which news text hardly contains) for a common one (every noun in
#: ``-ма``). Genuinely ambiguous cases are reported through
#: :attr:`StressResult.alternatives` rather than silently resolved.
_MIN_STEM_SYLLABLES: dict[str, int] = {
    "pred_1sg": 1,
    "pred_2sg": 1,
    "pred_2sg_polite": 1,
    "pred_1pl": 1,
    "question": 1,
    "habitual": 1,
    "negative": 2,
    "emphatic": 2,
}

#: Flattened, longest first, so -сыңбы strips in the right order.
_UNSTRESSABLE_FLAT: tuple[tuple[str, str], ...] = tuple(
    sorted(
        ((form, name) for name, forms in UNSTRESSABLE.items() for form in forms),
        key=lambda pair: -len(pair[0]),
    )
)


@dataclass
class StressResult:
    """Where stress falls in a word, and how much to trust it."""

    word: str
    syllables: list[Syllable]
    index: int | None
    rule: str
    confidence: str  # 'high' | 'medium' | 'unknown'
    #: Other defensible placements, when the analysis is not forced. Present so
    #: an annotator sees the competing reading instead of only the winner.
    alternatives: list[int] = field(default_factory=list)

    @property
    def n_syllables(self) -> int:
        return len(self.syllables)

    @property
    def needs_review(self) -> bool:
        """True when a human should confirm this before it reaches a voice model."""
        return self.confidence == "unknown" or bool(self.alternatives)

    def marked(self, sep: str = "-", upper: bool = True) -> str:
        """Hyphenated word with the stressed syllable highlighted."""
        if not self.syllables:
            return self.word
        parts = []
        for i, s in enumerate(self.syllables):
            t = s.text
            parts.append(t.upper() if (upper and i == self.index) else t)
        return sep.join(parts)

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"{self.marked()} [{self.rule}/{self.confidence}]"


def _strip_unstressable(word: str) -> tuple[str, list[str], bool]:
    """Peel non-stress-bearing suffixes off the right edge.

    Returns ``(stem, morpheme_names, ambiguous)``. ``ambiguous`` is set when a
    strip was declined only because the stem was too short — the caller can then
    report the competing analysis. See :data:`_MIN_STEM_SYLLABLES`.
    """
    stem = word.lower()
    removed: list[str] = []
    ambiguous = False
    changed = True
    while changed:
        changed = False
        for form, name in _UNSTRESSABLE_FLAT:
            if not stem.endswith(form) or len(stem) - len(form) < 2:
                continue
            candidate = stem[: -len(form)]
            if syllable_count(candidate) < _MIN_STEM_SYLLABLES[name]:
                ambiguous = True
                continue
            stem = candidate
            removed.append(name)
            changed = True
            break
    return stem, removed, ambiguous


def stress(word: str, *, assume_native: bool = False) -> StressResult:
    """Locate stress in ``word``.

    ``assume_native`` skips the loanword check, which is useful when the caller
    already knows the token is a native stem (or wants the native prediction for
    comparison).
    """
    clean = word.strip()
    syls = syllabify(clean)
    if not syls:
        return StressResult(word, syls, None, "no-vowel", "unknown")
    if len(syls) == 1:
        return StressResult(word, syls, 0, "monosyllable", "high")

    lower = clean.lower()

    if not assume_native:
        # 1. Known borrowing, possibly inflected. Kyrgyz suffixes stack onto a
        #    borrowed stem without moving its Russian stress, so the stem's index
        #    carries over unchanged: министр -> ми-НИСТР-лиг-и-нин.
        stem = known_loan_stem(lower)
        if stem is not None and stem in LOANWORD_STRESS:
            idx = min(LOANWORD_STRESS[stem], len(syls) - 1)
            rule = "loanword-lexicon" if stem == lower else f"loanword-lexicon ({stem}+)"
            return StressResult(word, syls, idx, rule, "high")

        # 2. Unknown borrowing — refuse to guess. See the module docstring.
        verdict = _loan.analyze(lower)
        if verdict.is_loanword:
            return StressResult(
                word, syls, None, f"loanword-unknown ({verdict.confidence})", "unknown"
            )

    # 3. Native rule: final syllable, skipping non-stress-bearing morphemes.
    final = len(syls) - 1
    stripped, removed, ambiguous = _strip_unstressable(lower)
    if removed:
        stem_syls = syllabify(stripped)
        if stem_syls:
            idx = min(len(stem_syls) - 1, final)
            return StressResult(
                word, syls, idx, "native-final minus " + "+".join(removed), "medium"
            )

    alternatives = [final - 1] if ambiguous and final >= 1 else []
    return StressResult(word, syls, final, "native-final", "high", alternatives)


def stress_index(word: str) -> int | None:
    return stress(word).index
