"""Pronunciation profiling for named entities.

Pulls the rest of :mod:`kgvoice.phon` together into one record per token or
entity: syllables, IPA, stress, loanword verdict, harmony, and a **difficulty
score** that answers the question a voice-data pipeline actually asks — *which
of these 16,829 entities are worth a human recording and a human check?*

The score is a weighted sum of independently observable signals, and every
contributing signal is kept on the record. That matters more than the number
itself: an annotator triaging a queue needs to know a token scored high because
its stress is unknown, not merely that it scored high.

Signals, in rough order of how much trouble they cause a Kyrgyz voice model:

``unknown-stress``   an unrecognised borrowing — placement cannot be derived
``loanword``         source-language segments and stress in a Kyrgyz frame
``disharmony``       vowels that break the harmony a TTS front end assumes
``ambiguous-stress`` two defensible placements (``ал-МА`` vs ``АЛ-ма``)
``complex-onset``    a cluster native phonotactics cannot open a syllable with
``final-cluster``    a two-or-more consonant coda
``long-vowel``       digraph length, routinely shortened by synthetic voices
``rare-phoneme``     ң ө ү — the segments non-native and low-data voices miss
``non-alphabetic``   digits, Latin, symbols needing verbalisation before TTS
``length``           a mild penalty for long, many-syllable tokens
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Sequence

from kgvoice.corpus.conll import EntitySpan
from kgvoice.phon import loanword as _loan
from kgvoice.phon.alphabet import LONG_VOWELS, is_kyrgyz_letter
from kgvoice.phon.g2p import ipa, transcribe
from kgvoice.phon.harmony import check_harmony
from kgvoice.phon.stress import StressResult, stress
from kgvoice.phon.syllable import (
    Syllable,
    final_cluster_size,
    has_complex_onset,
    hyphenate,
    syllabify,
)

#: Phonemes with no equivalent in Russian or English. Low-resource voices and
#: non-native speakers collapse them into neighbours (ө->о, ү->у, ң->н), which is
#: both the most common Kyrgyz pronunciation error and the easiest to hear.
RARE_PHONEMES = frozenset({"ң", "ө", "ү"})

#: Signal weights. Tuned so that a token needs more than one mild signal to clear
#: the review threshold, but a single unknown stress clears it alone.
WEIGHTS: dict[str, float] = {
    "unknown-stress": 0.45,
    "loanword": 0.20,
    "disharmony": 0.15,
    "ambiguous-stress": 0.20,
    "complex-onset": 0.15,
    "final-cluster": 0.10,
    "long-vowel": 0.08,
    "rare-phoneme": 0.10,
    "non-alphabetic": 0.25,
    "length": 0.10,
}


@dataclass
class WordProfile:
    """Everything the toolkit knows about how one token is pronounced."""

    word: str
    syllables: list[Syllable]
    ipa: str
    ipa_narrow: str
    stress: StressResult
    loanword: _loan.LoanwordVerdict
    signals: dict[str, float] = field(default_factory=dict)
    difficulty: float = 0.0

    @property
    def n_syllables(self) -> int:
        return len(self.syllables)

    @property
    def hyphenated(self) -> str:
        return hyphenate(self.word)

    @property
    def needs_review(self) -> bool:
        """Should a human look at this before it enters a voice dataset?"""
        return self.difficulty >= 0.40 or self.stress.needs_review

    def reasons(self) -> list[str]:
        """Signals that fired, strongest first."""
        return [k for k, v in sorted(self.signals.items(), key=lambda kv: -kv[1]) if v > 0]

    def as_dict(self) -> dict:
        return {
            "word": self.word,
            "syllables": [s.text for s in self.syllables],
            "hyphenated": self.hyphenated,
            "ipa": self.ipa,
            "ipa_narrow": self.ipa_narrow,
            "stress_index": self.stress.index,
            "stress_rule": self.stress.rule,
            "stress_confidence": self.stress.confidence,
            "stress_alternatives": self.stress.alternatives,
            "stress_marked": self.stress.marked(),
            "is_loanword": self.loanword.is_loanword,
            "loanword_confidence": self.loanword.confidence,
            "loanword_reasons": self.loanword.reasons,
            "difficulty": round(self.difficulty, 3),
            "signals": {k: round(v, 3) for k, v in self.signals.items()},
            "needs_review": self.needs_review,
        }


@lru_cache(maxsize=65536)
def profile_word(word: str) -> WordProfile:
    """Build a :class:`WordProfile` for a single token.

    Memoised. Profiling is the hot path for anything that walks a corpus —
    :mod:`kgvoice.bench.select` scores every sentence — and running text has far
    fewer distinct types than tokens (21k against 140k in KyrgyzNER), so the
    cache pays for itself immediately.

    The returned object is shared between callers and must be treated as
    read-only. Nothing in the toolkit mutates a profile; if you need to, take a
    copy first.
    """
    w = word.strip()
    syls = syllabify(w)
    st = stress(w)
    loan = _loan.analyze(w)
    lower = w.lower()

    signals: dict[str, float] = {}

    def fire(name: str, magnitude: float = 1.0) -> None:
        if magnitude > 0:
            signals[name] = WEIGHTS[name] * magnitude

    if st.confidence == "unknown":
        fire("unknown-stress")
    if st.alternatives:
        fire("ambiguous-stress")
    if loan.is_loanword:
        fire("loanword", loan.score)

    violations = check_harmony(lower)
    if violations:
        fire("disharmony", min(len(violations), 3) / 3)

    if has_complex_onset(lower):
        fire("complex-onset")

    coda = final_cluster_size(lower)
    if coda >= 2:
        fire("final-cluster", min(coda - 1, 2) / 2)

    n_long = sum(1 for d in LONG_VOWELS if d in lower)
    if n_long:
        fire("long-vowel", min(n_long, 2) / 2)

    n_rare = sum(1 for ch in lower if ch in RARE_PHONEMES)
    if n_rare:
        fire("rare-phoneme", min(n_rare, 3) / 3)

    non_alpha = [ch for ch in w if not is_kyrgyz_letter(ch) and not ch.isspace() and ch != "-"]
    if non_alpha:
        fire("non-alphabetic", min(len(non_alpha), 4) / 4)

    if len(syls) >= 4:
        fire("length", min(len(syls) - 3, 3) / 3)

    return WordProfile(
        word=w,
        syllables=syls,
        ipa=ipa(w),
        ipa_narrow=ipa(w, narrow=True),
        stress=st,
        loanword=loan,
        signals=signals,
        difficulty=min(sum(signals.values()), 1.0),
    )


@dataclass
class EntityProfile:
    """Profile of a multi-token entity mention."""

    text: str
    label: str
    words: list[WordProfile]
    sent_id: str = ""

    @property
    def ipa(self) -> str:
        return " ".join(w.ipa for w in self.words)

    @property
    def difficulty(self) -> float:
        """Hardest constituent, nudged up for multi-word mentions.

        Max rather than mean: an entity is as hard to say as its worst word, and
        averaging would let one unpronounceable name hide behind three easy ones.
        """
        if not self.words:
            return 0.0
        base = max(w.difficulty for w in self.words)
        return min(base + 0.03 * (len(self.words) - 1), 1.0)

    @property
    def needs_review(self) -> bool:
        return any(w.needs_review for w in self.words)

    def reasons(self) -> list[str]:
        seen: dict[str, None] = {}
        for w in self.words:
            for r in w.reasons():
                seen.setdefault(r, None)
        return list(seen)

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "label": self.label,
            "sent_id": self.sent_id,
            "ipa": self.ipa,
            "difficulty": round(self.difficulty, 3),
            "needs_review": self.needs_review,
            "reasons": self.reasons(),
            "words": [w.as_dict() for w in self.words],
        }


def profile_entity(span: EntitySpan) -> EntityProfile:
    """Profile every token of an entity span."""
    return EntityProfile(
        text=span.text,
        label=span.label,
        sent_id=span.sent_id,
        words=[profile_word(t) for t in span.tokens if t.strip()],
    )


def profile_entities(spans: Iterable[EntitySpan]) -> list[EntityProfile]:
    return [profile_entity(s) for s in spans]


def rank_by_difficulty(
    profiles: Sequence[EntityProfile], *, limit: int | None = None, min_difficulty: float = 0.0
) -> list[EntityProfile]:
    """Hardest entities first — the recording and review queue."""
    ranked = sorted(
        (p for p in profiles if p.difficulty >= min_difficulty),
        key=lambda p: -p.difficulty,
    )
    return ranked[:limit] if limit else ranked


def signal_histogram(profiles: Iterable[EntityProfile]) -> dict[str, int]:
    """How often each difficulty signal fires across a set of entities."""
    counts: dict[str, int] = {k: 0 for k in WEIGHTS}
    for p in profiles:
        for r in p.reasons():
            counts[r] += 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def transcribe_phrase(text: str, *, narrow: bool = False) -> str:
    """IPA for a whitespace-separated phrase."""
    return " ".join(transcribe(w, narrow=narrow) for w in text.split())
