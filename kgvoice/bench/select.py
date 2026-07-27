"""Choosing what to record.

A benchmark is only as good as the sentences in it, and the naive choices are
both bad. Taking the first N sentences gives a set dominated by whatever the
corpus happens to lead with; taking N at random gives a set that mostly measures
easy, common vocabulary, because that is what most sentences contain.

What a *voice* benchmark needs is coverage of the things that break voice models:
entity-dense text, numbers and dates read aloud, acronyms, and words the
phonology module has flagged as difficult. This module scores sentences on those
axes and then selects greedily for **phoneme coverage** so the resulting set is
diverse rather than N variations on one hard sentence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from kgvoice.corpus.conll import Sentence
from kgvoice.phon.g2p import phones
from kgvoice.phon.profile import profile_word

_DIGIT = re.compile(r"\d")
_ACRONYM = re.compile(r"^[А-ЯЁӨҮҢ]{2,}$")
_LATIN = re.compile(r"[A-Za-z]")
_CYRILLIC = re.compile(r"[А-Яа-яЁёӨөҮүҢң]")
_URLISH = re.compile(r"https?|www\.|://|\.kg\b|\.com\b", re.IGNORECASE)

#: Residue of HTML entity decoding in the source scrape («  -> ``laquo``). These
#: are not words, and a sentence containing them is not a sentence.
_ENTITY_JUNK = frozenset({"laquo", "raquo", "quot", "amp", "nbsp", "ndash", "mdash", "rsquo"})


@dataclass(frozen=True)
class Unreadable:
    """Why a sentence cannot serve as a recording prompt."""

    sent_id: str
    reason: str
    text: str


def readability_problem(sentence: Sentence, *, max_latin_ratio: float = 0.05) -> str | None:
    """Reason ``sentence`` is unfit to read aloud, or ``None`` if it is fine.

    This filter is not optional polish. KyrgyzNER is scraped from a news site and
    retains URLs, Latin-script transliteration, and HTML entity residue
    (``laquodyiykanraquo``). Those sentences score *well* on the difficulty axis —
    they are full of non-alphabetic material — so without an explicit gate they
    dominate the top of the ranking and a manifest built from it is unreadable.

    Rejecting them here rather than silently down-weighting keeps the decision
    visible: :func:`rejected` returns the list with reasons, which is a data
    quality report in its own right.
    """
    text = sentence.text
    if not text.strip():
        return "empty"
    if _URLISH.search(text):
        return "contains a URL or domain"
    lowered = {t.text.lower() for t in sentence.tokens}
    if lowered & _ENTITY_JUNK or any(
        any(j in t for j in _ENTITY_JUNK) for t in lowered
    ):
        return "HTML entity residue"
    letters = _LATIN.findall(text)
    cyrillic = _CYRILLIC.findall(text)
    total = len(letters) + len(cyrillic)
    if total == 0:
        return "no letters"
    if len(letters) / total > max_latin_ratio:
        return f"{len(letters) / total:.0%} Latin script"
    if len(cyrillic) < 10:
        return "too little Kyrgyz text"
    return None


def is_readable(sentence: Sentence) -> bool:
    return readability_problem(sentence) is None


def rejected(sentences: Iterable[Sentence]) -> list[Unreadable]:
    """Every sentence the readability gate excludes, with its reason."""
    out = []
    for s in sentences:
        why = readability_problem(s)
        if why:
            out.append(Unreadable(s.sent_id, why, s.text[:120]))
    return out


#: Entity classes whose spoken form differs from their written form, and which
#: therefore exercise the text-normalisation front end.
SPOKEN_FORM_LABELS = frozenset({"MEASURE", "PERIOD", "IDENTIFIER", "WEBSITE", "ACRONYM"})

#: Default weights. Entity density dominates, because the entity-WER figure the
#: benchmark exists to produce is meaningless on sentences with few entities.
WEIGHTS: dict[str, float] = {
    "entity_density": 2.0,
    "spoken_form": 1.5,
    "numerals": 1.2,
    "acronyms": 1.0,
    "phonetic_difficulty": 1.5,
    "length_fit": 0.8,
}


@dataclass
class SentenceScore:
    """A candidate sentence with its component scores."""

    sentence: Sentence
    components: dict[str, float] = field(default_factory=dict)
    total: float = 0.0
    phoneme_set: frozenset[str] = frozenset()

    @property
    def text(self) -> str:
        return self.sentence.text

    @property
    def n_tokens(self) -> int:
        return len(self.sentence.tokens)

    def as_dict(self) -> dict:
        return {
            "sent_id": self.sentence.sent_id,
            "text": self.text,
            "n_tokens": self.n_tokens,
            "entities": [
                {"text": e.text, "label": e.label} for e in self.sentence.entities()
            ],
            "score": round(self.total, 4),
            "components": {k: round(v, 4) for k, v in self.components.items()},
        }


def _length_fit(n: int, ideal_low: int = 8, ideal_high: int = 22) -> float:
    """Prefer sentences a person can read in one comfortable breath.

    Too short and there is no prosody to evaluate; too long and the reader will
    stumble for reasons that have nothing to do with the model under test.
    """
    if n < 4 or n > 45:
        return 0.0
    if ideal_low <= n <= ideal_high:
        return 1.0
    if n < ideal_low:
        return (n - 3) / (ideal_low - 3)
    return max(0.0, 1.0 - (n - ideal_high) / (45 - ideal_high))


def score_sentence(sentence: Sentence, weights: dict[str, float] | None = None) -> SentenceScore:
    """Score one sentence for benchmark value."""
    weights = weights or WEIGHTS
    tokens = [t.text for t in sentence.tokens]
    n = len(tokens)
    if n == 0:
        return SentenceScore(sentence, {}, 0.0, frozenset())

    spans = sentence.entities()
    entity_tokens = sum(s.length for s in spans)

    comp: dict[str, float] = {}
    comp["entity_density"] = entity_tokens / n
    comp["spoken_form"] = min(
        sum(1 for s in spans if s.label in SPOKEN_FORM_LABELS) / 3.0, 1.0
    )
    comp["numerals"] = min(sum(1 for t in tokens if _DIGIT.search(t)) / 3.0, 1.0)
    comp["acronyms"] = min(sum(1 for t in tokens if _ACRONYM.match(t)) / 2.0, 1.0)

    word_profiles = [profile_word(t) for t in tokens if t.isalpha() and len(t) > 2]
    comp["phonetic_difficulty"] = (
        sum(p.difficulty for p in word_profiles) / len(word_profiles) if word_profiles else 0.0
    )
    comp["length_fit"] = _length_fit(n)

    weighted = {k: v * weights.get(k, 0.0) for k, v in comp.items()}
    total = sum(weighted.values()) / sum(weights.values())

    phoneme_set = frozenset(
        p.ipa for t in tokens if t.isalpha() for p in phones(t)
    )
    return SentenceScore(sentence, comp, total, phoneme_set)


def rank(
    sentences: Iterable[Sentence],
    weights: dict[str, float] | None = None,
    *,
    readable_only: bool = True,
) -> list[SentenceScore]:
    """All sentences scored, best first.

    ``readable_only`` applies :func:`readability_problem`. Leave it on for
    anything that produces prompts; turn it off only to score the raw corpus.
    """
    pool = [s for s in sentences if not readable_only or is_readable(s)]
    scored = [score_sentence(s, weights) for s in pool]
    scored.sort(key=lambda s: -s.total)
    return scored


def select(
    sentences: Iterable[Sentence],
    n: int = 100,
    *,
    weights: dict[str, float] | None = None,
    diversity: float = 0.5,
    min_score: float = 0.0,
    readable_only: bool = True,
) -> list[SentenceScore]:
    """Pick ``n`` sentences, trading raw score against phoneme coverage.

    ``diversity`` at 0 selects the top ``n`` by score alone. At 1 each pick is
    dominated by how many *new* phonemes it contributes. The default splits the
    difference: high-value sentences first, but a sentence that only repeats
    sounds already covered loses to a slightly weaker one that adds new ones.

    Greedy rather than optimal — maximum coverage is NP-hard, and greedy is
    within a constant factor of optimal for it, which is far inside the noise on
    a subjective scoring function.
    """
    pool = [
        s
        for s in rank(sentences, weights, readable_only=readable_only)
        if s.total >= min_score
    ]
    if diversity <= 0:
        return pool[:n]

    chosen: list[SentenceScore] = []
    covered: set[str] = set()
    remaining = list(pool)

    while remaining and len(chosen) < n:
        best_idx = 0
        best_value = -1.0
        for i, cand in enumerate(remaining):
            new = len(cand.phoneme_set - covered)
            novelty = new / max(len(cand.phoneme_set), 1)
            value = (1 - diversity) * cand.total + diversity * novelty
            if value > best_value:
                best_value, best_idx = value, i
        pick = remaining.pop(best_idx)
        chosen.append(pick)
        covered |= pick.phoneme_set

    return chosen


def coverage_report(selected: Sequence[SentenceScore], universe: Sequence[SentenceScore]) -> dict:
    """How much of the corpus's phoneme inventory the selection covers."""
    sel = set().union(*(s.phoneme_set for s in selected)) if selected else set()
    uni = set().union(*(s.phoneme_set for s in universe)) if universe else set()
    labels: dict[str, int] = {}
    for s in selected:
        for e in s.sentence.entities():
            labels[e.label] = labels.get(e.label, 0) + 1
    return {
        "sentences": len(selected),
        "tokens": sum(s.n_tokens for s in selected),
        "phonemes_covered": len(sel),
        "phonemes_in_corpus": len(uni),
        "phoneme_coverage": round(len(sel) / len(uni), 4) if uni else 0.0,
        "missing_phonemes": sorted(uni - sel),
        "entity_labels": dict(sorted(labels.items(), key=lambda kv: -kv[1])),
        "mean_score": round(sum(s.total for s in selected) / len(selected), 4) if selected else 0.0,
    }
