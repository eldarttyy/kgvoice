"""Address register: сен vs сиз consistency across a catalogue.

Kyrgyz distinguishes familiar ``сен`` from polite ``сиз`` in the second person,
and the choice propagates into verb endings and possessives. A product has to
pick one and hold it. Mixing them inside one interface reads as careless in a way
that is obvious to a native speaker and invisible to a translation memory:

    Сырсөз**үңүздү** киргиз**иңиз**     (formal)
    Сырсөз**үңдү** киргиз               (familiar)

Both are correct Kyrgyz. Only one of them belongs in a given product, and the
defect is that the catalogue contains both.

Detection is by marker, with confidence, and the markers are chosen for
precision rather than coverage. Two deliberate omissions are worth stating.

**``-сыз`` is not used as a formality marker.** It is the polite predicative
ending (``барасыз`` "you go"), and it is *also* the caritive suffix meaning
"without" — ``үнсүз`` "silent", ``акчасыз`` "without money", ``чексиз``
"unlimited". Both are frequent in UI copy. Separating them needs to know whether
the stem is a verb or a noun, which this module does not, so counting ``-сыз``
as formal would misread ordinary vocabulary as an address choice. Its familiar
counterpart ``-сың`` has no such homonym and *is* used.

**Bare imperatives are not counted as familiar.** The familiar imperative is the
bare verb stem (``сакта`` "save"), which is indistinguishable from a noun or an
infinitive-style label without parsing. Kyrgyz UI convention favours verbal nouns
for buttons (``Сактоо``) anyway, so treating every bare stem as familiar would
flag most button labels in the catalogue.

What remains — the pronoun paradigms and the polite endings ``-ңыз/-ңиз`` — is
high precision. Recall is deliberately traded away: a register report that cries
wolf gets switched off.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

FORMAL = "formal"
FAMILIAR = "familiar"

#: The ``сиз`` pronoun paradigm, including the plural ``сиздер``.
FORMAL_PRONOUNS: frozenset[str] = frozenset(
    {
        "сиз", "сизди", "сиздин", "сизге", "сизде", "сизден", "сиздики",
        "сиздер", "сиздерди", "сиздердин", "сиздерге", "сиздерде", "сиздерден",
    }
)

#: The ``сен`` pronoun paradigm. ``сага`` and ``сеники`` are irregular.
FAMILIAR_PRONOUNS: frozenset[str] = frozenset(
    {"сен", "сени", "сенин", "сага", "сенде", "сенден", "сеники"}
)

#: Word-final endings, longest first. ``(ending, register, confidence, kind)``.
#: Order matters: ``-ыңыз`` must be tried before ``-ың`` or every polite
#: possessive would be read as a familiar one.
_ENDINGS: tuple[tuple[str, str, str, str], ...] = (
    ("ыңыз", FORMAL, "high", "polite possessive/imperative"),
    ("иңиз", FORMAL, "high", "polite possessive/imperative"),
    ("уңуз", FORMAL, "high", "polite possessive/imperative"),
    ("үңүз", FORMAL, "high", "polite possessive/imperative"),
    ("ңыз", FORMAL, "high", "polite imperative"),
    ("ңиз", FORMAL, "high", "polite imperative"),
    ("ңуз", FORMAL, "high", "polite imperative"),
    ("ңүз", FORMAL, "high", "polite imperative"),
    ("сың", FAMILIAR, "medium", "familiar predicative"),
    ("сиң", FAMILIAR, "medium", "familiar predicative"),
    ("суң", FAMILIAR, "medium", "familiar predicative"),
    ("сүң", FAMILIAR, "medium", "familiar predicative"),
    ("гын", FAMILIAR, "medium", "familiar imperative"),
    ("гин", FAMILIAR, "medium", "familiar imperative"),
    ("гун", FAMILIAR, "medium", "familiar imperative"),
    ("гүн", FAMILIAR, "medium", "familiar imperative"),
    ("ың", FAMILIAR, "low", "familiar possessive"),
    ("иң", FAMILIAR, "low", "familiar possessive"),
    ("уң", FAMILIAR, "low", "familiar possessive"),
    ("үң", FAMILIAR, "low", "familiar possessive"),
)

#: Minimum stem length left after stripping an ending, so ``сиң`` does not match
#: the whole of a three-letter word.
_MIN_STEM = 3

_WORD_RE = re.compile(r"[а-яёөүң]+", re.IGNORECASE)

_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True)
class RegisterMarker:
    """One piece of evidence for an address register."""

    word: str
    register: str
    confidence: str
    kind: str

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"{self.word} ({self.kind}, {self.confidence})"


@dataclass
class RegisterVerdict:
    """The register a string is written in, and why."""

    text: str
    markers: list[RegisterMarker] = field(default_factory=list)

    @property
    def registers(self) -> set[str]:
        return {m.register for m in self.markers}

    @property
    def register(self) -> str | None:
        """The dominant register, or ``None`` when the string is neutral.

        Ties are broken by the strongest evidence rather than by count: one
        pronoun outweighs two low-confidence suffix guesses.
        """
        if not self.markers:
            return None
        scores: Counter = Counter()
        for m in self.markers:
            scores[m.register] += _CONFIDENCE_RANK[m.confidence] + 1
        best = scores.most_common()
        if len(best) > 1 and best[0][1] == best[1][1]:
            return None
        return best[0][0]

    @property
    def is_mixed(self) -> bool:
        """True when one string uses both registers — always a defect."""
        return len(self.registers) > 1

    @property
    def is_neutral(self) -> bool:
        return not self.markers

    def evidence(self, register: str | None = None) -> list[RegisterMarker]:
        if register is None:
            return list(self.markers)
        return [m for m in self.markers if m.register == register]


def _classify_word(word: str) -> RegisterMarker | None:
    lower = word.lower()
    if lower in FORMAL_PRONOUNS:
        return RegisterMarker(word, FORMAL, "high", "polite pronoun")
    if lower in FAMILIAR_PRONOUNS:
        return RegisterMarker(word, FAMILIAR, "high", "familiar pronoun")
    for ending, register, confidence, kind in _ENDINGS:
        if lower.endswith(ending) and len(lower) - len(ending) >= _MIN_STEM:
            return RegisterMarker(word, register, confidence, kind)
    return None


def detect(text: str) -> RegisterVerdict:
    """Find every register marker in ``text``."""
    markers = []
    for word in _WORD_RE.findall(text):
        m = _classify_word(word)
        if m is not None:
            markers.append(m)
    return RegisterVerdict(text=text, markers=markers)


def register_of(text: str) -> str | None:
    return detect(text).register


@dataclass
class RegisterReport:
    """Register usage across a whole catalogue."""

    by_key: dict[str, RegisterVerdict] = field(default_factory=dict)

    @property
    def counts(self) -> Counter:
        c: Counter = Counter()
        for v in self.by_key.values():
            r = v.register
            c[r or "neutral"] += 1
        return c

    @property
    def dominant(self) -> str | None:
        """The register the catalogue mostly uses, ignoring neutral strings."""
        c = Counter(
            {k: v for k, v in self.counts.items() if k in (FORMAL, FAMILIAR)}
        )
        return c.most_common(1)[0][0] if c else None

    @property
    def minority_keys(self) -> list[str]:
        """Keys that disagree with the dominant register."""
        dom = self.dominant
        if dom is None:
            return []
        return [
            k
            for k, v in self.by_key.items()
            if v.register is not None and v.register != dom
        ]

    @property
    def mixed_keys(self) -> list[str]:
        """Keys that use both registers inside one string."""
        return [k for k, v in self.by_key.items() if v.is_mixed]

    @property
    def is_consistent(self) -> bool:
        return not self.minority_keys and not self.mixed_keys

    def summary(self) -> dict:
        return {
            "dominant": self.dominant,
            "counts": dict(self.counts),
            "inconsistent_keys": len(self.minority_keys),
            "mixed_keys": len(self.mixed_keys),
            "consistent": self.is_consistent,
        }


def audit_catalog(entries) -> RegisterReport:
    """Build a :class:`RegisterReport` from a ``key -> string`` mapping or catalogue."""
    items = entries.items() if hasattr(entries, "items") else iter(entries)
    return RegisterReport(by_key={key: detect(text) for key, text in items})
