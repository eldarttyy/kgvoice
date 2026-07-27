"""Loader for the KyrgyzNER CoNLL-2003 style release.

File shape (tab separated, four columns)::

    # 1 0 1
    Жалал	-	-	B-LOCATION
    -	-	-	I-LOCATION
    ...

Two properties of this corpus routinely break naive loaders and are handled here:

1. A *token* may contain a space (``"басма сөз"``, ``"орун басары"``). The
   separator is the tab, never whitespace, so the file must not be ``.split()``.
2. A token may be a bare ``"-"``. Hyphenated place names are tokenised as
   ``Жалал / - / Абад``, so a line beginning with ``-`` is data, not a comment.

Entity spans are decoded from BIO tags defensively: an ``I-X`` that opens a span
(no preceding ``B-X``) is treated as the start of a new span rather than dropped,
and a label switch inside a span closes it. Both occur in the released files.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

_HEADER = re.compile(r"^#\s+\S+")


@dataclass(frozen=True)
class Token:
    """A single corpus token and its BIO tag."""

    text: str
    tag: str = "O"

    @property
    def label(self) -> str | None:
        """The entity label without the BIO prefix, or ``None`` outside entities."""
        if self.tag in ("O", "", "-"):
            return None
        return self.tag[2:] if self.tag[:2] in ("B-", "I-") else self.tag


@dataclass(frozen=True)
class EntitySpan:
    """A contiguous run of tokens carrying one entity label."""

    label: str
    start: int
    end: int  # exclusive
    tokens: tuple[str, ...]
    sent_id: str = ""

    @property
    def text(self) -> str:
        return " ".join(self.tokens)

    @property
    def length(self) -> int:
        return self.end - self.start

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"{self.text} [{self.label}]"


@dataclass
class Sentence:
    """One sentence: its tokens, plus lazily decoded entity spans."""

    sent_id: str
    tokens: list[Token] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Detokenised surface string.

        Punctuation is re-attached to the preceding token so the result reads as
        a natural recording prompt rather than a token dump.
        """
        out: list[str] = []
        for tok in self.tokens:
            t = tok.text
            if out and (t in _CLOSE_PUNCT or (len(t) == 1 and t in ".,!?;:%)»]")):
                out[-1] += t
            elif out and out[-1].endswith(tuple("(«[")):
                out[-1] += t
            else:
                out.append(t)
        return " ".join(out)

    @property
    def words(self) -> list[str]:
        return [t.text for t in self.tokens]

    def entities(self) -> list[EntitySpan]:
        """Decode BIO tags into spans."""
        spans: list[EntitySpan] = []
        cur_label: str | None = None
        cur_start = 0
        for i, tok in enumerate(self.tokens):
            tag = tok.tag
            prefix, label = (tag[:1], tok.label) if tag not in ("O", "", "-") else ("O", None)
            starts_new = prefix == "B" or (
                prefix == "I" and (cur_label is None or label != cur_label)
            )
            if cur_label is not None and (label is None or starts_new):
                spans.append(self._span(cur_label, cur_start, i))
                cur_label = None
            if label is not None and starts_new:
                cur_label, cur_start = label, i
        if cur_label is not None:
            spans.append(self._span(cur_label, cur_start, len(self.tokens)))
        return spans

    def _span(self, label: str, start: int, end: int) -> EntitySpan:
        return EntitySpan(
            label=label,
            start=start,
            end=end,
            tokens=tuple(t.text for t in self.tokens[start:end]),
            sent_id=self.sent_id,
        )

    def entity_token_indices(self) -> set[int]:
        """Indices of every token that falls inside any entity span."""
        return {i for s in self.entities() for i in range(s.start, s.end)}

    def __len__(self) -> int:
        return len(self.tokens)


_CLOSE_PUNCT = {".", ",", "!", "?", ";", ":", "%", ")", "»", "]", "...", "…"}


def load_conll(path: str | Path) -> list[Sentence]:
    """Parse one KyrgyzNER file into sentences."""
    path = Path(path)
    sentences: list[Sentence] = []
    current: Sentence | None = None
    auto_id = 0

    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip():
                if current is not None and current.tokens:
                    sentences.append(current)
                current = None
                continue
            # A header is '# ...' with no tab; a token line always has tabs.
            if "\t" not in line and _HEADER.match(line):
                if current is not None and current.tokens:
                    sentences.append(current)
                current = Sentence(sent_id=line[1:].strip())
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            text, tag = parts[0], parts[-1].strip() or "O"
            if current is None:
                auto_id += 1
                current = Sentence(sent_id=f"auto-{auto_id}")
            current.tokens.append(Token(text=text, tag=tag))

    if current is not None and current.tokens:
        sentences.append(current)
    return sentences


@dataclass
class CorpusStats:
    sentences: int
    tokens: int
    entities: int
    label_counts: Counter
    entity_tokens: int

    @property
    def entity_token_ratio(self) -> float:
        return self.entity_tokens / self.tokens if self.tokens else 0.0

    def format(self, top: int = 30) -> str:  # pragma: no cover - display helper
        lines = [
            f"sentences      {self.sentences:>8,}",
            f"tokens         {self.tokens:>8,}",
            f"entity spans   {self.entities:>8,}",
            f"entity tokens  {self.entity_tokens:>8,}  ({self.entity_token_ratio:.1%} of tokens)",
            f"labels         {len(self.label_counts):>8,}",
            "",
            f"{'label':<16}{'spans':>8}{'share':>9}",
        ]
        for label, n in self.label_counts.most_common(top):
            lines.append(f"{label:<16}{n:>8,}{n / self.entities:>9.1%}")
        return "\n".join(lines)


class Corpus:
    """A collection of sentences with entity-aware helpers."""

    def __init__(self, sentences: Sequence[Sentence], name: str = "corpus"):
        self.sentences = list(sentences)
        self.name = name

    @classmethod
    def from_files(cls, *paths: str | Path, name: str = "corpus") -> "Corpus":
        sents: list[Sentence] = []
        for p in paths:
            sents.extend(load_conll(p))
        return cls(sents, name=name)

    def __len__(self) -> int:
        return len(self.sentences)

    def __iter__(self) -> Iterator[Sentence]:
        return iter(self.sentences)

    def entities(self) -> Iterator[EntitySpan]:
        for s in self.sentences:
            yield from s.entities()

    def tokens(self) -> Iterator[Token]:
        for s in self.sentences:
            yield from s.tokens

    def labels(self) -> list[str]:
        return sorted({e.label for e in self.entities()})

    def filter_label(self, *labels: str) -> Iterable[EntitySpan]:
        wanted = set(labels)
        return (e for e in self.entities() if e.label in wanted)

    def stats(self) -> CorpusStats:
        label_counts: Counter = Counter()
        n_entities = 0
        n_entity_tokens = 0
        n_tokens = 0
        for s in self.sentences:
            n_tokens += len(s.tokens)
            spans = s.entities()
            n_entities += len(spans)
            for sp in spans:
                label_counts[sp.label] += 1
                n_entity_tokens += sp.length
        return CorpusStats(
            sentences=len(self.sentences),
            tokens=n_tokens,
            entities=n_entities,
            label_counts=label_counts,
            entity_tokens=n_entity_tokens,
        )

    def vocabulary(self, lowercase: bool = True) -> Counter:
        c: Counter = Counter()
        for t in self.tokens():
            c[t.text.lower() if lowercase else t.text] += 1
        return c
