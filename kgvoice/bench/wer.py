"""Word error rate, weighted toward the tokens that carry the information.

Plain WER treats every token alike, which hides the failure mode that matters
most in practice: a transcript can score 5% WER and still be useless because the
5% it got wrong were the minister's name, the district, and the number of
casualties. Function words are recoverable from context; named entities are not.

This module computes the standard alignment once and then reads three things off
it:

``wer``        the usual figure, over all tokens
``entity_wer`` the same figure restricted to reference tokens inside an entity
``by_label``   entity WER broken out per entity class

Alignment is Levenshtein with a full backtrace, so each reference token gets a
definite operation (``equal``, ``sub``, ``del``, ``ins``) and can be attributed
to the entity span it belongs to. Insertions are charged to the span they fall
inside, or to the following one at a boundary — otherwise a hallucinated word
before a name would count against nothing.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

_PUNCT = re.compile(r"[^\w\s-]", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize(text: str, *, keep_case: bool = False) -> str:
    """Normalise for scoring: NFC, strip punctuation, collapse whitespace.

    Case folding is on by default. Kyrgyz capitalisation carries no phonetic
    information, and an ASR system that returns lowercase should not be punished
    for it — but the flag exists because entity casing does matter when you are
    scoring a *recogniser's* entity detection rather than its transcription.
    """
    text = unicodedata.normalize("NFC", text)
    if not keep_case:
        text = text.lower()
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def tokenize(text: str, *, keep_case: bool = False) -> list[str]:
    norm = normalize(text, keep_case=keep_case)
    return norm.split() if norm else []


@dataclass(frozen=True)
class Op:
    """One alignment operation."""

    kind: str  # 'equal' | 'sub' | 'del' | 'ins'
    ref_index: int | None
    hyp_index: int | None
    ref_token: str = ""
    hyp_token: str = ""


def align(ref: Sequence[str], hyp: Sequence[str]) -> list[Op]:
    """Levenshtein alignment of ``hyp`` against ``ref``, with backtrace.

    Costs are the conventional 1/1/1. Ties are resolved substitution-first so
    that a same-length rewrite aligns positionally instead of drifting into a
    delete/insert pair, which keeps entity attribution stable.
    """
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j - 1] + cost, d[i - 1][j] + 1, d[i][j - 1] + 1)

    ops: list[Op] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            if d[i][j] == d[i - 1][j - 1] + cost:
                ops.append(
                    Op("equal" if cost == 0 else "sub", i - 1, j - 1, ref[i - 1], hyp[j - 1])
                )
                i, j = i - 1, j - 1
                continue
        if i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(Op("del", i - 1, None, ref[i - 1], ""))
            i -= 1
            continue
        ops.append(Op("ins", None, j - 1, "", hyp[j - 1]))
        j -= 1
    ops.reverse()
    return ops


@dataclass
class WERResult:
    """Overall and entity-restricted error rates for one utterance or set."""

    ref_tokens: int
    hyp_tokens: int
    substitutions: int
    deletions: int
    insertions: int
    entity_ref_tokens: int = 0
    entity_substitutions: int = 0
    entity_deletions: int = 0
    entity_insertions: int = 0
    by_label: dict[str, Counter] = field(default_factory=dict)
    ops: list[Op] = field(default_factory=list)

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        return self.errors / self.ref_tokens if self.ref_tokens else 0.0

    @property
    def entity_errors(self) -> int:
        return self.entity_substitutions + self.entity_deletions + self.entity_insertions

    @property
    def entity_wer(self) -> float:
        return self.entity_errors / self.entity_ref_tokens if self.entity_ref_tokens else 0.0

    @property
    def non_entity_wer(self) -> float:
        ref = self.ref_tokens - self.entity_ref_tokens
        return (self.errors - self.entity_errors) / ref if ref else 0.0

    def label_wer(self, label: str) -> float:
        c = self.by_label.get(label)
        if not c or not c["ref"]:
            return 0.0
        return (c["sub"] + c["del"] + c["ins"]) / c["ref"]

    def __add__(self, other: "WERResult") -> "WERResult":
        merged: dict[str, Counter] = {k: Counter(v) for k, v in self.by_label.items()}
        for k, v in other.by_label.items():
            merged.setdefault(k, Counter()).update(v)
        return WERResult(
            ref_tokens=self.ref_tokens + other.ref_tokens,
            hyp_tokens=self.hyp_tokens + other.hyp_tokens,
            substitutions=self.substitutions + other.substitutions,
            deletions=self.deletions + other.deletions,
            insertions=self.insertions + other.insertions,
            entity_ref_tokens=self.entity_ref_tokens + other.entity_ref_tokens,
            entity_substitutions=self.entity_substitutions + other.entity_substitutions,
            entity_deletions=self.entity_deletions + other.entity_deletions,
            entity_insertions=self.entity_insertions + other.entity_insertions,
            by_label=merged,
            ops=[],  # ops are per-utterance; aggregating them would be misleading
        )

    def format(self) -> str:  # pragma: no cover - display helper
        lines = [
            f"WER            {self.wer:7.2%}   ({self.errors}/{self.ref_tokens})",
            f"  entity       {self.entity_wer:7.2%}   "
            f"({self.entity_errors}/{self.entity_ref_tokens})",
            f"  non-entity   {self.non_entity_wer:7.2%}",
            f"  S/D/I        {self.substitutions}/{self.deletions}/{self.insertions}",
        ]
        if self.by_label:
            lines.append("")
            lines.append(f"{'label':<16}{'ref':>6}{'S':>5}{'D':>5}{'I':>5}{'WER':>9}")
            for label in sorted(self.by_label, key=lambda k: -self.by_label[k]["ref"]):
                c = self.by_label[label]
                lines.append(
                    f"{label:<16}{c['ref']:>6}{c['sub']:>5}{c['del']:>5}{c['ins']:>5}"
                    f"{self.label_wer(label):>9.2%}"
                )
        return "\n".join(lines)


def _label_at(entity_labels: Sequence[str | None], index: int) -> str | None:
    if 0 <= index < len(entity_labels):
        return entity_labels[index]
    return None


def score(
    reference: str | Sequence[str],
    hypothesis: str | Sequence[str],
    entity_labels: Sequence[str | None] | None = None,
    *,
    keep_case: bool = False,
) -> WERResult:
    """Score one hypothesis against one reference.

    ``entity_labels`` runs parallel to the *reference tokens*: entry ``i`` is the
    entity label of reference token ``i``, or ``None`` outside entities. Build it
    with :func:`labels_from_sentence`.
    """
    ref = (
        tokenize(reference, keep_case=keep_case)
        if isinstance(reference, str)
        else list(reference)
    )
    hyp = (
        tokenize(hypothesis, keep_case=keep_case)
        if isinstance(hypothesis, str)
        else list(hypothesis)
    )
    if entity_labels is None:
        entity_labels = [None] * len(ref)
    if len(entity_labels) != len(ref):
        raise ValueError(
            f"entity_labels has {len(entity_labels)} entries but reference has {len(ref)} tokens; "
            "they must be parallel"
        )

    ops = align(ref, hyp)
    result = WERResult(ref_tokens=len(ref), hyp_tokens=len(hyp), substitutions=0, deletions=0,
                       insertions=0, ops=ops)
    result.entity_ref_tokens = sum(1 for lab in entity_labels if lab)
    by_label: dict[str, Counter] = {}
    for lab in entity_labels:
        if lab:
            by_label.setdefault(lab, Counter())["ref"] += 1

    for pos, op in enumerate(ops):
        if op.kind == "equal":
            continue
        if op.kind == "sub":
            result.substitutions += 1
            lab = _label_at(entity_labels, op.ref_index)
            if lab:
                result.entity_substitutions += 1
                by_label.setdefault(lab, Counter())["sub"] += 1
        elif op.kind == "del":
            result.deletions += 1
            lab = _label_at(entity_labels, op.ref_index)
            if lab:
                result.entity_deletions += 1
                by_label.setdefault(lab, Counter())["del"] += 1
        else:  # insertion — no reference index of its own
            result.insertions += 1
            lab = _insertion_label(ops, pos, entity_labels)
            if lab:
                result.entity_insertions += 1
                by_label.setdefault(lab, Counter())["ins"] += 1

    result.by_label = by_label
    return result


def _insertion_label(
    ops: Sequence[Op], pos: int, entity_labels: Sequence[str | None]
) -> str | None:
    """Attribute an insertion to a surrounding entity span, if any.

    An insertion has no reference index, so it is charged to the entity of the
    nearest reference token on either side — preferring the following one, since
    a spurious word before a name reads as part of that name. Charging nothing
    would let a system hallucinate freely around entities at no cost.
    """
    after = next(
        (o.ref_index for o in ops[pos + 1 :] if o.ref_index is not None),
        None,
    )
    before = next(
        (o.ref_index for o in reversed(ops[:pos]) if o.ref_index is not None),
        None,
    )
    for idx in (after, before):
        if idx is not None:
            lab = _label_at(entity_labels, idx)
            if lab:
                return lab
    return None


def labels_from_sentence(sentence) -> tuple[list[str], list[str | None]]:
    """Reference tokens and their parallel entity labels, from a corpus sentence.

    Returns ``(tokens, labels)`` already normalised the same way :func:`score`
    normalises text, so the two line up. Multi-word corpus tokens are split, and
    their label is copied onto each resulting piece.
    """
    tokens: list[str] = []
    labels: list[str | None] = []
    entity_idx = sentence.entity_token_indices()
    span_label: dict[int, str] = {}
    for span in sentence.entities():
        for i in range(span.start, span.end):
            span_label[i] = span.label

    for i, tok in enumerate(sentence.tokens):
        pieces = tokenize(tok.text)
        if not pieces:
            continue
        label = span_label.get(i) if i in entity_idx else None
        for piece in pieces:
            tokens.append(piece)
            labels.append(label)
    return tokens, labels


def score_corpus(pairs: Iterable[tuple[Sequence[str], Sequence[str], Sequence[str | None]]]):
    """Aggregate :func:`score` over many utterances."""
    total = WERResult(0, 0, 0, 0, 0)
    for ref, hyp, labels in pairs:
        total = total + score(ref, hyp, labels)
    return total
