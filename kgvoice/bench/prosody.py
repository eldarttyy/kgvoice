"""A prosody and disfluency annotation schema.

An annotation scheme is only useful if two annotators applying it to the same
take agree, so this one is built from categories that are *decidable from the
audio* rather than from intent. "Sounded unnatural" is not a label; "stress on
the wrong syllable, expected 2 got 1" is.

Every tag carries the evidence needed to check it: which word, which syllable,
what was expected, what was heard. That is what makes the resulting file
reviewable, and what lets :mod:`kgvoice.bench.wer` and the phonology module be
scored against real judgements instead of against themselves.

Tag families
------------

``stress``      placement errors, keyed to a syllable index
``segment``     substitution/deletion/insertion of a phoneme
``length``      long vowel produced short, or vice versa
``boundary``    phrase break inserted or omitted
``intonation``  contour wrong for the sentence type
``disfluency``  filled pause, repetition, false start, prolongation
``quality``     audible non-linguistic problems (noise, plosive, clipping)

The disfluency set follows standard transcription practice so annotations here
are comparable with other corpora rather than idiosyncratic to this project.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

#: Tag vocabulary. Values are the human-facing description an annotator sees.
TAGS: dict[str, dict[str, str]] = {
    "stress": {
        "stress-misplaced": "Stress on the wrong syllable",
        "stress-missing": "No audible stress on a content word",
        "stress-extra": "Stress on a normally unstressed clitic or suffix",
    },
    "segment": {
        "seg-substitution": "One phoneme produced as another (ө->о, ү->у, ң->н)",
        "seg-deletion": "Phoneme omitted",
        "seg-insertion": "Phoneme added, e.g. epenthesis in a foreign cluster",
    },
    "length": {
        "length-shortened": "Long vowel produced short",
        "length-lengthened": "Short vowel produced long",
    },
    "boundary": {
        "boundary-missing": "Required phrase break not produced",
        "boundary-extra": "Break inserted where the syntax does not license one",
        "boundary-misplaced": "Break in the wrong place, splitting a constituent",
    },
    "intonation": {
        "intonation-flat": "Declarative contour where a question was required",
        "intonation-rising": "Question contour on a statement",
        "intonation-listing": "List intonation missing across coordinated items",
    },
    "disfluency": {
        "filled-pause": "Filled hesitation (ээ, мм)",
        "repetition": "Word or fragment repeated",
        "false-start": "Abandoned and restarted utterance",
        "prolongation": "Sound audibly drawn out",
        "self-correction": "Speaker corrects themselves mid-utterance",
    },
    "quality": {
        "background-noise": "Audible noise behind the speech",
        "plosive": "Popped p/b on the microphone",
        "clipping": "Audible distortion from overload",
        "mouth-noise": "Clicks, breaths, or handling noise",
    },
}

#: Flat tag -> family lookup.
TAG_FAMILY: dict[str, str] = {
    tag: family for family, tags in TAGS.items() for tag in tags
}

#: Tags that make a take unusable for training rather than merely imperfect.
BLOCKING_TAGS = frozenset({"clipping", "false-start", "seg-deletion"})

SEVERITIES = ("minor", "major", "blocking")


@dataclass
class ProsodyTag:
    """One annotation on one recording."""

    tag: str
    word_index: int
    word: str = ""
    syllable_index: int | None = None
    expected: str = ""
    observed: str = ""
    start_s: float | None = None
    end_s: float | None = None
    severity: str = "minor"
    note: str = ""

    def __post_init__(self) -> None:
        if self.tag not in TAG_FAMILY:
            raise ValueError(
                f"unknown tag {self.tag!r}; valid tags: {', '.join(sorted(TAG_FAMILY))}"
            )
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}, got {self.severity!r}")

    @property
    def family(self) -> str:
        return TAG_FAMILY[self.tag]

    @property
    def description(self) -> str:
        return TAGS[self.family][self.tag]

    @property
    def is_blocking(self) -> bool:
        return self.severity == "blocking" or self.tag in BLOCKING_TAGS

    def __str__(self) -> str:  # pragma: no cover - display helper
        loc = f"#{self.word_index}"
        if self.word:
            loc += f" {self.word!r}"
        if self.syllable_index is not None:
            loc += f" syl {self.syllable_index}"
        detail = ""
        if self.expected or self.observed:
            detail = f" (expected {self.expected!r}, heard {self.observed!r})"
        return f"[{self.severity}] {self.tag} at {loc}{detail}"


@dataclass
class ProsodyAnnotation:
    """All annotations for one utterance, plus the annotator's verdict."""

    utterance_id: str
    audio_path: str = ""
    transcript: str = ""
    annotator: str = ""
    tags: list[ProsodyTag] = field(default_factory=list)
    overall_rating: int | None = None  # 1-5, 5 = indistinguishable from a good human read
    notes: str = ""

    def add(self, tag: str, word_index: int, **kwargs) -> ProsodyTag:
        t = ProsodyTag(tag=tag, word_index=word_index, **kwargs)
        self.tags.append(t)
        return t

    @property
    def is_usable(self) -> bool:
        """False when any tag blocks use of this take for training."""
        return not any(t.is_blocking for t in self.tags)

    def by_family(self) -> dict[str, list[ProsodyTag]]:
        out: dict[str, list[ProsodyTag]] = {}
        for t in self.tags:
            out.setdefault(t.family, []).append(t)
        return out

    def counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for t in self.tags:
            c[t.tag] = c.get(t.tag, 0) + 1
        return dict(sorted(c.items(), key=lambda kv: -kv[1]))

    def as_dict(self) -> dict:
        return {
            "utterance_id": self.utterance_id,
            "audio_path": self.audio_path,
            "transcript": self.transcript,
            "annotator": self.annotator,
            "overall_rating": self.overall_rating,
            "is_usable": self.is_usable,
            "notes": self.notes,
            "tags": [asdict(t) for t in self.tags],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProsodyAnnotation":
        ann = cls(
            utterance_id=d["utterance_id"],
            audio_path=d.get("audio_path", ""),
            transcript=d.get("transcript", ""),
            annotator=d.get("annotator", ""),
            overall_rating=d.get("overall_rating"),
            notes=d.get("notes", ""),
        )
        ann.tags = [ProsodyTag(**t) for t in d.get("tags", [])]
        return ann


def save(annotations: Sequence[ProsodyAnnotation], path: str | Path) -> Path:
    """Write annotations as JSON Lines — one utterance per line, appendable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for a in annotations:
            fh.write(json.dumps(a.as_dict(), ensure_ascii=False) + "\n")
    return path


def load(path: str | Path) -> list[ProsodyAnnotation]:
    out = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(ProsodyAnnotation.from_dict(json.loads(line)))
    return out


def agreement(a: Sequence[ProsodyAnnotation], b: Sequence[ProsodyAnnotation]) -> dict:
    """Inter-annotator agreement between two passes over the same utterances.

    Reports exact agreement on the (word_index, tag) pairs, which is the strict
    reading, alongside agreement on which *words* were flagged at all — the
    looser reading that separates "we disagree about what is wrong here" from
    "we disagree about whether anything is wrong here". Both are worth knowing;
    quoting only the strict figure makes a scheme look worse than it is, and only
    the loose one makes it look better.
    """
    by_id_a = {x.utterance_id: x for x in a}
    by_id_b = {x.utterance_id: x for x in b}
    shared = sorted(set(by_id_a) & set(by_id_b))
    if not shared:
        return {"utterances": 0}

    exact_hits = exact_total = word_hits = word_total = 0
    for uid in shared:
        pairs_a = {(t.word_index, t.tag) for t in by_id_a[uid].tags}
        pairs_b = {(t.word_index, t.tag) for t in by_id_b[uid].tags}
        exact_hits += len(pairs_a & pairs_b)
        exact_total += len(pairs_a | pairs_b)
        words_a = {t.word_index for t in by_id_a[uid].tags}
        words_b = {t.word_index for t in by_id_b[uid].tags}
        word_hits += len(words_a & words_b)
        word_total += len(words_a | words_b)

    return {
        "utterances": len(shared),
        "exact_agreement": round(exact_hits / exact_total, 4) if exact_total else 1.0,
        "word_level_agreement": round(word_hits / word_total, 4) if word_total else 1.0,
        "tags_a": sum(len(by_id_a[u].tags) for u in shared),
        "tags_b": sum(len(by_id_b[u].tags) for u in shared),
    }


def summarize(annotations: Iterable[ProsodyAnnotation]) -> dict:
    """Aggregate tag counts and usability across a set of annotations."""
    anns = list(annotations)
    if not anns:
        return {"utterances": 0}
    tag_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    for a in anns:
        for t in a.tags:
            tag_counts[t.tag] = tag_counts.get(t.tag, 0) + 1
            family_counts[t.family] = family_counts.get(t.family, 0) + 1
    rated = [a.overall_rating for a in anns if a.overall_rating is not None]
    return {
        "utterances": len(anns),
        "usable": sum(1 for a in anns if a.is_usable),
        "total_tags": sum(len(a.tags) for a in anns),
        "mean_rating": round(sum(rated) / len(rated), 2) if rated else None,
        "by_family": dict(sorted(family_counts.items(), key=lambda kv: -kv[1])),
        "by_tag": dict(sorted(tag_counts.items(), key=lambda kv: -kv[1])),
    }
