"""Recording manifests: what to read, and what came back.

A manifest is the contract between the person selecting sentences and the person
recording them. It carries, for each utterance: the prompt to read, the entity
spans that will be scored, the pronunciation guidance the phonology module
produced, and — once recorded — the path to the take and its QC verdict.

It is JSON, stable-ordered, and round-trips exactly. That matters because it is
the file a session is resumed from and the file two annotators diff.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from kgvoice.bench.select import SentenceScore
from kgvoice.bench.wer import labels_from_sentence
from kgvoice.phon.profile import profile_word


@dataclass
class EntityRef:
    """An entity span inside a prompt, as the scorer will see it."""

    text: str
    label: str
    start: int
    end: int
    ipa: str = ""
    difficulty: float = 0.0
    needs_review: bool = False


@dataclass
class PronunciationHint:
    """Guidance for one difficult word in a prompt."""

    word: str
    hyphenated: str
    ipa: str
    stress_marked: str
    stress_confidence: str
    difficulty: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class ManifestItem:
    """One utterance to record."""

    utterance_id: str
    text: str
    sent_id: str = ""
    n_tokens: int = 0
    score: float = 0.0
    entities: list[EntityRef] = field(default_factory=list)
    hints: list[PronunciationHint] = field(default_factory=list)
    #: Normalised reference tokens and their parallel entity labels, stored so
    #: that scoring a transcript needs only the manifest — not the corpus it was
    #: built from, and not a re-derivation of the tokenisation that might drift
    #: from the one used here.
    ref_tokens: list[str] = field(default_factory=list)
    ref_labels: list[str | None] = field(default_factory=list)
    audio_path: str = ""
    recorded_at: str = ""
    qc_passed: bool | None = None
    qc_problems: list[str] = field(default_factory=list)

    @property
    def is_recorded(self) -> bool:
        return bool(self.audio_path)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ManifestItem":
        return cls(
            utterance_id=d["utterance_id"],
            text=d["text"],
            sent_id=d.get("sent_id", ""),
            n_tokens=d.get("n_tokens", 0),
            score=d.get("score", 0.0),
            entities=[EntityRef(**e) for e in d.get("entities", [])],
            hints=[PronunciationHint(**h) for h in d.get("hints", [])],
            audio_path=d.get("audio_path", ""),
            recorded_at=d.get("recorded_at", ""),
            qc_passed=d.get("qc_passed"),
            qc_problems=d.get("qc_problems", []),
            ref_tokens=d.get("ref_tokens", []),
            ref_labels=d.get("ref_labels", []),
        )


@dataclass
class Manifest:
    """A recording session."""

    name: str
    created_at: str = ""
    source: str = "KyrgyzNER (Akyl-AI, CC BY-NC-SA 4.0)"
    spec: str = "default"
    items: list[ManifestItem] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    @property
    def recorded(self) -> list[ManifestItem]:
        return [i for i in self.items if i.is_recorded]

    @property
    def pending(self) -> list[ManifestItem]:
        return [i for i in self.items if not i.is_recorded]

    def by_id(self, utterance_id: str) -> ManifestItem | None:
        return next((i for i in self.items if i.utterance_id == utterance_id), None)

    def progress(self) -> dict:
        rec = self.recorded
        passed = [i for i in rec if i.qc_passed]
        return {
            "total": len(self.items),
            "recorded": len(rec),
            "qc_passed": len(passed),
            "qc_failed": len([i for i in rec if i.qc_passed is False]),
            "pending": len(self.pending),
            "percent_complete": round(100 * len(rec) / len(self.items), 1) if self.items else 0.0,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "created_at": self.created_at,
            "source": self.source,
            "spec": self.spec,
            "items": [i.as_dict() for i in self.items],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Manifest":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=d["name"],
            created_at=d.get("created_at", ""),
            source=d.get("source", ""),
            spec=d.get("spec", "default"),
            items=[ManifestItem.from_dict(i) for i in d.get("items", [])],
        )


def build(
    selected: Sequence[SentenceScore],
    *,
    name: str = "session",
    prefix: str = "kg",
    hint_threshold: float = 0.35,
    max_hints: int = 6,
) -> Manifest:
    """Turn selected sentences into a recording manifest.

    Only words at or above ``hint_threshold`` get a pronunciation hint. A prompt
    covered in guidance is a prompt nobody reads; the point is to flag the two or
    three words in a sentence that a reader would otherwise get wrong.
    """
    items: list[ManifestItem] = []
    for n, sc in enumerate(selected, start=1):
        sentence = sc.sentence
        tokens = [t.text for t in sentence.tokens]

        entities = []
        for span in sentence.entities():
            profs = [profile_word(t) for t in span.tokens if t.strip()]
            entities.append(
                EntityRef(
                    text=span.text,
                    label=span.label,
                    start=span.start,
                    end=span.end,
                    ipa=" ".join(p.ipa for p in profs),
                    difficulty=round(max((p.difficulty for p in profs), default=0.0), 3),
                    needs_review=any(p.needs_review for p in profs),
                )
            )

        hints: list[PronunciationHint] = []
        seen: set[str] = set()
        for tok in tokens:
            key = tok.lower()
            if not tok.isalpha() or len(tok) < 3 or key in seen:
                continue
            p = profile_word(tok)
            if p.difficulty < hint_threshold:
                continue
            seen.add(key)
            hints.append(
                PronunciationHint(
                    word=tok,
                    hyphenated=p.hyphenated,
                    ipa=p.ipa,
                    stress_marked=p.stress.marked(),
                    stress_confidence=p.stress.confidence,
                    difficulty=round(p.difficulty, 3),
                    reasons=p.reasons(),
                )
            )
        hints.sort(key=lambda h: -h.difficulty)

        ref_tokens, ref_labels = labels_from_sentence(sentence)
        items.append(
            ManifestItem(
                utterance_id=f"{prefix}-{n:04d}",
                text=sc.text,
                sent_id=sentence.sent_id,
                n_tokens=len(tokens),
                score=round(sc.total, 4),
                entities=entities,
                hints=hints[:max_hints],
                ref_tokens=ref_tokens,
                ref_labels=ref_labels,
            )
        )

    return Manifest(name=name, items=items)


def attach_recordings(
    manifest: Manifest, audio_dir: str | Path, *, spec_name: str = "default"
) -> Manifest:
    """Match ``<utterance_id>.wav`` files in ``audio_dir`` to manifest items and QC them.

    Import is local so that the manifest module stays usable — for planning a
    session, printing prompts — without NumPy present.
    """
    from kgvoice.bench.audio import RecordingSpec, analyze_file

    spec = RecordingSpec.strict() if spec_name == "strict" else RecordingSpec()
    audio_dir = Path(audio_dir)
    for item in manifest.items:
        candidate = audio_dir / f"{item.utterance_id}.wav"
        if not candidate.exists():
            continue
        item.audio_path = str(candidate)
        item.recorded_at = datetime.fromtimestamp(
            candidate.stat().st_mtime, tz=timezone.utc
        ).isoformat(timespec="seconds")
        try:
            qc = analyze_file(candidate, spec=spec)
            item.qc_passed = qc.passed
            item.qc_problems = qc.problems()
        except ValueError as exc:
            item.qc_passed = False
            item.qc_problems = [str(exc)]
    manifest.spec = spec_name
    return manifest


def write_prompts(manifest: Manifest, path: str | Path) -> Path:
    """Write a plain-text prompt sheet a reader can work from at the mic."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [f"# {manifest.name}", f"# {len(manifest)} utterances", ""]
    for item in manifest.items:
        lines.append(f"[{item.utterance_id}]")
        lines.append(item.text)
        if item.hints:
            lines.append("  pronunciation:")
            for h in item.hints:
                conf = "" if h.stress_confidence == "high" else f"  ({h.stress_confidence} stress)"
                lines.append(f"    {h.word}  ->  {h.stress_marked}  {h.ipa}{conf}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def iter_pending(manifests: Iterable[Manifest]):
    for m in manifests:
        yield from m.pending
