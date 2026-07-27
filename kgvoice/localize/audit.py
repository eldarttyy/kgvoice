"""Compose the catalogue, placeholder, and register checks into one audit.

:mod:`kgvoice.localize.catalog`, :mod:`kgvoice.localize.placeholders`, and
:mod:`kgvoice.localize.register` are each independently complete — this module
does not add checking logic of its own. It runs the three over the same
source/target pair and reads the results together, because a translator acting
on them wants one report, not three imports.

Deliberately out of scope: **entity preservation** (does a named entity survive
translation intact?), the third failure class named in the package docstring.
That check needs gold entity spans tied to the *source* text, which a free-form
UI-string catalogue does not carry — it is meaningful for corpus sentences
(:mod:`kgvoice.corpus`), not arbitrary product copy. Wiring it up would mean
inventing a source of truth this module cannot verify, so it is left undone
rather than approximated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from kgvoice.localize.catalog import Catalog, CatalogPair
from kgvoice.localize.placeholders import SuffixCollision, find_suffix_collisions, placeholder_counts
from kgvoice.localize.register import RegisterReport, audit_catalog


@dataclass
class PlaceholderIssue:
    """A key where the target's placeholders don't match the source's."""

    key: str
    source_text: str
    target_text: str
    missing: list[str] = field(default_factory=list)  # in source, dropped from target
    extra: list[str] = field(default_factory=list)  # in target, not in source

    @property
    def is_clean(self) -> bool:
        return not self.missing and not self.extra


@dataclass
class SuffixIssue:
    """A key whose target string hardcodes a case suffix onto a placeholder."""

    key: str
    target_text: str
    collisions: list[SuffixCollision] = field(default_factory=list)


@dataclass
class LocalizationAudit:
    """The full report over one source/target catalogue pair."""

    pair: CatalogPair
    placeholder_issues: list[PlaceholderIssue] = field(default_factory=list)
    suffix_issues: list[SuffixIssue] = field(default_factory=list)
    register: RegisterReport = field(default_factory=RegisterReport)

    @property
    def coverage(self) -> float:
        return self.pair.coverage()

    @property
    def missing_keys(self) -> list[str]:
        return self.pair.missing_keys

    @property
    def extra_keys(self) -> list[str]:
        return self.pair.extra_keys

    @property
    def dirty_placeholder_keys(self) -> list[str]:
        return [i.key for i in self.placeholder_issues if not i.is_clean]

    @property
    def suffix_collision_keys(self) -> list[str]:
        return [i.key for i in self.suffix_issues if i.collisions]

    @property
    def is_clean(self) -> bool:
        """True when nothing here would block shipping the catalogue."""
        return not (
            self.missing_keys
            or self.dirty_placeholder_keys
            or self.suffix_collision_keys
        ) and self.register.is_consistent

    def summary(self) -> dict:
        return {
            "keys": len(self.pair.source),
            "coverage": round(self.coverage, 4),
            "missing_keys": len(self.missing_keys),
            "extra_keys": len(self.extra_keys),
            "placeholder_mismatches": len(self.dirty_placeholder_keys),
            "suffix_collisions": len(self.suffix_collision_keys),
            "register_dominant": self.register.dominant,
            "register_minority_keys": len(self.register.minority_keys),
            "register_mixed_keys": len(self.register.mixed_keys),
            "clean": self.is_clean,
        }

    def as_dict(self) -> dict:
        return {
            "summary": self.summary(),
            "missing_keys": self.missing_keys,
            "extra_keys": self.extra_keys,
            "placeholder_issues": [
                {
                    "key": i.key,
                    "source": i.source_text,
                    "target": i.target_text,
                    "missing": i.missing,
                    "extra": i.extra,
                }
                for i in self.placeholder_issues
                if not i.is_clean
            ],
            "suffix_issues": [
                {
                    "key": i.key,
                    "target": i.target_text,
                    "collisions": [
                        {
                            "placeholder": c.placeholder.raw,
                            "suffix": c.suffix,
                            "template": c.label,
                            "suggestion": c.suggestion,
                            "wrong_for": c.wrong_for(),
                        }
                        for c in i.collisions
                    ],
                }
                for i in self.suffix_issues
            ],
            "register": {
                "dominant": self.register.dominant,
                "counts": dict(self.register.counts),
                "minority_keys": self.register.minority_keys,
                "mixed_keys": self.register.mixed_keys,
            },
        }

    def format(self) -> str:
        """A markdown report — safe to write straight to a ``.md`` file."""
        src_locale = self.pair.source.locale or "source"
        tgt_locale = self.pair.target.locale or "target"
        s = self.summary()
        lines = [
            f"# Localisation audit — {src_locale} -> {tgt_locale}",
            "",
            f"- keys: {s['keys']}",
            f"- coverage: {s['coverage']:.1%}",
            f"- missing translations: {s['missing_keys']}",
            f"- extra (stale) keys: {s['extra_keys']}",
            f"- placeholder mismatches: {s['placeholder_mismatches']}",
            f"- suffix collisions: {s['suffix_collisions']}",
            f"- register: {s['register_dominant'] or 'none detected'} "
            f"({'consistent' if not (s['register_minority_keys'] or s['register_mixed_keys']) else 'INCONSISTENT'})",
            f"- **overall: {'CLEAN' if s['clean'] else 'ISSUES FOUND'}**",
        ]

        if self.missing_keys:
            lines += ["", "## Missing translations", ""]
            lines += [f"- `{k}`" for k in self.missing_keys]

        if self.extra_keys:
            lines += ["", "## Stale keys (in target, not in source)", ""]
            lines += [f"- `{k}`" for k in self.extra_keys]

        dirty = [i for i in self.placeholder_issues if not i.is_clean]
        if dirty:
            lines += ["", "## Placeholder mismatches", ""]
            for i in dirty:
                lines.append(f"- `{i.key}`" + (f" — missing {i.missing}" if i.missing else ""))
                if i.extra:
                    lines[-1] += f", unexpected {i.extra}"
                lines.append(f"  - source: `{i.source_text}`")
                lines.append(f"  - target: `{i.target_text}`")

        if self.suffix_issues:
            lines += ["", "## Suffix collisions (Kyrgyz-specific)", ""]
            for i in self.suffix_issues:
                for c in i.collisions:
                    lines.append(f"- `{i.key}` — {c.placeholder.raw}{'-' if c.hyphenated else ''}{c.suffix} "
                                 f"read as **{c.label}**")
                    lines.append(f"  - target: `{i.target_text}`")
                    wrong = c.wrong_for()
                    if wrong:
                        examples = "; ".join(f"{v} -> *{p}* (want **{co}**)" for v, p, co in wrong)
                        lines.append(f"  - wrong for: {examples}")
                    lines.append(f"  - fix: {c.suggestion}")

        if self.register.mixed_keys or self.register.minority_keys:
            lines += ["", "## Register inconsistencies (сен/сиз)", ""]
            for k in self.register.mixed_keys:
                lines.append(f"- `{k}` — mixes both registers inside one string")
            for k in self.register.minority_keys:
                if k in self.register.mixed_keys:
                    continue
                v = self.register.by_key[k]
                lines.append(f"- `{k}` — uses **{v.register}**, catalogue is dominantly {self.register.dominant}")

        return "\n".join(lines) + "\n"


def audit(source: Catalog, target: Catalog) -> LocalizationAudit:
    """Run every check over one source/target catalogue pair."""
    pair = CatalogPair(source=source, target=target)
    placeholder_issues: list[PlaceholderIssue] = []
    suffix_issues: list[SuffixIssue] = []

    for key, src_text, tgt_text in pair.pairs():
        src_counts = placeholder_counts(src_text)
        tgt_counts = placeholder_counts(tgt_text)
        missing = sorted((src_counts - tgt_counts).elements())
        extra = sorted((tgt_counts - src_counts).elements())
        if missing or extra:
            placeholder_issues.append(PlaceholderIssue(key, src_text, tgt_text, missing, extra))

        collisions = find_suffix_collisions(tgt_text)
        if collisions:
            suffix_issues.append(SuffixIssue(key, tgt_text, collisions))

    return LocalizationAudit(
        pair=pair,
        placeholder_issues=placeholder_issues,
        suffix_issues=suffix_issues,
        register=audit_catalog(target),
    )


def audit_files(source_path: str | Path, target_path: str | Path) -> LocalizationAudit:
    """Load two catalogue files and audit them. Convenience wrapper for the CLI."""
    source = Catalog.load(source_path, locale=Path(source_path).stem)
    target = Catalog.load(target_path, locale=Path(target_path).stem)
    return audit(source, target)
