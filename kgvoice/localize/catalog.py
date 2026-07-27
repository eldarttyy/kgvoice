"""String catalogues: loading, flattening, and pairing source with target.

A localisation catalogue is a key/string map, and every framework spells it
differently — flat JSON, nested namespaces, ICU-style objects with ``message``
and ``description`` keys. This module normalises all of them to an ordered
``key -> string`` mapping so the checks in :mod:`kgvoice.localize.audit` never
have to care which tool produced the file.

Key order is preserved. A localisation report that reorders the catalogue is
harder to diff against the file it describes, and diffing is how these reports
get used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

#: Keys that carry the translatable string inside an object-valued entry.
#: ``message`` is Chrome/WebExtension style, ``defaultMessage`` is react-intl,
#: ``string`` and ``value`` are common in hand-rolled formats.
_MESSAGE_KEYS = ("message", "defaultMessage", "string", "value", "text")

#: Keys that carry translator-facing notes rather than translatable text.
_METADATA_KEYS = ("description", "comment", "context", "placeholders", "note")


@dataclass
class Catalog:
    """An ordered ``key -> string`` catalogue with its provenance."""

    entries: dict[str, str] = field(default_factory=dict)
    path: str = ""
    locale: str = ""

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[tuple[str, str]]:
        return iter(self.entries.items())

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def __getitem__(self, key: str) -> str:
        return self.entries[key]

    def get(self, key: str, default: str = "") -> str:
        return self.entries.get(key, default)

    @property
    def keys(self) -> list[str]:
        return list(self.entries)

    @classmethod
    def from_dict(cls, data: dict, *, path: str = "", locale: str = "") -> "Catalog":
        return cls(entries=_flatten(data), path=path, locale=locale)

    @classmethod
    def load(cls, path: str | Path, *, locale: str = "") -> "Catalog":
        """Read a JSON catalogue.

        Raises ``ValueError`` with the file name attached — a localisation run
        typically loads several catalogues, and a bare ``JSONDecodeError`` does
        not say which one was malformed.
        """
        p = Path(path)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p.name} is not valid JSON: {exc}") from exc
        except OSError as exc:
            raise ValueError(f"could not read {p}: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError(
                f"{p.name} must contain a JSON object of keys to strings, "
                f"not {type(data).__name__}"
            )
        return cls.from_dict(data, path=str(p), locale=locale or p.stem)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return p


def _flatten(data: dict, prefix: str = "") -> dict[str, str]:
    """Flatten a nested catalogue into dotted keys.

    Object-valued entries are treated as message wrappers when they carry a
    recognised message key, and as namespaces otherwise. Getting this wrong in
    either direction is silent data loss, so the distinction is made on the
    presence of a *string-valued* message key rather than on the key alone.
    """
    out: dict[str, str] = {}
    for key, value in data.items():
        full = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, str):
            out[full] = value
        elif isinstance(value, dict):
            message = next(
                (value[k] for k in _MESSAGE_KEYS if isinstance(value.get(k), str)), None
            )
            if message is not None:
                out[full] = message
            else:
                nested = {k: v for k, v in value.items() if k not in _METADATA_KEYS}
                out.update(_flatten(nested, full))
        elif isinstance(value, (int, float, bool)) or value is None:
            continue  # not translatable content
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, str):
                    out[f"{full}[{i}]"] = item
    return out


@dataclass
class CatalogPair:
    """A source catalogue and its translation, aligned by key."""

    source: Catalog
    target: Catalog

    @property
    def shared_keys(self) -> list[str]:
        """Keys present in both, in source order."""
        return [k for k in self.source.keys if k in self.target]

    @property
    def missing_keys(self) -> list[str]:
        """In the source but not translated."""
        return [k for k in self.source.keys if k not in self.target]

    @property
    def extra_keys(self) -> list[str]:
        """In the translation but not in the source — usually a stale key."""
        return [k for k in self.target.keys if k not in self.source]

    def pairs(self) -> Iterator[tuple[str, str, str]]:
        """``(key, source_string, target_string)`` for every shared key."""
        for key in self.shared_keys:
            yield key, self.source[key], self.target[key]

    def coverage(self) -> float:
        return len(self.shared_keys) / len(self.source) if len(self.source) else 1.0
