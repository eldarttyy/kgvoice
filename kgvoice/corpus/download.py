"""Fetch the KyrgyzNER release on demand.

The corpus is *not* vendored into this repository. It is published by Akyl-AI
under CC BY-NC-SA 4.0, which carries share-alike obligations that would
otherwise propagate to this MIT-licensed code. Downloading on first use keeps
the licence boundary clean and keeps the repo small.

Source: https://github.com/Akyl-AI/KyrgyzNER
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path

RAW_BASE = "https://raw.githubusercontent.com/Akyl-AI/KyrgyzNER/main/data"

DATA_FILES = {
    "train": "KyrgyzNER_TRAIN.jsonl.txt",
    "test": "KyrgyzNER_TEST.jsonl.gold.txt",
}

ATTRIBUTION = (
    "KyrgyzNER (Akyl-AI), CC BY-NC-SA 4.0 — https://github.com/Akyl-AI/KyrgyzNER\n"
    "1,499 news articles from 24.kg (2017-2022), annotated by 59 Kyrgyz linguists.\n"
)


def corpus_dir() -> Path:
    """Where corpus files are cached. Override with ``KGVOICE_DATA``."""
    env = os.environ.get("KGVOICE_DATA")
    if env:
        return Path(env).expanduser()
    return Path(
        os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    ).expanduser() / "kgvoice" / "KyrgyzNER"


def ensure_corpus(split: str = "train", *, force: bool = False) -> Path:
    """Return a local path to ``split``, downloading it if absent.

    Raises ``KeyError`` for an unknown split and ``RuntimeError`` if the
    download fails, so callers can print an actionable message instead of a
    stack trace.
    """
    if split not in DATA_FILES:
        raise KeyError(f"unknown split {split!r}; expected one of {sorted(DATA_FILES)}")

    target_dir = corpus_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / DATA_FILES[split]

    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest

    url = f"{RAW_BASE}/{DATA_FILES[split]}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - fixed https host
            tmp.write_bytes(resp.read())
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"could not download {url}: {exc}\n"
            f"Download it manually and place it at {dest}, "
            f"or set KGVOICE_DATA to a directory that already contains it."
        ) from exc

    tmp.replace(dest)
    (target_dir / "ATTRIBUTION.txt").write_text(ATTRIBUTION, encoding="utf-8")
    return dest


def ensure_all(force: bool = False) -> dict[str, Path]:
    return {name: ensure_corpus(name, force=force) for name in DATA_FILES}
