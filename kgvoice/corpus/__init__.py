"""KyrgyzNER corpus access."""

from kgvoice.corpus.conll import (
    Corpus,
    CorpusStats,
    EntitySpan,
    Sentence,
    Token,
    load_conll,
)
from kgvoice.corpus.download import DATA_FILES, corpus_dir, ensure_corpus

__all__ = [
    "Corpus",
    "CorpusStats",
    "EntitySpan",
    "Sentence",
    "Token",
    "load_conll",
    "DATA_FILES",
    "corpus_dir",
    "ensure_corpus",
]
