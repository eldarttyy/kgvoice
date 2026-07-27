"""Kyrgyz grapheme-to-phoneme conversion.

Kyrgyz Cyrillic is close to phonemic, so most of the work is in the handful of
context-sensitive rules that a letter-for-letter mapping gets wrong:

* **Dorsal backing.** ``к`` and ``г`` are uvular [q ʁ] beside back vowels and
  velar/palatal [k ɡ] ~ [c ɟ] beside front ones. ``кол`` [qol] vs ``көл`` [køl] —
  a minimal pair distinguished by *both* the vowel and the consonant.
* **Lateral velarisation.** ``л`` is [ɫ] in back contexts, [l] in front.
* **ж.** /d͡ʒ/ in native words, /ʒ/ in Russian borrowings — ``жыл`` [d͡ʒɯl]
  vs ``журнал`` [ʒurnɑl]. Decided by :mod:`kgvoice.phon.loanword`.
* **Long vowels** are digraphs: ``саат`` [sɑːt], not [sɑɑt].
* **Iotated letters.** ``я ю ё`` are glide + vowel; ``е`` is [je] word-initially.

Conditioning is *local* rather than word-level: the nearest vowel decides, not
the word's overall harmony class. For native stems the two are identical, but
borrowings are routinely disharmonic (``телевизор``), and there the local rule is
the one that matches how the word is actually said.

Two transcription depths are offered. Broad (default) gives phonemes suitable for
a lexicon or a TTS front end. Narrow adds allophonic detail — palatal [c ɟ],
palatalisation from ``ь``, nasal place assimilation, final devoicing — which is
what you want when annotating what a speaker actually produced.
"""

from __future__ import annotations

from dataclasses import dataclass

from kgvoice.phon.alphabet import (
    CONSONANTS,
    LONG_VOWELS,
    SIGNS,
    VOWELS,
    Vowel,
)
from kgvoice.phon.loanword import analyze as _loan_analyze
from kgvoice.phon.stress import StressResult, stress
from kgvoice.phon.syllable import syllable_spans

_BACK_DORSAL = {"к": "q", "г": "ʁ"}
_FRONT_DORSAL_BROAD = {"к": "k", "г": "ɡ"}
_FRONT_DORSAL_NARROW = {"к": "c", "г": "ɟ"}

_IOTATED = {"я": ("j", "ɑ"), "ю": ("j", "u"), "ё": ("j", "o")}

#: Nasal place assimilation targets (narrow transcription only).
_NASAL_ASSIM = {"к": "ŋ", "г": "ŋ", "q": "ŋ", "ʁ": "ŋ", "п": "m", "б": "m"}


@dataclass(frozen=True)
class Phone:
    """One output segment and the grapheme index it came from."""

    ipa: str
    source_index: int
    long: bool = False

    def __str__(self) -> str:  # pragma: no cover - display helper
        return self.ipa


def _vowel_positions(word: str) -> list[tuple[int, Vowel]]:
    return [(i, VOWELS[ch]) for i, ch in enumerate(word.lower()) if ch in VOWELS]


def _local_is_back(word: str, index: int, default: bool = True) -> bool:
    """Backness of the vowel nearest ``index``, preferring the following one.

    Kyrgyz suffix vowels agree with the stem, so looking rightward first matches
    the direction harmony actually spreads, and gives the right answer for a
    dorsal that opens a syllable (``кол`` -> the ``о`` conditions the ``к``).
    """
    positions = _vowel_positions(word)
    if not positions:
        return default
    after = [(i, v) for i, v in positions if i > index]
    before = [(i, v) for i, v in positions if i < index]
    if after and before:
        di_after = after[0][0] - index
        di_before = index - before[-1][0]
        chosen = after[0][1] if di_after <= di_before else before[-1][1]
    elif after:
        chosen = after[0][1]
    else:
        chosen = before[-1][1]
    return chosen.is_back


def phones(word: str, *, narrow: bool = False) -> list[Phone]:
    """Convert ``word`` to a list of :class:`Phone`."""
    w = word.lower()
    n = len(w)
    out: list[Phone] = []
    zh_fricative = _loan_analyze(w).is_loanword

    i = 0
    while i < n:
        ch = w[i]

        # Long vowel digraph.
        if i + 1 < n and w[i : i + 2] in LONG_VOWELS:
            out.append(Phone(LONG_VOWELS[w[i : i + 2]], i, long=True))
            i += 2
            continue

        # Iotated Russian vowels.
        if ch in _IOTATED:
            glide, vowel = _IOTATED[ch]
            out.append(Phone(glide, i))
            out.append(Phone(vowel, i))
            i += 1
            continue

        # 'е' is [je] word-initially, [e] elsewhere.
        if ch == "е":
            if i == 0:
                out.append(Phone("j", i))
            out.append(Phone("e", i))
            i += 1
            continue

        if ch in VOWELS:
            out.append(Phone(VOWELS[ch].ipa, i))
            i += 1
            continue

        # Soft/hard signs: no segment of their own.
        if ch in SIGNS:
            if narrow and ch == "ь" and out:
                prev = out[-1]
                out[-1] = Phone(prev.ipa + "ʲ", prev.source_index, prev.long)
            i += 1
            continue

        if ch in CONSONANTS:
            out.append(Phone(_consonant_ipa(w, i, ch, narrow, zh_fricative), i))
            i += 1
            continue

        # Anything else (digits, latin, punctuation) is passed through so the
        # caller can see what was not handled rather than losing it silently.
        out.append(Phone(ch, i))
        i += 1

    if narrow:
        out = _apply_narrow_rules(w, out)
    return out


def _consonant_ipa(word: str, i: int, ch: str, narrow: bool, zh_fricative: bool) -> str:
    if ch in _BACK_DORSAL:
        if _local_is_back(word, i):
            return _BACK_DORSAL[ch]
        return (_FRONT_DORSAL_NARROW if narrow else _FRONT_DORSAL_BROAD)[ch]
    if ch == "л":
        return "ɫ" if _local_is_back(word, i) else "l"
    if ch == "ж":
        return "ʒ" if zh_fricative else "d͡ʒ"
    return CONSONANTS[ch].ipa


def _apply_narrow_rules(word: str, out: list[Phone]) -> list[Phone]:
    """Post-lexical detail: nasal place assimilation and final devoicing."""
    result = list(out)

    # /n/ -> [ŋ]/[m] before a matching-place stop.
    for idx, ph in enumerate(result):
        if ph.ipa != "n" or idx + 1 >= len(result):
            continue
        nxt = result[idx + 1].ipa
        if nxt in ("q", "ʁ", "k", "ɡ", "c", "ɟ"):
            result[idx] = Phone("ŋ", ph.source_index, ph.long)
        elif nxt in ("p", "b"):
            result[idx] = Phone("m", ph.source_index, ph.long)

    # Word-final /z/ is commonly devoiced.
    if result and result[-1].ipa == "z":
        last = result[-1]
        result[-1] = Phone("s", last.source_index, last.long)

    return result


def transcribe(word: str, *, narrow: bool = False) -> str:
    """Broad or narrow IPA for ``word``, without syllable or stress marks."""
    return "".join(p.ipa for p in phones(word, narrow=narrow))


def ipa(
    word: str,
    *,
    narrow: bool = False,
    mark_stress: bool = True,
    syllable_sep: str = ".",
    brackets: bool = True,
) -> str:
    """Full IPA rendering with syllable boundaries and a stress mark.

    Stress is written ``ˈ`` before the stressed syllable. When stress cannot be
    determined — an unrecognised borrowing — no mark is emitted rather than a
    guessed one, and :func:`kgvoice.phon.stress.stress` will say why.

    >>> ipa("кыргызстан", brackets=False)
    'qɯr.ʁɯz.ˈstɑn'
    >>> ipa("көл", brackets=False)
    'ˈkøl'
    """
    spans = syllable_spans(word)
    if not spans:
        body = transcribe(word, narrow=narrow)
        return f"[{body}]" if brackets and body else body

    # Transcribe the whole word once, then slice phones into syllables by the
    # grapheme index they came from. Converting a syllable in isolation would
    # lose its conditioning environment — ``кол`` cut into ``ко`` + ``л`` still
    # needs the ``о`` to select [q] over [k].
    all_phones = phones(word, narrow=narrow)
    st: StressResult | None = stress(word) if mark_stress else None

    chunks = []
    for i, (start, end) in enumerate(spans):
        piece = "".join(p.ipa for p in all_phones if start <= p.source_index < end)
        if st is not None and st.index == i:
            piece = "ˈ" + piece
        chunks.append(piece)
    body = syllable_sep.join(chunks)
    return f"[{body}]" if brackets else body


def phoneme_count(word: str, *, narrow: bool = False) -> int:
    return len(phones(word, narrow=narrow))
