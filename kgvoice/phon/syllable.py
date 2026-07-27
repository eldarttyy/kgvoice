"""Kyrgyz syllabification.

Kyrgyz permits V, VC, VCC, CV, CVC and CVCC syllables. Crucially it does **not**
permit complex onsets in native vocabulary, so the medial rule is simple: of a
run of consonants between two nuclei, exactly one goes to the following onset
and the rest close the preceding syllable (``ал-ма``, ``кыр-гыз``, ``эрк-түү``).

The one place this needs care is unassimilated borrowings, which do carry
complex onsets (``спорт``, ``трактор``, ``прог-рам-ма``). Word-initial clusters
are therefore kept together as an onset rather than being forced to fit the
native template — the alternative is a phantom epenthetic syllable, which is
precisely the error a Kyrgyz speaker reading a loanword aloud may or may not
make, and the profiler wants to flag it, not bake it in.

Long vowels are written as digraphs (``аа``, ``үү``) and form a single nucleus.
"""

from __future__ import annotations

from dataclasses import dataclass

from kgvoice.phon.alphabet import LONG_VOWELS, SIGNS, VOWEL_LETTERS, is_consonant

#: Two-consonant sequences that can open a syllable in borrowed vocabulary.
#: Used only to break up a medial run of three or more consonants, where the
#: native one-consonant-onset rule would otherwise force an unpronounceable
#: CVCC coda (``кыр-гызс-тан``). Kept here rather than imported from
#: :mod:`kgvoice.phon.loanword` because that module syllabifies.
LICIT_COMPLEX_ONSETS = frozenset(
    {
        "ст", "сп", "ск", "сл", "см", "сн", "св", "тр", "пр", "кр", "гр",
        "бр", "др", "фр", "хр", "пл", "кл", "гл", "бл", "фл", "шк", "шт",
        "шп", "зд", "зн", "тв", "кв", "дв", "гв", "пс", "вл", "вр",
    }
)


@dataclass(frozen=True)
class Syllable:
    onset: str
    nucleus: str
    coda: str

    @property
    def text(self) -> str:
        return self.onset + self.nucleus + self.coda

    @property
    def is_long(self) -> bool:
        """True when the nucleus is a written long vowel."""
        return self.nucleus.lower() in LONG_VOWELS

    @property
    def is_open(self) -> bool:
        return not self.coda

    @property
    def shape(self) -> str:
        """CV skeleton, e.g. ``'CVC'``."""
        return "C" * len(self.onset) + "V" + "C" * len(self.coda)

    def __str__(self) -> str:  # pragma: no cover - display helper
        return self.text


def _nuclei_spans(word: str) -> list[tuple[int, int]]:
    """Index spans of each vowel nucleus, merging written long vowels."""
    spans: list[tuple[int, int]] = []
    i = 0
    lw = word.lower()
    n = len(word)
    while i < n:
        if lw[i] in VOWEL_LETTERS:
            if i + 1 < n and lw[i : i + 2] in LONG_VOWELS:
                spans.append((i, i + 2))
                i += 2
            else:
                spans.append((i, i + 1))
                i += 1
        else:
            i += 1
    return spans


def syllabify(word: str) -> list[Syllable]:
    """Split ``word`` into syllables.

    A token with no vowel (an initialism such as ``КР``, or punctuation) yields
    an empty list; callers should treat that as "not syllabifiable" rather than
    as a one-syllable word.
    """
    if not word:
        return []
    spans = _nuclei_spans(word)
    if not spans:
        return []

    syllables: list[Syllable] = []
    for idx, (start, end) in enumerate(spans):
        # Onset: consonants between the previous nucleus and this one.
        prev_end = spans[idx - 1][1] if idx else 0
        cluster = word[prev_end:start]
        if idx == 0:
            onset = cluster  # word-initial cluster stays whole (loanwords)
            carry_coda = ""
        elif len(cluster) <= 1:
            onset = cluster
            carry_coda = ""
        elif len(cluster) >= 3 and cluster[-2:].lower() in LICIT_COMPLEX_ONSETS:
            # Maximal onset, capped by what Kyrgyz can actually pronounce. Only
            # applies to runs of 3+, so native ``дос-тор`` is untouched while
            # ``кыр-гыз-стан`` avoids a ``гызс`` coda.
            onset = cluster[-2:]
            carry_coda = cluster[:-2]
        else:
            onset = cluster[-1]
            carry_coda = cluster[:-1]

        if carry_coda and syllables:
            prev = syllables[-1]
            syllables[-1] = Syllable(prev.onset, prev.nucleus, prev.coda + carry_coda)

        nucleus = word[start:end]
        # Coda of the final syllable: everything after the last nucleus.
        coda = word[end:] if idx == len(spans) - 1 else ""
        syllables.append(Syllable(onset, nucleus, coda))

    return syllables


def nuclei(word: str) -> list[str]:
    """Vowel nuclei of ``word``, with long digraphs kept whole.

    ``'алуу'`` -> ``['а', 'уу']``, not ``['а', 'у', 'у']``. Any analysis that
    counts vowels — harmony checking above all — has to go through this, because
    treating a long vowel as two short ones invents a vowel sequence that is not
    there.
    """
    return [word[s:e] for s, e in _nuclei_spans(word)]


def syllable_spans(word: str) -> list[tuple[int, int]]:
    """Character offsets ``(start, end)`` of each syllable in ``word``.

    Syllabification is lossless — concatenating the syllables reproduces the
    input — so offsets can be accumulated. Callers need this to line syllables up
    with phones, which cannot be done by searching for the syllable text (a word
    like ``катарлар`` contains the same syllable twice).
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    for syl in syllabify(word):
        spans.append((pos, pos + len(syl.text)))
        pos += len(syl.text)
    return spans


def syllable_count(word: str) -> int:
    """Number of syllables; ``0`` when the token has no vowel."""
    return len(_nuclei_spans(word))


def hyphenate(word: str, sep: str = "-") -> str:
    """``'кыргызстан'`` -> ``'кыр-гыз-стан'``.

    Falls back to the input unchanged for vowel-less tokens.
    """
    syls = syllabify(word)
    if not syls:
        return word
    return sep.join(s.text for s in syls)


def strip_signs(word: str) -> str:
    """Drop ъ/ь, which carry no syllabic weight."""
    return "".join(ch for ch in word if ch.lower() not in SIGNS)


def has_complex_onset(word: str) -> bool:
    """True when any syllable begins with more than one consonant.

    In practice this is a loanword diagnostic: a native Kyrgyz stem cannot do it.
    """
    return any(len(s.onset) > 1 for s in syllabify(word))


def final_cluster_size(word: str) -> int:
    """Length of the word-final consonant cluster."""
    n = 0
    for ch in reversed(word):
        if is_consonant(ch):
            n += 1
        else:
            break
    return n
