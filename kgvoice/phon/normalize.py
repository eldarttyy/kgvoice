"""Kyrgyz text normalisation — the TTS front end, written out.

This is the layer that turns written text into the words a speaker actually
says, and it is the layer Kyrgyz TTS currently does without. `AkylAI's tts-mini
<https://github.com/Akyl-AI/tts-mini>`_ states the position plainly in its
README:

    The text preprocessing does not include functionality for processing
    abbreviations and contractions; however, the built-in phonemizer can
    transcribe numbers, but to avoid errors, it is better to write numbers in
    words.

"Write the numbers in words" is a workable instruction for a curated recording
script and no help at all for running text. It also happens to be the single
most common normalisation need in Kyrgyz news: ``MEASURE`` is the largest entity
class in KyrgyzNER at 22.1% of spans, ahead of ``PERSON`` and ``LOCATION``.

Three properties make Kyrgyz numerals more than a lookup table:

1. **Ordinals are harmonic.** The suffix is an archiphoneme, not a string:
   ``бир`` → ``биринчи`` but ``тогуз`` → ``тогузунчу`` and ``үч`` → ``үчүнчү``.
   This module derives them with :func:`kgvoice.phon.harmony.realize_for_stem`
   rather than hardcoding a table, so the ordinals and the rest of the package
   are answerable to the same rule.
2. **Only the final word takes the suffix.** ``2024`` → ``эки миң жыйырма
   төртүнчү``, not ``*экинчи миңинчи…``.
3. **Scale words drop their unit.** ``миң`` and ``жүз`` stand alone for 1000 and
   100, while ``миллион`` requires ``бир``.

What this module deliberately does **not** do is guess. Ranges, bare years in
ambiguous contexts, and unknown abbreviations are reported through
:func:`issues` instead of being silently expanded — a wrong expansion is worse
than an unexpanded token, because it is confidently wrong and no one reviews it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from kgvoice.phon.alphabet import is_vowel
from kgvoice.phon.harmony import realize_for_stem

# --------------------------------------------------------------------------
# Numerals
# --------------------------------------------------------------------------

ZERO = "нөл"

UNITS: dict[int, str] = {
    1: "бир",
    2: "эки",
    3: "үч",
    4: "төрт",
    5: "беш",
    6: "алты",
    7: "жети",
    8: "сегиз",
    9: "тогуз",
}

TENS: dict[int, str] = {
    10: "он",
    20: "жыйырма",
    30: "отуз",
    40: "кырк",
    50: "элүү",
    60: "алтымыш",
    70: "жетимиш",
    80: "сексен",
    90: "токсон",
}

#: Scale words, largest first. ``needs_unit`` marks the scales that require an
#: explicit ``бир`` — you say ``бир миллион`` but simply ``миң`` and ``жүз``.
SCALES: tuple[tuple[int, str, bool], ...] = (
    (10**9, "миллиард", True),
    (10**6, "миллион", True),
    (10**3, "миң", False),
    (10**2, "жүз", False),
)


def cardinal(n: int) -> str:
    """Spell out ``n`` in Kyrgyz.

    >>> cardinal(0)
    'нөл'
    >>> cardinal(135)
    'жүз отуз беш'
    >>> cardinal(2024)
    'эки миң жыйырма төрт'
    >>> cardinal(1_000_000)
    'бир миллион'
    """
    if n < 0:
        return f"минус {cardinal(-n)}"
    if n == 0:
        return ZERO

    parts: list[str] = []
    for value, word, needs_unit in SCALES:
        if n >= value:
            count, n = divmod(n, value)
            if count > 1 or needs_unit:
                parts.append(cardinal(count))
            parts.append(word)
    if n >= 10:
        ten, n = divmod(n, 10)
        parts.append(TENS[ten * 10])
    if n:
        parts.append(UNITS[n])
    return " ".join(parts)


def ordinal_suffix(stem: str) -> str:
    """The ordinal suffix as realised after ``stem``.

    The vowel is an archiphoneme resolved by vowel harmony, and a stem ending in
    a vowel drops the suffix-initial one:

    >>> ordinal_suffix("бир"), ordinal_suffix("тогуз"), ordinal_suffix("эки")
    ('инчи', 'унчу', 'нчи')
    """
    v = realize_for_stem("I", stem)
    return f"нч{v}" if is_vowel(stem[-1]) else f"{v}нч{v}"


def ordinal(n: int) -> str:
    """Spell out ``n`` as a Kyrgyz ordinal.

    Only the final word of the cardinal carries the suffix.

    >>> ordinal(1), ordinal(3), ordinal(9)
    ('биринчи', 'үчүнчү', 'тогузунчу')
    >>> ordinal(2024)
    'эки миң жыйырма төртүнчү'
    """
    words = cardinal(n).split()
    words[-1] += ordinal_suffix(words[-1])
    return " ".join(words)


# --------------------------------------------------------------------------
# Lexical inventories
# --------------------------------------------------------------------------

MONTHS: dict[str, str] = {
    "январь": "январь",
    "февраль": "февраль",
    "март": "март",
    "апрель": "апрель",
    "май": "май",
    "июнь": "июнь",
    "июль": "июль",
    "август": "август",
    "сентябрь": "сентябрь",
    "октябрь": "октябрь",
    "ноябрь": "ноябрь",
    "декабрь": "декабрь",
}

MONTH_BY_NUMBER: dict[int, str] = dict(enumerate(MONTHS, start=1))

#: Third-person possessive of each month, as used in spoken dates ("the 1st of
#: September" -> ``биринчи сентябры``). A stem-final soft sign drops before the
#: suffix. Entered explicitly rather than derived, because the soft-sign stems
#: and the ``я``-containing stems do not both fall out of vowel harmony cleanly.
#:
#: .. note:: **Needs native review.** These twelve forms are the one table in
#:    this module not derived from a rule or lifted from a cited source.
MONTH_POSSESSIVE: dict[str, str] = {
    "январь": "январы",
    "февраль": "февралы",
    "март": "марты",
    "апрель": "апрели",
    "май": "майы",
    "июнь": "июну",
    "июль": "июлу",
    "август": "августу",
    "сентябрь": "сентябры",
    "октябрь": "октябры",
    "ноябрь": "ноябры",
    "декабрь": "декабры",
}

#: Abbreviations that expand to a fixed phrase. Entered by hand and kept small
#: on the same principle as :mod:`kgvoice.phon.lexicon`: a missing entry is
#: reported, a wrong entry is spoken.
ABBREVIATIONS: dict[str, str] = {
    "кр": "Кыргыз Республикасы",
    "жк": "Жогорку Кеңеш",
    "акш": "Америка Кошмо Штаттары",
    "буу": "Бириккен Улуттар Уюму",
    "ммк": "массалык маалымат каражаттары",
    "ичм": "Ички иштер министрлиги",
    "б.а.": "башкача айтканда",
    "ж.б.": "жана башкалар",
    "т.а.": "тактап айтканда",
    "ж.у.с.": "жана ушул сыяктуу",
}

#: Symbols read as words.
SYMBOLS: dict[str, str] = {
    "%": "пайыз",
    "№": "номер",
    "€": "евро",
    "$": "доллар",
    "₽": "рубль",
}

#: Units that follow a number and are already Kyrgyz words; listed so that
#: :func:`issues` does not flag them as unknown tokens.
UNITS_OF_MEASURE: frozenset[str] = frozenset(
    {"сом", "тыйын", "км", "м", "см", "мм", "кг", "г", "т", "л", "саат", "мүнөт", "секунд"}
)

# --------------------------------------------------------------------------
# Patterns
# --------------------------------------------------------------------------

_INT = re.compile(r"(?<![\d.,:])(\d{1,12})(?![\d.,:])")
_ORDINAL_DASH = re.compile(r"(?<!\d)(\d{1,4})-(?=[а-яёңөүA-Za-zӨҮҢ])")
_DATE_DOTTED = re.compile(r"(?<!\d)(\d{1,2})\.(\d{1,2})\.(\d{4})(?!\d)")
_TIME = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)")
#: A range must not be silently expanded, so it is matched before anything else
#: and held aside. Note the digits-then-dash-then-digits shape distinguishes it
#: from the ordinal dash (``16-май``), which is followed by a letter.
_RANGE = re.compile(r"(?<!\d)(\d{1,4})\s*[–—-]\s*(\d{1,4})(?!\d)")
_PERCENT = re.compile(r"(\d+)\s*%")
_DECIMAL = re.compile(r"(?<!\d)(\d+)[.,](\d+)(?!\d)")


# --------------------------------------------------------------------------
# Issue reporting
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Issue:
    """One thing in the input that a front end should not silently guess at."""

    kind: str  # 'range' | 'unknown-abbreviation' | 'latin' | 'decimal' | 'bare-year'
    token: str
    note: str

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"[{self.kind}] {self.token} — {self.note}"


_LATIN = re.compile(r"[A-Za-z]{2,}")
#: An acronym may carry a Kyrgyz case suffix in lowercase (``БГУда``), so the
#: run of capitals is bounded by "not another capital" rather than a word break.
_ABBREV_CANDIDATE = re.compile(r"(?<![А-ЯЁӨҮҢа-яёөүң])([А-ЯЁӨҮҢ]{2,6})(?![А-ЯЁӨҮҢ])")


def issues(text: str) -> list[Issue]:
    """Report tokens that :func:`expand` will not or cannot resolve.

    This is the failure-corpus hook: run it over a body of text and the result
    is a worklist of exactly what the front end still gets wrong, rather than a
    silent stream of mispronunciations.
    """
    found: list[Issue] = []

    for m in _RANGE.finditer(text):
        found.append(
            Issue(
                "range",
                m.group(0),
                "a numeric range needs case marking (-дан ... -га чейин) that "
                "depends on context; not expanded",
            )
        )
    for m in _DECIMAL.finditer(text):
        found.append(
            Issue(
                "decimal",
                m.group(0),
                "decimal fraction — read as бүтүн/ондон depending on register; not expanded",
            )
        )
    for m in _ABBREV_CANDIDATE.finditer(text):
        if m.group(1).lower() not in ABBREVIATIONS:
            found.append(
                Issue(
                    "unknown-abbreviation",
                    m.group(1),
                    "all-caps token not in ABBREVIATIONS; will be read letter-by-letter "
                    "or as a word, and which one is right is unknowable from orthography",
                )
            )
    for m in _LATIN.finditer(text):
        found.append(
            Issue(
                "latin",
                m.group(0),
                "Latin-script run inside Kyrgyz text; needs a transliteration or "
                "pronunciation decision",
            )
        )
    return found


# --------------------------------------------------------------------------
# Expansion
# --------------------------------------------------------------------------


@dataclass
class Expansion:
    """The result of normalising one string."""

    original: str
    text: str
    issues: list[Issue] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.original != self.text

    def __str__(self) -> str:  # pragma: no cover - display helper
        return self.text


def _expand_dotted_date(m: re.Match) -> str:
    """``01.09.2024`` -> ``эки миң жыйырма төртүнчү жылдын биринчи сентябры``.

    The year is an *ordinal* — Kyrgyz says "of the 2024th year" — and the month
    takes a third-person possessive, which drops a stem-final soft sign.
    """
    day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return m.group(0)
    return f"{ordinal(year)} жылдын {ordinal(day)} {MONTH_POSSESSIVE[MONTH_BY_NUMBER[month]]}"


def _expand_time(m: re.Match) -> str:
    hour, minute = int(m.group(1)), int(m.group(2))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return m.group(0)
    if minute == 0:
        return f"{cardinal(hour)} саат"
    return f"{cardinal(hour)} {cardinal(minute)}"


def expand(text: str, *, collect_issues: bool = True) -> Expansion:
    """Normalise ``text`` for speech.

    Applied in order, most specific pattern first, so that a dotted date is not
    first shredded into three bare integers.

    Ranges are held aside and restored untouched, so a reported issue and the
    output agree with each other:

    >>> expand("16-майда 1500 сом").text
    'он алтынчы майда миң беш жүз сом'
    >>> expand("5% өстү").text
    'беш пайыз өстү'
    >>> expand("2020-2024").text
    '2020-2024'
    """
    found = issues(text) if collect_issues else []
    out = text

    # Held spans are keyed by a private-use codepoint rather than an index, so
    # that the placeholder itself contains no digits for _INT to expand.
    held: list[str] = []

    def _hold(m: re.Match) -> str:
        held.append(m.group(0))
        return f"\x00{chr(0xE000 + len(held) - 1)}\x00"

    # Ranges first: 2020-2024 must not be read as an ordinal or two cardinals.
    out = _RANGE.sub(_hold, out)
    # Dates before decimals, or 01.09.2024 is shredded into a decimal plus junk.
    out = _DATE_DOTTED.sub(_expand_dotted_date, out)
    out = _DECIMAL.sub(_hold, out)
    out = _TIME.sub(_expand_time, out)
    out = _PERCENT.sub(lambda m: f"{cardinal(int(m.group(1)))} пайыз", out)

    # 16-май -> он алтынчы май. The dash marks an ordinal in Kyrgyz orthography.
    out = _ORDINAL_DASH.sub(lambda m: f"{ordinal(int(m.group(1)))} ", out)

    for symbol, word in SYMBOLS.items():
        if symbol != "%":
            out = out.replace(symbol, f" {word}")

    for abbr, full in ABBREVIATIONS.items():
        if "." in abbr:
            out = re.sub(re.escape(abbr), full, out, flags=re.IGNORECASE)
        else:
            out = re.sub(rf"\b{re.escape(abbr)}\b", full, out, flags=re.IGNORECASE)

    out = _INT.sub(lambda m: cardinal(int(m.group(1))), out)
    out = re.sub(r"\s+", " ", out).strip()
    out = re.sub(r"\x00(\d+)\x00", lambda m: held[int(m.group(1))], out)
    return Expansion(original=text, text=out, issues=found)


def expand_text(text: str) -> str:
    """``expand(text).text``, for callers that only want the string."""
    return expand(text, collect_issues=False).text
