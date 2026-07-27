"""Placeholders in Kyrgyz UI strings, and the suffix bug that follows them.

Two separate concerns live here.

**Integrity** — the mechanical check every localisation pipeline has: does the
translation carry the same placeholders as the source? A dropped ``{count}``
loses information; an invented ``{nmae}`` throws at runtime. This is
language-neutral and cheap.

**Suffix collision** — the one that is specific to Kyrgyz, and that no
general-purpose i18n linter catches.

Kyrgyz is agglutinative and harmonic, so the case suffix on a noun is chosen by
that noun's *last vowel and last consonant*. When the noun is a runtime value,
its suffix cannot be written into the template::

    "{name}ге кат жөнөттүңүз"     # "you sent a letter to {name}"

That ``-ге`` is correct for exactly one shape of name and wrong for the rest:

===========  ==================  ==================
value        template produces   correct Kyrgyz
===========  ==================  ==================
Айбек        Айбек**ге**         Айбек**ке**
Нурлан       Нурлан**ге**        Нурлан**га**
Гүл          Гүл**ге**           Гүл**гө**
Чолпон       Чолпон**ге**        Чолпон**го**
===========  ==================  ==================

Four different suffixes, one hardcoded string. The dative alone has eight
surface forms. A reviewer reading the Kyrgyz sees a fluent sentence and approves
it, because the defect is invisible until a real name is substituted — which is
exactly why it needs a linter rather than a proofreader.

The check works by expanding each archiphonemic suffix template from
:mod:`kgvoice.phon.harmony` into its full surface set, then looking at whatever
is glued to the right edge of a placeholder. If it is in that set, the string is
reported with the forms it would actually need and with the runtime call that
produces them.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from itertools import product

from kgvoice.phon.harmony import SUFFIXES, attach

#: Archiphoneme -> the surface segments it can appear as.
_ARCHIPHONEMES: dict[str, str] = {
    "A": "аеоө",
    "I": "ыиуү",
    "L": "лдт",
    "K": "лдт",
    "N": "ндт",
    "D": "дт",
    "G": "гк",
}

#: Suffix templates worth checking, beyond the nominal paradigm in
#: :data:`kgvoice.phon.harmony.SUFFIXES`. These attach to a bare noun and so
#: routinely end up glued to a placeholder in UI copy.
EXTRA_TEMPLATES: dict[str, str] = {
    "possessive_3": "сI",  # {name}сы — "their {name}"
    "possessive_2_formal": "IңIз",  # {name}ыңыз
    "predicative_3": "дIр",  # {name}дир
}

#: Human-facing names for the templates, used in report messages.
TEMPLATE_LABELS: dict[str, str] = {
    "plural": "plural",
    "genitive": "genitive",
    "dative": "dative",
    "accusative": "accusative",
    "locative": "locative",
    "ablative": "ablative",
    "similative": "similative",
    "denominal_adj": "denominal adjective",
    "possessive_3": "3rd-person possessive",
    "possessive_2_formal": "2nd-person formal possessive",
    "predicative_3": "predicative",
}

ALL_TEMPLATES: dict[str, str] = {**SUFFIXES, **EXTRA_TEMPLATES}

#: Values chosen to span the harmony space: front/back x rounded/unrounded, with
#: differing final consonants so the consonant archiphonemes alternate too.
DEMO_VALUES: tuple[str, ...] = ("Айбек", "Нурлан", "Гүл", "Чолпон")


def expand_template(template: str) -> set[str]:
    """Every surface string ``template`` can realise as.

    Over-generates: ``лор`` is only reachable after a back rounded low trigger,
    but the combination is enumerated anyway. That is the right trade for
    *detection* — a false membership costs one reviewed finding, a missed one
    costs a shipped bug — and the report never quotes this set. It quotes
    :func:`required_forms`, which is computed per real value.
    """
    slots = [_ARCHIPHONEMES.get(ch, ch) for ch in template]
    return {"".join(combo) for combo in product(*slots)}


def _build_suffix_index() -> dict[str, list[tuple[str, str]]]:
    """Surface form -> [(template_name, template)], longest forms first at match."""
    index: dict[str, list[tuple[str, str]]] = {}
    for name, template in ALL_TEMPLATES.items():
        for form in expand_template(template):
            index.setdefault(form, []).append((name, template))
    return index


SUFFIX_INDEX: dict[str, list[tuple[str, str]]] = _build_suffix_index()

#: Longest first, so ``дын`` matches the genitive before ``ды`` matches the
#: accusative.
_SUFFIX_FORMS_BY_LENGTH: tuple[str, ...] = tuple(
    sorted(SUFFIX_INDEX, key=len, reverse=True)
)


# --------------------------------------------------------------------------
# Placeholder detection
# --------------------------------------------------------------------------

#: Placeholder syntaxes, in the order they are tried. Longer/more specific
#: patterns come first so ``{{x}}`` is not read as ``{`` + ``{x}``.
_PLACEHOLDER_RE = re.compile(
    r"""
    (?P<double_brace>\{\{\s*[\w.]+\s*\}\})       # {{name}}       i18next, mustache
  | (?P<brace>\{\s*[\w.]*\s*(?::[^{}]*)?\})      # {name} {0} {}  ICU, python
  | (?P<printf_pos>%\d+\$[sdfx])                 # %1$s           positional printf
  | (?P<printf_named>%\([\w.]+\)[sdfx])          # %(name)s       python
  | (?P<printf>%[sdfx%])                         # %s %d          printf
  | (?P<dollar_brace>\$\{\s*[\w.]+\s*\})         # ${name}        JS template
  | (?P<dollar>\$[A-Za-z_]\w*)                   # $name          shell-ish
  | (?P<tag></?\d+>)                             # <0> </0>       react Trans
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class Placeholder:
    """One placeholder occurrence in a string."""

    raw: str
    start: int
    end: int
    kind: str

    @property
    def name(self) -> str:
        """Canonical identity, used to compare source against target.

        Positional and anonymous placeholders normalise to their syntax plus
        index so ``%s`` in the source can be matched against ``%s`` in the
        target without pretending they are named.
        """
        inner = self.raw.strip("{}$%<>/ ")
        inner = inner.split(":")[0].split("$")[0].strip("()sdfx")
        return inner or self.raw


def find_placeholders(text: str) -> list[Placeholder]:
    """Every placeholder in ``text``, left to right."""
    out: list[Placeholder] = []
    for m in _PLACEHOLDER_RE.finditer(text):
        kind = m.lastgroup or "unknown"
        out.append(Placeholder(raw=m.group(), start=m.start(), end=m.end(), kind=kind))
    return out


def placeholder_counts(text: str) -> Counter:
    """Multiset of placeholder identities — the unit of the integrity check."""
    return Counter(p.name for p in find_placeholders(text))


# --------------------------------------------------------------------------
# Suffix collision
# --------------------------------------------------------------------------

_CYRILLIC_RUN = re.compile(r"[а-яёөүң]+", re.IGNORECASE)


@dataclass
class SuffixCollision:
    """A suffix hardcoded onto a placeholder."""

    placeholder: Placeholder
    suffix: str
    #: Candidate analyses, most likely first: ``(template_name, template)``.
    candidates: list[tuple[str, str]] = field(default_factory=list)
    #: Was the suffix attached with a hyphen (``{name}-ге``)?
    hyphenated: bool = False

    @property
    def template_name(self) -> str:
        return self.candidates[0][0] if self.candidates else ""

    @property
    def template(self) -> str:
        return self.candidates[0][1] if self.candidates else ""

    @property
    def label(self) -> str:
        return TEMPLATE_LABELS.get(self.template_name, self.template_name)

    @property
    def variant_count(self) -> int:
        return len(expand_template(self.template)) if self.template else 0

    def required_forms(self, values: tuple[str, ...] = DEMO_VALUES) -> list[tuple[str, str, str]]:
        """``(value, produced_by_template, correct_form)`` for each demo value."""
        rows = []
        for value in values:
            produced = value + self.suffix
            correct = attach(self.template, value) if self.template else produced
            rows.append((value, produced, correct))
        return rows

    def wrong_for(self, values: tuple[str, ...] = DEMO_VALUES) -> list[tuple[str, str, str]]:
        """Only the demo values the hardcoded suffix gets wrong."""
        return [row for row in self.required_forms(values) if row[1] != row[2]]

    @property
    def suggestion(self) -> str:
        if not self.template:
            return (
                "Restructure so the runtime value is not directly suffixed, or confirm "
                "this text is not a case suffix."
            )
        return (
            f"Select the suffix at runtime — "
            f"kgvoice.phon.harmony.attach({self.template!r}, value) — "
            f"or restructure the sentence so {self.placeholder.raw} is not suffixed."
        )


def _match_suffix(run: str) -> list[tuple[str, str]]:
    """Longest suffix analysis of a Cyrillic run glued to a placeholder."""
    for form in _SUFFIX_FORMS_BY_LENGTH:
        if run == form:
            return list(SUFFIX_INDEX[form])
    # The run may be a suffix plus more text ("гө кат"); match a prefix of it.
    for form in _SUFFIX_FORMS_BY_LENGTH:
        if run.startswith(form):
            return list(SUFFIX_INDEX[form])
    return []


def find_suffix_collisions(text: str) -> list[SuffixCollision]:
    """Placeholders in ``text`` with a case suffix written directly after them."""
    out: list[SuffixCollision] = []
    for ph in find_placeholders(text):
        rest = text[ph.end :]
        hyphenated = rest.startswith("-")
        if hyphenated:
            rest = rest[1:]
        m = _CYRILLIC_RUN.match(rest)
        if not m:
            continue
        run = m.group().lower()
        candidates = _match_suffix(run)
        if not candidates:
            continue
        # The longest form the run starts with, mirroring _match_suffix's order.
        matched = max(
            (f for f in SUFFIX_INDEX if run == f or run.startswith(f)),
            key=len,
            default=run,
        )
        out.append(
            SuffixCollision(
                placeholder=ph,
                suffix=matched,
                candidates=candidates,
                hyphenated=hyphenated,
            )
        )
    return out


def is_safe(text: str) -> bool:
    """True when no placeholder in ``text`` carries a hardcoded suffix."""
    return not find_suffix_collisions(text)
