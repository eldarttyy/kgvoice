"""Kyrgyz vowel harmony, stated as feature rules and validated against corpus.

Kyrgyz suffix vowels are archiphonemes: a suffix is stored with an underspecified
vowel that surfaces in one of four shapes depending on the last vowel of the stem.
There are two archiphonemes.

**High archiphoneme /I/** — surfaces as ``ы и у ү``. Symmetric: it copies both
backness and rounding from the trigger. Shown with the genitive ``-NIн``:

==================  ==========  =====
trigger             genitive    /I/
==================  ==========  =====
а, ы  (back unr)    кыз-дын     ы
о, у  (back rnd)    жол-дун     у
е, и  (front unr)   эл-дин      и
ө, ү  (front rnd)   көл-дүн     ү
==================  ==========  =====

**Low archiphoneme /A/** — surfaces as ``а е о ө``. Backness copies as expected,
but rounding is *asymmetric*: it spreads from ``о ө ү`` and not from ``у``.
Shown with the plural ``-LAр``:

==================  ==========  =========
trigger             plural      /A/
==================  ==========  =========
а, ы  (back unr)    кыз-дар     а
о     (back low)    кол-дор     о
у     (back high)   кул-дар     **а**
е, и  (front unr)   эл-дер      е
ө     (front low)   көл-дөр     ө
ү     (front high)  гүл-дөр     ө
==================  ==========  =========

That ``у`` gap is the interesting part. General references routinely state the
rule as "/A/ is ``o`` after a back rounded vowel", which predicts ``*кулдор``.
The corpus says otherwise — 77 of 78 relevant plural types after ``у`` take
``-ар``, while ``ү`` takes ``-өр`` 39 times out of 39. The generalisation that
actually fits is height-sensitive: **a high back rounded trigger does not spread
rounding to a low target**, which is exactly the asymmetry Kaun's typology of
rounding harmony predicts for triggers that disagree with their target in height
while offering no front/back perceptual cue.

A second point the corpus forces: ``я ю ё`` are harmony *triggers*. They are
glide + back vowel, and Russian ``-ия`` borrowings — of which the news domain is
full — behave as ordinary back stems (``россия-нын``, ``биология-лык``). Treating
them as inert costs 13 points of accuracy on the ``-KIк`` audit alone. See
:data:`kgvoice.phon.alphabet.TRIGGER_VOWELS`.

Every rule here is written to a generalisation and then checked: ``kgvoice phon
harmony-audit`` re-derives the whole table from whatever corpus you point it at.
On KyrgyzNER (21,463 types) the five audited templates agree with the rules at
94.7%–98.7% by token weight, with no trigger cell whose observed majority differs
from the prediction. The residue is dominated by unrelated words that merely end
in the same letters (``оператор``, ``доллар``), which the audit prints rather
than hides.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from kgvoice.phon.alphabet import (
    BACK,
    FRONT,
    HIGH,
    LOW,
    NATIVE_VOWEL_LETTERS,
    TRIGGER_VOWELS,
    VOWELS,
    Vowel,
    last_harmonic_vowel,
)
from kgvoice.phon.syllable import nuclei as _nuclei

#: Display order for audit tables: native eight first, then the iotated
#: Russian letters that also trigger harmony.
TRIGGER_ORDER = "аыоуеиөүэяюё"

# --------------------------------------------------------------------------
# Archiphoneme resolution
# --------------------------------------------------------------------------

#: (backness, rounded, height) -> surface letter
_SURFACE: dict[tuple[str, bool, str], str] = {
    (BACK, False, HIGH): "ы",
    (BACK, True, HIGH): "у",
    (FRONT, False, HIGH): "и",
    (FRONT, True, HIGH): "ү",
    (BACK, False, LOW): "а",
    (BACK, True, LOW): "о",
    (FRONT, False, LOW): "е",
    (FRONT, True, LOW): "ө",
}


def spreads_rounding(trigger: Vowel, target_height: str) -> bool:
    """Does ``trigger`` spread [+round] onto a target of ``target_height``?

    Rounding always spreads to a high target. It spreads to a *low* target only
    when the trigger is itself low (``о``, ``ө``) or front (``ү``); the high back
    rounded ``у`` does not. See the module docstring.
    """
    if not trigger.rounded:
        return False
    if target_height == HIGH:
        return True
    return not (trigger.is_high and trigger.is_back)


def realize(archiphoneme: str, trigger: Vowel | None) -> str:
    """Surface form of ``'A'`` (low) or ``'I'`` (high) after ``trigger``.

    With no harmonic trigger (an all-loanword stem, say) the back unrounded
    default is used, which is what Kyrgyz does with opaque foreign stems.
    """
    height = {"A": LOW, "I": HIGH}[archiphoneme.upper()]
    if trigger is None:
        return _SURFACE[(BACK, False, height)]
    return _SURFACE[(trigger.backness, spreads_rounding(trigger, height), height)]


def realize_for_stem(archiphoneme: str, stem: str) -> str:
    """Surface vowel that ``archiphoneme`` takes when suffixed to ``stem``."""
    return realize(archiphoneme, last_harmonic_vowel(stem))


# --------------------------------------------------------------------------
# Suffix-initial consonant alternation
# --------------------------------------------------------------------------

#: Obstruents that surface voiceless word-finally. Kyrgyz orthography already
#: spells native finals as п/т/к/с, so this set mostly matters for loanwords.
_VOICELESS = set("пткссшчфхцщ") | set("бвгд")
_SONORANT = set("йлмнңр")
_VOICED_OBSTRUENT = set("жз")


def _final_sound(stem: str) -> str:
    for ch in reversed(stem.lower()):
        if ch in VOWELS or ch in _SONORANT or ch in _VOICELESS or ch in _VOICED_OBSTRUENT:
            return ch
    return ""


def realize_consonant(archiphoneme: str, stem: str) -> str:
    """Surface form of a suffix-initial consonant archiphoneme.

    ``L`` — plural-type (``л``/``д``/``т``)
    ``K`` — denominal-adjective-type (``л``/``д``/``т``)
    ``N`` — genitive/accusative-type (``н``/``д``/``т``)
    ``D`` — locative/ablative-type (``д``/``т``)
    ``G`` — dative-type (``г``/``к``)

    ``L`` and ``K`` differ in exactly one environment, after ``р``: the plural
    takes ``л`` (``кызматкер-лер``, 47:2 in corpus) while the denominal adjective
    takes ``д`` (``борбор-дук``, ``аскер-дик``, 36:18). The ``K``-after-``р``
    minority is not noise but a lexical split — it is almost entirely
    unassimilated loanword stems (``километр-лик``, ``рейдер-лик``), which is why
    :func:`realize_consonant` accepts the majority rule and
    :mod:`kgvoice.phon.loanword` is what decides whether to trust it.
    """
    ch = _final_sound(stem)
    after_vowel = ch in VOWELS
    voiceless = ch in _VOICELESS
    arch = archiphoneme.upper()

    if arch == "L":
        if after_vowel or ch in "йр":
            return "л"
        return "т" if voiceless else "д"
    if arch == "K":
        if after_vowel or ch == "й":
            return "л"
        return "т" if voiceless else "д"
    if arch == "N":
        if after_vowel:
            return "н"
        return "т" if voiceless else "д"
    if arch == "D":
        return "т" if voiceless else "д"
    if arch == "G":
        return "к" if voiceless else "г"
    raise KeyError(f"unknown consonant archiphoneme {archiphoneme!r}")


def harmonize(template: str, stem: str) -> str:
    """Realise a suffix ``template`` for ``stem``.

    Templates use ``A``/``I`` for vowel archiphonemes and ``L``/``N``/``D``/``G``
    for consonant archiphonemes; every other character is literal.

    >>> harmonize("LAр", "кол")
    'дор'
    >>> harmonize("LAр", "кул")
    'дар'
    >>> harmonize("LAр", "гүл")
    'дөр'
    >>> harmonize("NIн", "окуу")
    'нун'
    """
    out = []
    for ch in template:
        if ch in ("A", "I"):
            out.append(realize_for_stem(ch, stem))
        elif ch in ("L", "K", "N", "D", "G"):
            out.append(realize_consonant(ch, stem))
        else:
            out.append(ch)
    return "".join(out)


def attach(template: str, stem: str) -> str:
    """``stem`` + realised ``template``."""
    return stem + harmonize(template, stem)


#: A minimal nominal paradigm, enough to exercise both archiphonemes and all
#: four consonant alternations.
SUFFIXES: dict[str, str] = {
    "plural": "LAр",
    "genitive": "NIн",
    "dative": "GA",
    "accusative": "NI",
    "locative": "DA",
    "ablative": "DAн",
    "similative": "DAй",
    "denominal_adj": "KIк",
}


def paradigm(stem: str) -> dict[str, str]:
    """Full realised paradigm of ``stem``."""
    return {name: attach(tpl, stem) for name, tpl in SUFFIXES.items()}


# --------------------------------------------------------------------------
# Stem-internal harmony checking
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HarmonyViolation:
    """One adjacent nucleus pair in a word that breaks harmony."""

    index: int
    left: str
    right: str
    kind: str  # 'backness' | 'rounding'

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"{self.left}->{self.right} ({self.kind}) at vowel {self.index}"


def check_harmony(word: str, *, rounding: bool = True) -> list[HarmonyViolation]:
    """Adjacent nucleus pairs in ``word`` that a harmonic native stem would not show.

    Two things make this narrower than a naive vowel-by-vowel scan, and both are
    needed to stop it firing on perfectly ordinary Kyrgyz words.

    **Nuclei, not letters.** A long vowel is written as a digraph, so scanning
    characters turns ``алуу`` into ``а-у-у`` and reports a spurious ``а->у``
    rounding violation on a word that has none. Nuclei are taken from
    :func:`kgvoice.phon.syllable.nuclei`.

    **Long targets are exempt from rounding.** The 4-way rounding alternation is
    a property of the short archiphonemes. The suffixes that carry a long high
    round vowel — the verbal noun ``-уу/-үү`` (``алуу``, ``аткаруу``) and the
    adjectival ``-лУУ`` (``пландуу``, ``сапаттуу``) — alternate for backness only.
    Checking rounding against them misfires on 514 corpus types by itself.

    Backness is the reliable signal: 86.3% of native-looking corpus types are
    backness-clean, and the residue is dominated by real compounds and
    borrowings. Rounding is the noisier one, which is why ``rounding=False`` is
    offered for callers that want only the strong evidence.
    """
    nuc = [n for n in _nuclei(word.lower()) if n[0] in NATIVE_VOWEL_LETTERS]
    violations: list[HarmonyViolation] = []
    for i in range(len(nuc) - 1):
        left, right = nuc[i], nuc[i + 1]
        a, b = VOWELS[left[0]], VOWELS[right[0]]
        if a.backness != b.backness:
            violations.append(HarmonyViolation(i, left, right, "backness"))
            continue
        if not rounding or len(right) > 1:
            continue  # long target: backness-only alternation
        if spreads_rounding(a, b.height) != b.rounded:
            violations.append(HarmonyViolation(i, left, right, "rounding"))
    return violations


def is_harmonic(word: str, *, rounding: bool = True) -> bool:
    return not check_harmony(word, rounding=rounding)


def harmonic_class(word: str) -> str | None:
    """``'back'`` / ``'front'`` for a word, from its last harmonic vowel."""
    v = last_harmonic_vowel(word)
    return None if v is None else v.backness


# --------------------------------------------------------------------------
# Corpus audit — re-derive the table from data
# --------------------------------------------------------------------------

#: Surface vowel sets, used to recognise a realised suffix in running text.
_LOW_SET = "аеоө"
_HIGH_SET = "ыиуү"

#: Surface consonants each consonant archiphoneme can appear as.
_CONS_OPTIONS = {"L": "лдт", "K": "лдт", "N": "ндт", "D": "дт", "G": "гк", "": ""}


@dataclass
class AuditCell:
    """Observed realisations of one archiphoneme after one trigger vowel."""

    trigger: str
    observed: Counter = field(default_factory=Counter)
    examples: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    @property
    def total(self) -> int:
        return sum(self.observed.values())

    @property
    def majority(self) -> str | None:
        return self.observed.most_common(1)[0][0] if self.observed else None

    def agreement(self, predicted: str) -> float:
        return self.observed[predicted] / self.total if self.total else 0.0


@dataclass
class AuditResult:
    archiphoneme: str
    template: str
    cells: dict[str, AuditCell]

    def rows(self) -> list[tuple[str, str, str, int, float]]:
        """(trigger, predicted, observed_majority, n, agreement) per trigger."""
        out = []
        for letter in TRIGGER_ORDER:
            cell = self.cells.get(letter)
            if not cell or not cell.total:
                continue
            predicted = realize(self.archiphoneme, VOWELS[letter])
            out.append(
                (letter, predicted, cell.majority or "", cell.total, cell.agreement(predicted))
            )
        return out

    @property
    def weighted_agreement(self) -> float:
        num = den = 0
        for letter, predicted, _maj, n, agree in self.rows():
            num += agree * n
            den += n
        return num / den if den else 0.0

    def format(self) -> str:  # pragma: no cover - display helper
        head = (
            f"/{self.archiphoneme}/ via template {self.template!r}\n"
            f"{'trigger':<9}{'predicted':<11}{'observed':<11}{'n':>7}{'agree':>9}\n"
        )
        lines = []
        for letter, predicted, majority, n, agree in self.rows():
            flag = "" if majority == predicted else "   <-- MISMATCH"
            lines.append(f"{letter:<9}{predicted:<11}{majority:<11}{n:>7}{agree:>8.1%}{flag}")
        lines.append(f"{'':<9}{'':<11}{'weighted':<11}{'':>7}{self.weighted_agreement:>8.1%}")
        return head + "\n".join(lines)


def audit_archiphoneme(
    words: Iterable[str],
    template: str,
    archiphoneme: str,
    *,
    min_stem_len: int = 2,
    max_examples: int = 4,
) -> AuditResult:
    """Re-derive an archiphoneme's realisation table from a word list.

    ``template`` is a suffix template such as ``"LAр"``. Every word that ends in
    some surface realisation of it is segmented into stem + suffix, and the
    suffix vowel is tallied against the stem's last harmonic vowel.

    This is deliberately shallow morphology: it will pick up unrelated words that
    merely end in the same letters (``доллар`` looks like a plural). Those land in
    the minority column and are reported as examples, which is the point — the
    audit surfaces its own noise instead of hiding it.
    """
    vowel_set = _LOW_SET if archiphoneme.upper() == "A" else _HIGH_SET
    prefix, suffix_tail = template.split(archiphoneme.upper(), 1)
    cons_arch = prefix  # e.g. 'L', 'N', 'D', 'G' or ''
    cells: dict[str, AuditCell] = {v: AuditCell(v) for v in TRIGGER_VOWELS}

    # Candidate surface consonants for the templated consonant archiphoneme.
    cons_options = _CONS_OPTIONS.get(cons_arch, "")

    for word in words:
        w = word.lower()
        if suffix_tail and not w.endswith(suffix_tail):
            continue
        body = w[: len(w) - len(suffix_tail)] if suffix_tail else w
        if not body:
            continue
        surface_vowel = body[-1]
        if surface_vowel not in vowel_set:
            continue
        body = body[:-1]
        if cons_options:
            if not body or body[-1] not in cons_options:
                continue
            body = body[:-1]
        stem = body
        if len(stem) < min_stem_len:
            continue
        trigger = last_harmonic_vowel(stem)
        if trigger is None:
            continue
        cell = cells[trigger.letter]
        cell.observed[surface_vowel] += 1
        if len(cell.examples[surface_vowel]) < max_examples:
            cell.examples[surface_vowel].append(word)

    return AuditResult(archiphoneme=archiphoneme.upper(), template=template, cells=cells)


#: Ordered environment classes for the consonant audit.
CONS_ENVIRONMENTS: tuple[tuple[str, str], ...] = (
    ("vowel", ""),
    ("й", "й"),
    ("р", "р"),
    ("л м н ң", "лмнң"),
    ("ж з", "жз"),
    ("voiceless", ""),
)


def _cons_environment(stem: str) -> str:
    ch = _final_sound(stem)
    if ch in VOWELS:
        return "vowel"
    if ch == "й":
        return "й"
    if ch == "р":
        return "р"
    if ch in "лмнң":
        return "л м н ң"
    if ch in _VOICED_OBSTRUENT:
        return "ж з"
    if ch in _VOICELESS:
        return "voiceless"
    return "other"


@dataclass
class ConsonantAuditResult:
    archiphoneme: str
    template: str
    cells: dict[str, AuditCell]

    def rows(self) -> list[tuple[str, str, str, int, float, list[str]]]:
        out = []
        for name, _ in CONS_ENVIRONMENTS:
            cell = self.cells.get(name)
            if not cell or not cell.total:
                continue
            probe = {"vowel": "ата", "й": "сый", "р": "борбор", "л м н ң": "эл",
                     "ж з": "кыз", "voiceless": "ат"}[name]
            predicted = realize_consonant(self.archiphoneme, probe)
            minority = [
                ex
                for surf, n in cell.observed.most_common()
                if surf != predicted
                for ex in cell.examples[surf][:2]
            ]
            out.append(
                (name, predicted, cell.majority or "", cell.total,
                 cell.agreement(predicted), minority[:4])
            )
        return out

    def format(self) -> str:  # pragma: no cover - display helper
        head = (
            f"/{self.archiphoneme}/ via template {self.template!r}\n"
            f"{'after':<11}{'predicted':<11}{'observed':<11}{'n':>6}{'agree':>8}  counterexamples\n"
        )
        lines = []
        for env, predicted, majority, n, agree, minority in self.rows():
            flag = "" if majority == predicted else "  <-- MISMATCH"
            lines.append(
                f"{env:<11}{predicted:<11}{majority:<11}{n:>6}{agree:>7.1%}  "
                f"{', '.join(minority)}{flag}"
            )
        return head + "\n".join(lines)


def audit_consonant(
    words: Iterable[str], template: str, *, min_stem_len: int = 2, max_examples: int = 4
) -> ConsonantAuditResult:
    """Re-derive a consonant archiphoneme's alternation table from a word list.

    Mirrors :func:`audit_archiphoneme` but tallies the suffix-initial consonant
    against the phonological class of the stem-final segment.
    """
    arch = template[0].upper()
    if arch not in _CONS_OPTIONS or not _CONS_OPTIONS[arch]:
        raise ValueError(f"template {template!r} does not start with a consonant archiphoneme")
    vowel_arch = "A" if "A" in template else "I"
    vowel_set = _LOW_SET if vowel_arch == "A" else _HIGH_SET
    suffix_tail = template.split(vowel_arch, 1)[1]

    cells: dict[str, AuditCell] = {name: AuditCell(name) for name, _ in CONS_ENVIRONMENTS}

    for word in words:
        w = word.lower()
        if suffix_tail and not w.endswith(suffix_tail):
            continue
        body = w[: len(w) - len(suffix_tail)] if suffix_tail else w
        if len(body) < 2 or body[-1] not in vowel_set:
            continue
        body = body[:-1]
        if not body or body[-1] not in _CONS_OPTIONS[arch]:
            continue
        surface_cons = body[-1]
        stem = body[:-1]
        if len(stem) < min_stem_len:
            continue
        env = _cons_environment(stem)
        if env == "other":
            continue
        cell = cells[env]
        cell.observed[surface_cons] += 1
        if len(cell.examples[surface_cons]) < max_examples:
            cell.examples[surface_cons].append(word)

    return ConsonantAuditResult(archiphoneme=arch, template=template, cells=cells)


def audit_all(words: Sequence[str], templates: dict[str, str] | None = None) -> list[AuditResult]:
    """Audit every suffix template that contains a vowel archiphoneme."""
    templates = templates or SUFFIXES
    results = []
    for _name, tpl in templates.items():
        for arch in ("A", "I"):
            if arch in tpl:
                results.append(audit_archiphoneme(words, tpl, arch))
                break
    return results
