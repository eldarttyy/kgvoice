"""Hand-checked lexical data shared by the loanword and stress modules.

Kept in its own module because both need it and neither should import the other:
:mod:`kgvoice.phon.loanword` asks *is this borrowed* (which decides ``ж`` -> /ʒ/),
and :mod:`kgvoice.phon.stress` asks *where is it stressed*, and the answer to the
second is only meaningful for words where the answer to the first is yes.

Everything here is entered by hand with the syllabification written out. It is
small on purpose: an unverified entry is worse than a missing one, because a
missing entry produces ``confidence='unknown'`` and gets reviewed, while a wrong
entry produces confident nonsense.
"""

from __future__ import annotations

#: Source-language stress for common borrowings, as a 0-based syllable index
#: *under this package's syllabification* (see :mod:`kgvoice.phon.syllable`).
LOANWORD_STRESS: dict[str, int] = {
    # Government and institutions — the dominant entity vocabulary in news text.
    "министр": 1,        # ми-НИСТР
    "министрлик": 1,     # ми-НИСТР-лик — Kyrgyz suffixes do not move Russian stress
    "президент": 2,      # пре-зи-ДЕНТ
    "комитет": 1,        # ко-МИ-тет
    "депутат": 2,        # де-пу-ТАТ
    "парламент": 1,      # пар-ЛА-мент
    "прокурор": 2,       # про-ку-РОР
    "прокуратура": 3,    # про-ку-ра-ТУ-ра
    "губернатор": 2,     # гу-бер-НА-тор
    "директор": 1,       # ди-РЕК-тор
    "профессор": 1,      # про-ФЕС-сор
    "доктор": 0,         # ДОК-тор
    "область": 0,        # ОБ-ласть
    "район": 1,          # ра-ЙОН
    "департамент": 3,    # де-пар-та-МЕНТ
    "агентство": 1,      # а-ГЕНТ-ство
    "администрация": 3,  # ад-ми-ни-СТРА-ци-я
    "комиссия": 1,       # ко-МИС-си-я
    "конференция": 2,    # кон-фе-РЕН-ци-я
    "милиция": 1,        # ми-ЛИ-ци-я
    "полиция": 1,        # по-ЛИ-ци-я
    "инспекция": 1,      # ин-СПЕК-ци-я
    "резиденция": 2,     # ре-зи-ДЕН-ци-я
    # Places and proper nouns.
    "россия": 1,         # рос-СИ-я
    "москва": 1,         # мо-СКВА
    "казакстан": 2,      # ка-зак-СТАН
    "азия": 0,           # А-зи-я
    "европа": 1,         # ев-РО-па
    "евразия": 2,        # ев-ра-ЗИ-я
    "украина": 2,        # ук-ра-И-на
    "турция": 0,         # ТУР-ци-я
    # Measures, money, time.
    "процент": 1,        # про-ЦЕНТ
    "миллион": 2,        # мил-ли-ОН
    "миллиард": 2,       # мил-ли-АРД
    "доллар": 0,         # ДОЛ-лар
    "километр": 2,       # ки-ло-МЕТР
    "гектар": 1,         # гек-ТАР
    "тонна": 0,          # ТОН-на
    # Everyday borrowings.
    "телефон": 2,        # те-ле-ФОН
    "телевизор": 2,      # те-ле-ВИ-зор
    "компания": 1,       # ком-ПА-ни-я
    "университет": 4,    # у-ни-вер-си-ТЕТ
    "институт": 2,       # ин-сти-ТУТ
    "академия": 2,       # а-ка-ДЕ-ми-я
    "программа": 1,      # прог-РАМ-ма
    "система": 1,        # си-СТЕ-ма
    "автобус": 1,        # ав-ТО-бус
    "газета": 1,         # га-ЗЕ-та
    "журнал": 1,         # жур-НАЛ
    "журналист": 2,      # жур-на-ЛИСТ
    "банк": 0,           # БАНК
    "футбол": 1,         # фут-БОЛ
    "интернет": 2,       # ин-тер-НЕТ
    "фестиваль": 2,      # фе-сти-ВАЛЬ
    "режим": 1,          # ре-ЖИМ
    "проект": 1,         # про-ЕКТ
    "бюджет": 1,         # бю-ДЖЕТ
}

#: Borrowed stems that carry no reliable stress judgement here but should still
#: be treated as borrowings — chiefly for the ``ж`` -> /ʒ/ rule and for
#: suppressing spurious disharmony warnings.
EXTRA_LOAN_STEMS: frozenset[str] = frozenset(
    {
        "жанр", "жест", "жетон", "жюри", "инженер", "пляж", "багаж", "этаж",
        "гараж", "тираж", "монтаж", "репортаж", "жаргон", "жилет",
    }
)

#: Every stem known to be borrowed.
KNOWN_LOAN_STEMS: frozenset[str] = frozenset(LOANWORD_STRESS) | EXTRA_LOAN_STEMS

#: Longest first, so ``министрлик`` matches before ``министр``.
_BY_LENGTH: tuple[str, ...] = tuple(sorted(KNOWN_LOAN_STEMS, key=len, reverse=True))


def known_loan_stem(word: str) -> str | None:
    """Longest known borrowed stem that ``word`` begins with, if any.

    Prefix matching rather than exact lookup, because Kyrgyz suffixes stack
    freely onto borrowed stems: ``министрлигинин``, ``процентке``, ``банктан``.
    A three-character floor avoids matching short stems inside unrelated words.
    """
    w = word.lower().strip()
    if not w:
        return None
    for stem in _BY_LENGTH:
        if len(stem) >= 3 and w.startswith(stem):
            return stem
    return None
