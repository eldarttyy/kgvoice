"""Harmony rules, including the у/ү asymmetry the module docstring argues for."""

import pytest

from kgvoice.phon.harmony import (
    attach,
    check_harmony,
    harmonize,
    is_harmonic,
    paradigm,
    realize_for_stem,
)


@pytest.mark.parametrize(
    "stem,expected",
    [
        ("китеп", "тер"),  # front unrounded trigger, voiceless final -> т
        ("кол", "дор"),  # low back rounded spreads rounding
        ("кул", "дар"),  # HIGH back rounded does NOT spread to a low target
        ("гүл", "дөр"),  # high FRONT rounded does spread
        ("көл", "дөр"),
        ("эл", "дер"),
    ],
)
def test_plural(stem, expected):
    assert harmonize("LAр", stem) == expected
    assert attach("LAр", stem) == stem + expected


def test_the_u_gap():
    """кул and гүл differ only in the backness of a high rounded vowel.

    The commonly stated rule ("/A/ is о after any back rounded vowel") predicts
    *кулдор. The corpus-fitted rule predicts кулдар. This test pins the claim.
    """
    assert attach("LAр", "кул") == "кулдар"
    assert attach("LAр", "гүл") == "гүлдөр"
    assert realize_for_stem("A", "кул") == "а"
    assert realize_for_stem("I", "кул") == "у"  # rounding still spreads to HIGH


@pytest.mark.parametrize(
    "template,stem,expected",
    [
        ("NIн", "окуу", "нун"),
        ("NIн", "көл", "дүн"),
        ("DA", "китеп", "те"),
        ("GA", "кол", "го"),
        ("DAн", "үй", "дөн"),
    ],
)
def test_consonant_alternation(template, stem, expected):
    assert harmonize(template, stem) == expected


def test_paradigm_is_complete():
    forms = paradigm("кол")
    assert set(forms) == {
        "plural",
        "genitive",
        "dative",
        "accusative",
        "locative",
        "ablative",
        "similative",
        "denominal_adj",
    }
    assert all(f.startswith("кол") for f in forms.values())


def test_no_harmonic_trigger_falls_back_to_back_unrounded():
    assert realize_for_stem("A", "xyz") == "а"
    assert realize_for_stem("I", "xyz") == "ы"


@pytest.mark.parametrize("word", ["китеп", "кол", "гүл", "балдар", "көлдөр"])
def test_native_words_are_harmonic(word):
    assert is_harmonic(word), check_harmony(word)


def test_loanword_breaks_harmony():
    # A Russian borrowing mixing front and back vowels should be flagged.
    assert check_harmony("телефон")
