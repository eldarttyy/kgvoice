"""Entity-weighted WER: alignment, attribution, and aggregation."""

import pytest

from kgvoice.bench import wer


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------


def test_tokenize_folds_case_and_strips_punctuation():
    assert wer.tokenize("Бишкек, шаары!") == ["бишкек", "шаары"]


def test_tokenize_can_keep_case():
    assert wer.tokenize("Бишкек", keep_case=True) == ["Бишкек"]


def test_tokenize_empty():
    assert wer.tokenize("") == []
    assert wer.tokenize("   ") == []


# --------------------------------------------------------------------------
# Plain WER
# --------------------------------------------------------------------------


def test_identical_is_zero():
    assert wer.score("бир эки үч", "бир эки үч").wer == 0.0


@pytest.mark.parametrize(
    "ref,hyp,kind,count,expected_wer",
    [
        ("бир эки үч", "бир эки", "deletions", 1, 1 / 3),
        ("бир эки үч", "бир эки төрт", "substitutions", 1, 1 / 3),
        ("бир эки", "бир жаңы эки", "insertions", 1, 1 / 2),
    ],
)
def test_single_edit(ref, hyp, kind, count, expected_wer):
    r = wer.score(ref, hyp)
    assert getattr(r, kind) == count
    assert r.wer == pytest.approx(expected_wer)


def test_empty_reference_scores_zero_not_divide_by_zero():
    r = wer.score("", "шумдук")
    assert r.insertions == 1
    assert r.wer == 0.0


def test_empty_hypothesis_is_total_loss():
    r = wer.score("бир эки", "")
    assert (r.deletions, r.wer) == (2, 1.0)


def test_both_empty():
    r = wer.score("", "")
    assert (r.wer, r.ref_tokens) == (0.0, 0)


# --------------------------------------------------------------------------
# Entity weighting
# --------------------------------------------------------------------------

REF = "президент бишкекке келди"
LABELS = [None, "LOCATION", None]


def test_entity_error_separates_from_overall():
    r = wer.score(REF, "президент ошко келди", LABELS)
    assert r.entity_substitutions == 1
    assert r.entity_wer == 1.0
    assert r.non_entity_wer == 0.0
    assert r.wer == pytest.approx(1 / 3)


def test_non_entity_error_leaves_entity_clean():
    r = wer.score(REF, "министр бишкекке келди", LABELS)
    assert r.entity_errors == 0
    assert r.entity_wer == 0.0
    assert r.non_entity_wer == pytest.approx(1 / 2)


def test_by_label_counts():
    r = wer.score(REF, "президент ошко келди", LABELS)
    assert dict(r.by_label["LOCATION"]) == {"ref": 1, "sub": 1}
    assert r.label_wer("LOCATION") == 1.0
    assert r.label_wer("PERSON") == 0.0  # absent label, not an error


def test_insertion_is_charged_to_a_neighbouring_entity():
    """A hallucinated word beside a name must cost something."""
    r = wer.score(REF, "президент чоң бишкекке келди", LABELS)
    assert r.entity_insertions == 1


def test_mismatched_labels_raise():
    with pytest.raises(ValueError, match="parallel"):
        wer.score("бир эки", "бир", [None])


def test_labels_default_to_none():
    r = wer.score("бир эки", "бир үч")
    assert r.entity_ref_tokens == 0
    assert r.entity_wer == 0.0


# --------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------


def test_align_ops_cover_every_reference_token():
    ops = wer.align(["a", "b", "c"], ["a", "x", "c"])
    kinds = [o.kind for o in ops]
    assert kinds == ["equal", "sub", "equal"]


def test_align_prefers_substitution_over_del_ins_pair():
    """Same-length rewrites should align positionally, keeping attribution stable."""
    ops = wer.align(["a", "b"], ["a", "z"])
    assert [o.kind for o in ops] == ["equal", "sub"]


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def test_addition_aggregates_counts():
    a = wer.score("бир эки үч", "бир эки төрт", [None, "X", None])
    b = wer.score("төрт беш", "төрт беш", [None, None])
    t = a + b
    assert t.ref_tokens == 5
    assert t.wer == pytest.approx(1 / 5)
    assert t.ops == []  # per-utterance ops are not meaningful once merged


def test_addition_merges_labels():
    a = wer.score("бишкек келди", "ош келди", ["LOC", None])
    t = a + a
    assert dict(t.by_label["LOC"]) == {"ref": 2, "sub": 2}


def test_score_corpus_matches_manual_sum():
    pairs = [
        (["бир", "эки"], ["бир", "үч"], [None, "X"]),
        (["төрт"], ["төрт"], [None]),
    ]
    total = wer.score_corpus(pairs)
    assert total.ref_tokens == 3
    assert total.entity_substitutions == 1
