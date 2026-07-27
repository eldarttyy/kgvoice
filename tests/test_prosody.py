"""Prosody annotation schema: validation, persistence, and agreement."""

import pytest

from kgvoice.bench import prosody as P


def ann(uid="u1", **kw):
    return P.ProsodyAnnotation(utterance_id=uid, **kw)


# --------------------------------------------------------------------------
# Schema integrity
# --------------------------------------------------------------------------


def test_every_tag_maps_to_its_family():
    for family, tags in P.TAGS.items():
        for tag in tags:
            assert P.TAG_FAMILY[tag] == family


def test_tag_names_are_unique_across_families():
    flat = [t for tags in P.TAGS.values() for t in tags]
    assert len(flat) == len(set(flat))


def test_unknown_tag_is_rejected():
    with pytest.raises(ValueError, match="unknown tag"):
        P.ProsodyTag(tag="sounds-bad", word_index=0)


def test_unknown_severity_is_rejected():
    with pytest.raises(ValueError, match="severity"):
        P.ProsodyTag(tag="stress-misplaced", word_index=0, severity="catastrophic")


def test_tag_exposes_description_and_family():
    t = P.ProsodyTag(tag="stress-misplaced", word_index=2, syllable_index=1)
    assert t.family == "stress"
    assert "wrong syllable" in t.description


def test_blocking_tags_are_blocking_regardless_of_severity():
    assert P.ProsodyTag(tag="clipping", word_index=0, severity="minor").is_blocking


def test_severity_can_make_any_tag_blocking():
    assert P.ProsodyTag(tag="filled-pause", word_index=0, severity="blocking").is_blocking


def test_ordinary_tag_is_not_blocking():
    assert not P.ProsodyTag(tag="filled-pause", word_index=0).is_blocking


# --------------------------------------------------------------------------
# Annotation behaviour
# --------------------------------------------------------------------------


def test_add_appends_and_returns():
    a = ann()
    t = a.add("stress-misplaced", 3, word="кулдар", syllable_index=0, expected="1", observed="0")
    assert a.tags == [t]
    assert t.expected == "1"


def test_usability_reflects_blocking_tags():
    a = ann()
    a.add("filled-pause", 1)
    assert a.is_usable
    a.add("clipping", 2)
    assert not a.is_usable


def test_by_family_groups():
    a = ann()
    a.add("stress-misplaced", 0)
    a.add("stress-missing", 1)
    a.add("filled-pause", 2)
    grouped = a.by_family()
    assert len(grouped["stress"]) == 2
    assert len(grouped["disfluency"]) == 1


def test_counts_are_sorted_descending():
    a = ann()
    a.add("filled-pause", 0)
    a.add("filled-pause", 1)
    a.add("stress-missing", 2)
    assert list(a.counts()) == ["filled-pause", "stress-missing"]


def test_str_includes_evidence():
    t = P.ProsodyTag(
        tag="seg-substitution", word_index=4, word="көл", expected="ø", observed="o"
    )
    s = str(t)
    assert "seg-substitution" in s and "'ø'" in s and "'o'" in s


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def test_round_trip(tmp_path):
    a = ann("u1", transcript="Бишкек шаары", annotator="eldar", overall_rating=4)
    a.add("stress-misplaced", 1, word="шаары", syllable_index=0)
    b = ann("u2")
    b.add("clipping", 0, severity="blocking")

    path = P.save([a, b], tmp_path / "ann.jsonl")
    loaded = P.load(path)

    assert [x.utterance_id for x in loaded] == ["u1", "u2"]
    assert loaded[0].as_dict() == a.as_dict()
    assert loaded[1].is_usable is False


def test_saved_file_is_one_line_per_utterance(tmp_path):
    path = P.save([ann("u1"), ann("u2"), ann("u3")], tmp_path / "a.jsonl")
    assert len([ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]) == 3


# --------------------------------------------------------------------------
# Agreement
# --------------------------------------------------------------------------


def test_perfect_agreement():
    a, b = ann("u1"), ann("u1")
    a.add("stress-misplaced", 2)
    b.add("stress-misplaced", 2)
    result = P.agreement([a], [b])
    assert result["exact_agreement"] == 1.0
    assert result["word_level_agreement"] == 1.0


def test_same_word_different_tag_splits_the_two_measures():
    """Both annotators flagged word 2; they disagree on what is wrong with it.

    Exact agreement should be 0 while word-level agreement is 1 — the distinction
    the module docstring argues is worth reporting separately.
    """
    a, b = ann("u1"), ann("u1")
    a.add("stress-misplaced", 2)
    b.add("stress-missing", 2)
    result = P.agreement([a], [b])
    assert result["exact_agreement"] == 0.0
    assert result["word_level_agreement"] == 1.0


def test_disjoint_words_agree_on_nothing():
    a, b = ann("u1"), ann("u1")
    a.add("stress-misplaced", 1)
    b.add("stress-misplaced", 5)
    assert P.agreement([a], [b])["word_level_agreement"] == 0.0


def test_agreement_ignores_unshared_utterances():
    a, b = ann("u1"), ann("u2")
    assert P.agreement([a], [b])["utterances"] == 0


def test_agreement_on_two_clean_annotations_is_one():
    assert P.agreement([ann("u1")], [ann("u1")])["exact_agreement"] == 1.0


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------


def test_summarize():
    a = ann("u1", overall_rating=5)
    a.add("filled-pause", 0)
    b = ann("u2", overall_rating=3)
    b.add("clipping", 1)
    b.add("stress-misplaced", 2)

    s = P.summarize([a, b])
    assert s["utterances"] == 2
    assert s["usable"] == 1
    assert s["total_tags"] == 3
    assert s["mean_rating"] == 4.0
    assert s["by_family"]["stress"] == 1


def test_summarize_empty():
    assert P.summarize([])["utterances"] == 0
