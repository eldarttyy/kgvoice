"""Benchmark sentence selection: readability gating, scoring, and diversity."""

import pytest

from kgvoice.bench import select
from kgvoice.corpus.conll import Sentence, Token


def sent(text, tags=None, sent_id="s1"):
    """Build a Sentence from whitespace-separated text and optional BIO tags."""
    words = text.split()
    tags = tags or ["O"] * len(words)
    return Sentence(sent_id=sent_id, tokens=[Token(w, t) for w, t in zip(words, tags)])


NORMAL = sent(
    "Бишкек шаарында 200 миң киши жашайт жана алар жаңы турак жай куруп жатышат",
    ["B-LOCATION"] + ["O"] * 2 + ["B-MEASURE"] + ["O"] * 9,
)


# --------------------------------------------------------------------------
# Readability gate
# --------------------------------------------------------------------------


def test_normal_sentence_is_readable():
    assert select.is_readable(NORMAL)
    assert select.readability_problem(NORMAL) is None


def test_url_is_rejected():
    s = sent("https : / / 24. kg / kyrgyzcha / 52608 деген шилтеме боюнча окуңуз")
    assert select.readability_problem(s) == "contains a URL or domain"


def test_html_entity_residue_is_rejected():
    s = sent("Бул laquodyiykanraquo базарынын сатуучулары тууралуу маалымат берилди")
    assert select.readability_problem(s) == "HTML entity residue"


def test_latin_heavy_sentence_is_rejected():
    s = sent("Количество Create bar charts from data служебных заграничных командировок")
    assert "Latin script" in select.readability_problem(s)


def test_too_short_sentence_is_rejected():
    assert select.readability_problem(sent("20 жаштагы К. Р.")) == "too little Kyrgyz text"


def test_empty_sentence_is_rejected():
    assert select.readability_problem(Sentence(sent_id="x", tokens=[])) == "empty"


def test_rejected_reports_reasons():
    bad = sent("https : / / 24. kg / a", sent_id="bad")
    out = select.rejected([NORMAL, bad])
    assert [r.sent_id for r in out] == ["bad"]
    assert out[0].reason == "contains a URL or domain"


def test_rank_filters_unreadable_by_default():
    bad = sent("https : / / 24. kg / a", sent_id="bad")
    assert [s.sentence.sent_id for s in select.rank([NORMAL, bad])] == ["s1"]
    assert len(select.rank([NORMAL, bad], readable_only=False)) == 2


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def test_entity_density_raises_score():
    plain = sent("бул жерде эч кандай атайын аталыш жок болгон себептен ал жөнөкөй")
    dense = sent(
        "Бишкек шаарында 200 миң киши жашайт жана алар турак жай куруп жатышат",
        ["B-LOCATION"] + ["O"] * 2 + ["B-MEASURE"] + ["O"] * 8,
    )
    assert select.score_sentence(dense).total > select.score_sentence(plain).total


def test_numerals_component_fires():
    s = sent("Бул жерде 200 жана 300 жана 500 сандары жазылган турат экен")
    assert select.score_sentence(s).components["numerals"] > 0


def test_length_fit_penalises_extremes():
    short = select.score_sentence(sent("Бишкек шаары чоң"))
    good = select.score_sentence(NORMAL)
    assert good.components["length_fit"] > short.components["length_fit"]


def test_empty_sentence_scores_zero():
    assert select.score_sentence(Sentence(sent_id="x", tokens=[])).total == 0.0


def test_score_components_are_bounded():
    sc = select.score_sentence(NORMAL)
    assert all(0.0 <= v <= 1.0 for v in sc.components.values())
    assert 0.0 <= sc.total <= 1.0


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


@pytest.fixture
def pool():
    return [
        sent("Бишкек шаарында 200 миң киши жашайт жана турак жай куруп жатышат",
             ["B-LOCATION"] + ["O"] * 2 + ["B-MEASURE"] + ["O"] * 7, "a"),
        sent("Ош шаарында 300 миң киши жашайт жана турак жай куруп жатышат",
             ["B-LOCATION"] + ["O"] * 2 + ["B-MEASURE"] + ["O"] * 7, "b"),
        sent("Өзгөчө кырдаалдар министрлиги көчкү тууралуу эскертүү жарыялап жатат",
             ["B-INSTITUTION"] * 3 + ["O"] * 5, "c"),
        sent("Жогорку Кеңештин депутаттары жаңы мыйзам долбоорун кабыл алышты",
             ["B-INSTITUTION"] * 2 + ["O"] * 6, "d"),
    ]


def test_select_respects_n(pool):
    assert len(select.select(pool, n=2)) == 2


def test_select_returns_everything_when_n_exceeds_pool(pool):
    assert len(select.select(pool, n=99)) == len(pool)


def test_zero_diversity_is_pure_score_order(pool):
    chosen = select.select(pool, n=4, diversity=0.0)
    scores = [c.total for c in chosen]
    assert scores == sorted(scores, reverse=True)


def test_diversity_prefers_new_phonemes(pool):
    """Sentences a and b are near-duplicates; diverse selection should not take
    both before touching c or d."""
    ids = [c.sentence.sent_id for c in select.select(pool, n=2, diversity=1.0)]
    assert not {"a", "b"} <= set(ids)


def test_min_score_filters(pool):
    assert select.select(pool, n=4, min_score=1.1) == []


def test_coverage_report_shape(pool):
    chosen = select.select(pool, n=2)
    report = select.coverage_report(chosen, select.rank(pool))
    assert report["sentences"] == 2
    assert 0.0 <= report["phoneme_coverage"] <= 1.0
    assert report["phonemes_covered"] <= report["phonemes_in_corpus"]
    assert set(report["entity_labels"]) <= {"LOCATION", "MEASURE", "INSTITUTION"}


def test_coverage_report_handles_empty():
    assert select.coverage_report([], [])["sentences"] == 0
