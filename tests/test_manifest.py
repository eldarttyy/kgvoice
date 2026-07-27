"""Manifest construction, round-tripping, and recording attachment."""

import json
import wave

import numpy as np
import pytest

from kgvoice.bench import manifest as M
from kgvoice.bench import select
from kgvoice.corpus.conll import Sentence, Token


def sent(text, tags=None, sent_id="s1"):
    words = text.split()
    tags = tags or ["O"] * len(words)
    return Sentence(sent_id=sent_id, tokens=[Token(w, t) for w, t in zip(words, tags)])


@pytest.fixture
def selected():
    pool = [
        sent(
            "Бишкек шаарында 200 миң киши жашайт жана турак жай куруп жатышат",
            ["B-LOCATION"] + ["O"] * 2 + ["B-MEASURE"] + ["O"] * 7,
            "a",
        ),
        sent(
            "Өзгөчө кырдаалдар министрлиги көчкү тууралуу эскертүү жарыялап жатат",
            ["B-INSTITUTION"] * 3 + ["O"] * 5,
            "b",
        ),
    ]
    return select.select(pool, n=2)


@pytest.fixture
def man(selected):
    return M.build(selected, name="test-session", prefix="tst")


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def test_ids_are_sequential_and_prefixed(man):
    assert [i.utterance_id for i in man.items] == ["tst-0001", "tst-0002"]


def test_entities_are_carried_with_ipa(man):
    ents = [e for i in man.items for e in i.entities]
    assert ents
    assert all(e.label for e in ents)
    assert any(e.ipa for e in ents)


def test_reference_tokens_are_stored_and_parallel(man):
    for item in man.items:
        assert item.ref_tokens
        assert len(item.ref_tokens) == len(item.ref_labels)


def test_stored_labels_mark_entity_tokens(man):
    item = man.by_id("tst-0001") or man.items[0]
    labelled = {lab for lab in item.ref_labels if lab}
    assert labelled  # at least one entity survived tokenisation


def test_hints_only_cover_difficult_words(man):
    for item in man.items:
        for hint in item.hints:
            assert hint.difficulty >= 0.35
            assert hint.ipa and hint.hyphenated


def test_hints_are_sorted_hardest_first(man):
    for item in man.items:
        diffs = [h.difficulty for h in item.hints]
        assert diffs == sorted(diffs, reverse=True)


def test_hints_are_deduplicated(man):
    for item in man.items:
        words = [h.word.lower() for h in item.hints]
        assert len(words) == len(set(words))


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def test_round_trip_is_exact(man, tmp_path):
    path = man.save(tmp_path / "m.json")
    again = M.Manifest.load(path)
    assert again.name == man.name
    assert [i.as_dict() for i in again.items] == [i.as_dict() for i in man.items]


def test_saved_file_is_utf8_json_not_escaped(man, tmp_path):
    path = man.save(tmp_path / "m.json")
    text = path.read_text(encoding="utf-8")
    assert "Бишкек" in text  # ensure_ascii=False
    json.loads(text)


def test_load_tolerates_manifest_without_ref_tokens(tmp_path):
    """Older manifests predate stored reference tokens; loading must not crash."""
    p = tmp_path / "old.json"
    p.write_text(
        json.dumps({"name": "old", "items": [{"utterance_id": "x-1", "text": "Бишкек"}]}),
        encoding="utf-8",
    )
    loaded = M.Manifest.load(p)
    assert loaded.items[0].ref_tokens == []


# --------------------------------------------------------------------------
# Progress and recordings
# --------------------------------------------------------------------------


def test_progress_on_fresh_manifest(man):
    p = man.progress()
    assert p["total"] == 2 and p["recorded"] == 0 and p["percent_complete"] == 0.0


def _write_wav(path, seconds=1.5, sample_rate=48000, rms_dbfs=-23.0):
    n = int(sample_rate * seconds)
    t = np.arange(n) / sample_rate
    sig = np.sin(2 * np.pi * 220 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
    sig = sig * (10 ** (rms_dbfs / 20)) / np.sqrt(np.mean(sig**2))
    sig = np.clip(sig, -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes((sig * 32767).astype("<i2").tobytes())


def test_attach_recordings_matches_by_utterance_id(man, tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _write_wav(audio_dir / "tst-0001.wav")

    M.attach_recordings(man, audio_dir)
    first, second = man.items
    assert first.is_recorded and first.qc_passed is not None
    assert not second.is_recorded
    assert man.progress()["recorded"] == 1


def test_attach_recordings_records_qc_failure(man, tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _write_wav(audio_dir / "tst-0001.wav", rms_dbfs=-55.0)
    M.attach_recordings(man, audio_dir)
    assert man.items[0].qc_passed is False
    assert man.items[0].qc_problems


def test_attach_recordings_handles_corrupt_file(man, tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "tst-0001.wav").write_bytes(b"not audio")
    M.attach_recordings(man, audio_dir)
    assert man.items[0].qc_passed is False


def test_write_prompts_includes_text_and_hints(man, tmp_path):
    path = M.write_prompts(man, tmp_path / "prompts.txt")
    text = path.read_text(encoding="utf-8")
    for item in man.items:
        assert item.utterance_id in text
        assert item.text in text
