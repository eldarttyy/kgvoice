"""kgvoice.localize.audit: composing catalogue, placeholder, and register checks."""

from kgvoice.localize import Catalog, audit


def _pair(source: dict, target: dict):
    return Catalog.from_dict(source, locale="en"), Catalog.from_dict(target, locale="ky")


def test_clean_catalogue_is_clean():
    src, tgt = _pair(
        {"save": "Save", "cancel": "Cancel"},
        {"save": "Сактоо", "cancel": "Жокко чыгаруу"},
    )
    report = audit(src, tgt)
    assert report.is_clean
    assert report.coverage == 1.0
    assert report.missing_keys == []
    assert report.dirty_placeholder_keys == []
    assert report.suffix_collision_keys == []


def test_missing_translation_detected():
    src, tgt = _pair(
        {"save": "Save", "cancel": "Cancel"},
        {"save": "Сактоо"},
    )
    report = audit(src, tgt)
    assert not report.is_clean
    assert report.missing_keys == ["cancel"]
    assert report.coverage == 0.5


def test_extra_stale_key_detected():
    src, tgt = _pair(
        {"save": "Save"},
        {"save": "Сактоо", "old_key": "leftover"},
    )
    report = audit(src, tgt)
    assert report.extra_keys == ["old_key"]


def test_placeholder_dropped_from_target():
    src, tgt = _pair(
        {"welcome": "Welcome, {name}!"},
        {"welcome": "Кош келиңиз!"},  # {name} silently dropped
    )
    report = audit(src, tgt)
    assert report.dirty_placeholder_keys == ["welcome"]
    issue = report.placeholder_issues[0]
    assert issue.missing == ["name"]
    assert issue.extra == []
    assert not report.is_clean


def test_placeholder_invented_in_target():
    src, tgt = _pair(
        {"save": "Save"},
        {"save": "{count} Сактоо"},  # {count} invented, not in source
    )
    report = audit(src, tgt)
    issue = report.placeholder_issues[0]
    assert issue.extra == ["count"]
    assert issue.missing == []


def test_matching_placeholders_are_clean():
    src, tgt = _pair(
        {"count": "{count} items"},
        {"count": "{count} нерсе"},
    )
    report = audit(src, tgt)
    assert report.placeholder_issues == []


def test_suffix_collision_detected_on_dative():
    # {name}ге hardcodes the dative; wrong for Айбек (-ке), Нурлан (-га), Гүл (-гө).
    src, tgt = _pair(
        {"invite": "Send an invite to {name}"},
        {"invite": "{name}ге чакыруу жөнөттүңүз"},
    )
    report = audit(src, tgt)
    assert report.suffix_collision_keys == ["invite"]
    collision = report.suffix_issues[0].collisions[0]
    assert collision.label == "dative"
    wrong = collision.wrong_for()
    assert any(value == "Айбек" for value, _, _ in wrong)
    assert not report.is_clean


def test_no_suffix_collision_when_suffix_selected_correctly():
    # No hardcoded suffix glued to the placeholder -> nothing to flag.
    src, tgt = _pair(
        {"invite": "Send an invite to {name}"},
        {"invite": "Чакыруу: {name}"},
    )
    report = audit(src, tgt)
    assert report.suffix_issues == []


def test_register_consistency_detected():
    src, tgt = _pair(
        {"a": "Save", "b": "Cancel", "c": "Delete"},
        {
            "a": "Сактаңыз",  # formal (-ңыз)
            "b": "Жокко чыгарыңыз",  # formal
            "c": "Сеники өчүрүлсүн",  # familiar pronoun — minority register
        },
    )
    report = audit(src, tgt)
    assert report.register.dominant == "formal"
    assert "c" in report.register.minority_keys
    assert not report.is_clean


def test_format_produces_readable_markdown_with_all_sections():
    src, tgt = _pair(
        {"invite": "Send an invite to {name}", "save": "Save"},
        {"invite": "{name}ге чакыруу жөнөттүңүз"},  # missing 'save' too
    )
    report = audit(src, tgt)
    text = report.format()
    assert text.startswith("# Localisation audit")
    assert "ISSUES FOUND" in text
    assert "## Missing translations" in text
    assert "`save`" in text
    assert "## Suffix collisions" in text
    assert "dative" in text


def test_as_dict_is_json_serialisable():
    import json

    src, tgt = _pair(
        {"invite": "Send an invite to {name}"},
        {"invite": "{name}ге чакыруу жөнөттүңүз"},
    )
    report = audit(src, tgt)
    payload = report.as_dict()
    json.dumps(payload, ensure_ascii=False)  # must not raise
    assert payload["summary"]["clean"] is False
    assert payload["suffix_issues"][0]["key"] == "invite"


def test_audit_files_round_trip(tmp_path):
    from kgvoice.localize import audit_files

    src_path = tmp_path / "en.json"
    tgt_path = tmp_path / "ky.json"
    src_path.write_text('{"save": "Save"}', encoding="utf-8")
    tgt_path.write_text('{"save": "Сактоо"}', encoding="utf-8")

    report = audit_files(src_path, tgt_path)
    assert report.is_clean
    assert report.pair.source.locale == "en"
    assert report.pair.target.locale == "ky"
