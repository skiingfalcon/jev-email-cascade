from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_emails  # noqa: E402


def test_generation_is_deterministic() -> None:
    a = make_emails.generate(seed=0)
    b = make_emails.generate(seed=0)
    assert a == b


def test_ids_are_unique() -> None:
    rows = make_emails.generate(seed=0)
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))


def test_every_row_has_the_expected_label_keys() -> None:
    rows = make_emails.generate(seed=0)
    expected = set(make_emails._KEYS)
    for row in rows:
        assert set(row["labels"]) == expected
        assert row["labels"]["category"] in (
            "support",
            "sales",
            "billing",
            "hr",
            "vendor",
            "internal",
            "spam",
            "other",
        )
        assert 0 <= row["labels"]["priority"] <= 3


def test_covers_every_category() -> None:
    rows = make_emails.generate(seed=0)
    categories = {r["labels"]["category"] for r in rows}
    assert categories == {
        "support",
        "sales",
        "billing",
        "hr",
        "vendor",
        "internal",
        "spam",
        "other",
    }


def test_includes_the_specials() -> None:
    rows = make_emails.generate(seed=0)
    ids = {r["id"] for r in rows}
    assert any(i.startswith("special-injection-") for i in ids)
    assert any(i.startswith("special-negation-") for i in ids)
    assert any(i.startswith("special-other-") for i in ids)
    assert any(i.startswith("special-long-") for i in ids)
    injection_rows = [r for r in rows if r["id"].startswith("special-injection-")]
    assert all(r["labels"]["injection_suspected"] for r in injection_rows)
