from __future__ import annotations

from jev_email_cascade.prepare import MAX_BODY_CHARS, Email, prepare


def _email(body: str, **kw) -> Email:
    return Email(
        id=kw.get("id", "e1"),
        sender=kw.get("sender", "a@example.com"),
        to=kw.get("to", "inbox@corp.example"),
        subject=kw.get("subject", "Subject"),
        date=kw.get("date", "2026-09-11T00:00:00Z"),
        body=body,
        labels=kw.get("labels", {}),
    )


def test_strips_on_wrote_quoted_history() -> None:
    body = (
        "My real message.\n\nOn Tue, Sep 8, 2026 at 3:14 PM Dana wrote:\n"
        "> old stuff\n> more old stuff"
    )
    p = prepare(_email(body))
    assert "My real message." in p.state["body"]
    assert "old stuff" not in p.state["body"]


def test_strips_original_message_block() -> None:
    body = "Reply text here.\n\n-----Original Message-----\nFrom: someone\nOld content"
    p = prepare(_email(body))
    assert "Reply text here." in p.state["body"]
    assert "Old content" not in p.state["body"]


def test_strips_long_quoted_run() -> None:
    body = "New text.\n\n> line one\n> line two\n> line three\n> line four"
    p = prepare(_email(body))
    assert "New text." in p.state["body"]
    assert "line one" not in p.state["body"]


def test_from_line_after_first_lines_is_quoted_history() -> None:
    body = "Short reply.\n\nSome other line.\nFrom: dana@example.com\nSubject: old\nOld body text"
    p = prepare(_email(body))
    assert "Short reply." in p.state["body"]
    assert "Old body text" not in p.state["body"]


def test_from_line_in_first_lines_is_kept() -> None:
    body = "From the desk of the CEO:\n\nPlease review the attached proposal."
    p = prepare(_email(body))
    assert "From the desk of the CEO" in p.state["body"]


def test_strips_trailing_signature() -> None:
    body = "Message body.\n\n--\nDana Ruiz | Acme Supply"
    p = prepare(_email(body))
    assert "Message body." in p.state["body"]
    assert "Acme Supply" not in p.state["body"]


def test_caps_body_length() -> None:
    body = "x" * (MAX_BODY_CHARS + 500)
    p = prepare(_email(body))
    assert len(p.state["body"]) <= MAX_BODY_CHARS + len("\n[truncated]")
    assert p.state["body"].endswith("[truncated]")
    assert p.truncated is True


def test_short_body_not_truncated() -> None:
    p = prepare(_email("short message"))
    assert p.truncated is False
    assert p.state["body"] == "short message"


def test_state_has_expected_fields() -> None:
    p = prepare(_email("body text", subject="Subj", sender="a@b.com", date="2026-01-01T00:00:00Z"))
    assert set(p.state) == {"subject", "from", "received", "body"}
    assert p.state["subject"] == "Subj"
    assert p.state["from"] == "a@b.com"
    assert p.state["received"] == "2026-01-01T00:00:00Z"
