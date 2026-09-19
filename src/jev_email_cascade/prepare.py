"""Load synthetic emails and prepare the state sent to a decision backend.

TypeSafe's own guidance is "retrieve first, judge second": accuracy falls with padding, so only
the fields a question needs go in the state, and quoted history / signatures are stripped before
the email ever reaches a backend.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_BODY_CHARS = 6000

# Heuristics for where a reply's own text ends and quoted history begins.
_QUOTE_MARKERS = (
    re.compile(r"^On .+ wrote:\s*$", re.MULTILINE),
    re.compile(r"^-{2,}\s*Original Message\s*-{2,}", re.MULTILINE),
)
_FROM_LINE = re.compile(r"^From:\s", re.MULTILINE)
_QUOTED_RUN = re.compile(r"(?:^>.*\n?){3,}", re.MULTILINE)
_SIGNATURE = re.compile(r"\n--\s*\n.*\Z", re.DOTALL)


@dataclass(frozen=True)
class Email:
    id: str
    sender: str
    to: str
    subject: str
    date: str
    body: str
    labels: dict[str, Any]


@dataclass(frozen=True)
class Prepared:
    email: Email
    state: dict[str, str]
    truncated: bool


def load_emails(path: Path) -> list[Email]:
    emails: list[Email] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        emails.append(
            Email(
                id=row["id"],
                sender=row["from"],
                to=row["to"],
                subject=row["subject"],
                date=row["date"],
                body=row["body"],
                labels=row["labels"],
            )
        )
    return emails


def _cut_quoted_history(body: str) -> str:
    cut_at = len(body)
    for pattern in _QUOTE_MARKERS:
        m = pattern.search(body)
        if m:
            cut_at = min(cut_at, m.start())
    m = _FROM_LINE.search(body)
    if m:
        # Only treat a "From:" line as quoted history if it isn't in the first couple of lines
        # (a few of our synthetic bodies legitimately open with "From the desk of ...").
        line_no = body.count("\n", 0, m.start())
        if line_no > 2:
            cut_at = min(cut_at, m.start())
    m = _QUOTED_RUN.search(body)
    if m:
        cut_at = min(cut_at, m.start())
    return body[:cut_at]


def _strip_signature(body: str) -> str:
    return _SIGNATURE.sub("", body)


def prepare(email: Email) -> Prepared:
    body = _cut_quoted_history(email.body)
    body = _strip_signature(body)
    body = body.strip()
    truncated = False
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS] + "\n[truncated]"
        truncated = True
    state = {
        "subject": email.subject,
        "from": email.sender,
        "received": email.date,
        "body": body,
    }
    return Prepared(email=email, state=state, truncated=truncated)
