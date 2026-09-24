"""Minimal Chinese patent-text preprocessing."""

from __future__ import annotations

import re
import unicodedata

import jieba

TOKEN_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9+#.\-]+")


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = "".join(c if not unicodedata.category(c).startswith("C") else " " for c in text)
    return re.sub(r"\s+", " ", text).strip()


def combine_text(row: dict[str, object], fields: list[str]) -> str:
    return " ".join(
        text for field in fields if (text := clean_text(row.get(field, "")))
    )


def tokenize(text: object) -> list[str]:
    return [
        token
        for token in (clean_text(x) for x in jieba.cut(clean_text(text)))
        if token and TOKEN_RE.fullmatch(token)
    ]
