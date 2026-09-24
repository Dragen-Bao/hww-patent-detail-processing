"""Chinese patent text preprocessing for the KPST baseline."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

TOKEN_KEEP_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9+#.\-]+")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_patent_text(value: object) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = "".join(
        character if not unicodedata.category(character).startswith("C") else " "
        for character in text
    )
    return WHITESPACE_RE.sub(" ", text).strip()


def combine_patent_text(row: dict[str, object], fields: Iterable[str]) -> str:
    """Combine explicitly configured source fields in order."""
    parts: list[str] = []
    for field in fields:
        value = normalize_patent_text(row.get(field, ""))
        if value:
            parts.append(value)
    return " ".join(parts)


def _load_word_set(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


@dataclass
class PatentTextPreprocessor:
    """Configurable word-level tokenizer.

    Jieba is an explicit implementation choice. The supplied appendix does not
    publish the authors' exact tokenizer dictionary or stopword list.
    """

    stopwords: set[str] | None = None
    user_dictionary: Path | None = None
    keep_single_char: bool = True
    tokenizer: Callable[[str], Iterable[str]] | None = None

    @classmethod
    def from_files(
        cls,
        *,
        stopwords_path: Path | None = None,
        user_dictionary: Path | None = None,
        keep_single_char: bool = True,
    ) -> "PatentTextPreprocessor":
        return cls(
            stopwords=_load_word_set(stopwords_path),
            user_dictionary=user_dictionary,
            keep_single_char=keep_single_char,
        )

    def _jieba_tokenizer(self) -> Callable[[str], Iterable[str]]:
        try:
            import jieba
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Jieba is required for the default tokenizer. Install jieba or "
                "pass a custom tokenizer callable."
            ) from exc
        if self.user_dictionary is not None:
            jieba.load_userdict(str(self.user_dictionary))
        return jieba.cut

    def tokenize(self, text: object) -> list[str]:
        normalized = normalize_patent_text(text)
        if not normalized:
            return []
        tokenizer = self.tokenizer or self._jieba_tokenizer()
        stopwords = self.stopwords or set()
        tokens: list[str] = []
        for raw in tokenizer(normalized):
            token = normalize_patent_text(raw).strip()
            if not token or token in stopwords:
                continue
            if not TOKEN_KEEP_RE.fullmatch(token):
                continue
            if (
                not self.keep_single_char
                and len(token) == 1
                and re.fullmatch(r"[\u4e00-\u9fff]", token)
            ):
                continue
            tokens.append(token)
        return tokens
