"""Offset-preserving clinical text preprocessing.

The baseline never rewrites the source text.  Search and sentence views retain absolute
character offsets, so every downstream :class:`TextSpan` can be checked by slicing the
original document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from backend.app.clinical_nlp.versions import PREPROCESSING_VERSION

_HEADER_RE = re.compile(r"(?im)^[ \t]*(?P<header>[a-z][a-z0-9 /&_-]{1,48})[ \t]*:[ \t]*")
_ABBREVIATIONS = {
    "dr.",
    "mr.",
    "mrs.",
    "ms.",
    "prof.",
    "vs.",
    "e.g.",
    "i.e.",
}


@dataclass(frozen=True, slots=True)
class SentenceSpan:
    """A sentence-like region with offsets into the untouched input."""

    text: str
    start: int
    end: int
    section: str | None = None


@dataclass(frozen=True, slots=True)
class PreprocessedText:
    """Immutable, offset-safe preprocessing result."""

    text: str
    sentences: tuple[SentenceSpan, ...]
    version: str = PREPROCESSING_VERSION

    def sentence_at(self, offset: int) -> SentenceSpan | None:
        for sentence in self.sentences:
            if sentence.start <= offset < sentence.end:
                return sentence
        return None


def _is_abbreviation(text: str, punctuation_index: int) -> bool:
    prefix = text[max(0, punctuation_index - 8) : punctuation_index + 1].casefold()
    return any(prefix.endswith(abbreviation) for abbreviation in _ABBREVIATIONS)


def _raw_sentence_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    index = 0
    length = len(text)

    while index < length:
        character = text[index]
        boundary = character in "\r\n"
        if character in ".!?":
            next_character = text[index + 1] if index + 1 < length else ""
            boundary = (not next_character or next_character.isspace()) and not _is_abbreviation(
                text, index
            )

        if boundary:
            end = index + (1 if character not in "\r\n" else 0)
            ranges.append((start, end))
            if character == "\r" and index + 1 < length and text[index + 1] == "\n":
                index += 1
            start = index + 1
        index += 1

    if start < length:
        ranges.append((start, length))
    return ranges


def _trim_range(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _section_for_offset(headers: list[tuple[int, str]], offset: int) -> str | None:
    section: str | None = None
    for header_offset, header in headers:
        if header_offset > offset:
            break
        section = header
    return section


def preprocess(text: str) -> PreprocessedText:
    """Split *text* without normalizing or losing source offsets."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    headers = [
        (match.start(), re.sub(r"\s+", " ", match.group("header").strip()).casefold())
        for match in _HEADER_RE.finditer(text)
    ]
    sentences: list[SentenceSpan] = []
    for raw_start, raw_end in _raw_sentence_ranges(text):
        start, end = _trim_range(text, raw_start, raw_end)
        if start == end:
            continue
        sentence_text = text[start:end]
        sentences.append(
            SentenceSpan(
                text=sentence_text,
                start=start,
                end=end,
                section=_section_for_offset(headers, start),
            )
        )

    return PreprocessedText(text=text, sentences=tuple(sentences))


class OffsetSafePreprocessor:
    """Object-oriented facade used by the service and integrations."""

    version = PREPROCESSING_VERSION

    def process(self, text: str) -> PreprocessedText:
        return preprocess(text)
