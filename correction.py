"""Second-layer text correction interface for subtitle-like final output."""

from __future__ import annotations

import re


def apply_correction_layer(text: str) -> str:
    s = text.strip()
    if not s:
        return s

    s = merge_broken_sentences(s)
    s = remove_bad_fragments(s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def merge_broken_sentences(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return text
    merged = [lines[0]]
    for ln in lines[1:]:
        if merged[-1].endswith(("、", "と", "が", "で", "に", "を", "は")):
            merged[-1] += ln
        else:
            merged.append(ln)
    return "\n".join(merged)


def remove_bad_fragments(text: str) -> str:
    # drop obvious noise fragments like only punctuation or 1-char junk lines
    cleaned = []
    for ln in text.splitlines():
        pure = re.sub(r"[\W_]+", "", ln)
        if len(pure) <= 1:
            continue
        cleaned.append(ln)
    return "\n".join(cleaned)
