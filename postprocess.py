"""Japanese text post-processing for subtitle-like readability."""

from __future__ import annotations

import re


def cleanup_text(text: str) -> str:
    s = text.strip()
    if not s:
        return s

    s = remove_stutter_repeats(s)
    s = remove_duplicate_phrases(s)
    s = re.sub(r"([。！？、,.!?])\1+", r"\1", s)
    s = re.sub(r"\s+", " ", s)
    s = complete_punctuation(s)
    s = split_for_subtitles(s)
    return s.strip()


def remove_stutter_repeats(text: str) -> str:
    # e.g., はい はい はい -> はい
    tokens = text.split()
    out = []
    for t in tokens:
        if len(out) >= 2 and out[-1] == t and out[-2] == t:
            continue
        out.append(t)
    return " ".join(out)


def remove_duplicate_phrases(text: str) -> str:
    words = text.split()
    i = 0
    out = []
    while i < len(words):
        dedup = False
        for span in (4, 3, 2, 1):
            if i + span * 2 > len(words):
                continue
            p = words[i : i + span]
            if words[i + span : i + span * 2] == p:
                out.extend(p)
                i += span * 2
                while i + span <= len(words) and words[i : i + span] == p:
                    i += span
                dedup = True
                break
        if not dedup:
            out.append(words[i])
            i += 1
    return " ".join(out)


def complete_punctuation(text: str) -> str:
    if text and text[-1] not in "。！？!?":
        return text + "。"
    return text


def split_for_subtitles(text: str) -> str:
    # soft segmentation for readability
    text = text.replace("、ただ", "、\nただ")
    text = text.replace("。また", "。\nまた")
    return text
