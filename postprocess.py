"""Japanese text post-processing for subtitle-like readability."""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Deque


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


def normalize_for_repeat(text: str) -> str:
    s = text.strip()
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"([。！？、,.!?])\1+$", r"\1", s)
    return s.strip()


@dataclass
class RepeatDecision:
    blocked_repeat: bool
    repeat_reason: str
    normalized_text: str


class AntiRepeatGuard:
    def __init__(
        self,
        recent_cache_size: int = 10,
        short_text_len: int = 20,
        repeat_window_sec: float = 15.0,
        repeat_threshold: int = 2,
        silence_reset_sec: float = 2.5,
        draft_similarity_threshold: float = 0.98,
    ) -> None:
        self.recent_outputs: Deque[tuple[float, str]] = deque(maxlen=recent_cache_size)
        self.recent_hypotheses: Deque[str] = deque(maxlen=recent_cache_size)
        self.unconfirmed_tail: str = ""
        self.short_text_len = short_text_len
        self.repeat_window_sec = repeat_window_sec
        self.repeat_threshold = repeat_threshold
        self.silence_reset_sec = silence_reset_sec
        self.draft_similarity_threshold = draft_similarity_threshold
        self._last_event_ts = time.monotonic()

    def update_draft(self, draft_text: str) -> None:
        now = time.monotonic()
        self._maybe_reset_by_silence(now)
        normalized = normalize_for_repeat(draft_text)
        if normalized:
            self.unconfirmed_tail = normalized[-120:]
            self.recent_hypotheses.append(normalized)
        self._last_event_ts = now

    def should_block(self, text: str) -> RepeatDecision:
        now = time.monotonic()
        self._maybe_reset_by_silence(now)
        normalized = normalize_for_repeat(text)
        if not normalized:
            self._last_event_ts = now
            return RepeatDecision(False, "empty_after_normalize", normalized)

        if self.recent_outputs:
            last_ts, last_text = self.recent_outputs[-1]
            if normalized == last_text and now - last_ts <= 4.0:
                self._last_event_ts = now
                return RepeatDecision(True, "exact_duplicate_nearby", normalized)

        self.recent_outputs.append((now, normalized))
        self._last_event_ts = now
        return RepeatDecision(False, "accepted", normalized)

    def _is_draft_tail_similar(self, normalized: str) -> bool:
        if not self.unconfirmed_tail:
            return False
        if len(normalized.replace(" ", "")) > 30:
            return False
        ratio = SequenceMatcher(None, self.unconfirmed_tail[-80:], normalized[-80:]).ratio()
        return ratio >= self.draft_similarity_threshold

    def _maybe_reset_by_silence(self, now: float) -> None:
        if now - self._last_event_ts < self.silence_reset_sec:
            return
        self.recent_outputs.clear()
        self.recent_hypotheses.clear()
        self.unconfirmed_tail = ""
