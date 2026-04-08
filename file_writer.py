"""Writers for clean TXT/SRT subtitle-like outputs with overwrite support."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from transcriber import TranscriptionUpdate


class TranscriptFileWriter:
    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.txt_path: Optional[Path] = None
        self.srt_path: Optional[Path] = None

    def open(self, txt_path: str, export_srt: bool = False) -> None:
        self.txt_path = Path(txt_path)
        self.txt_path.parent.mkdir(parents=True, exist_ok=True)
        if export_srt:
            self.srt_path = self.txt_path.with_suffix(".srt")
        else:
            self.srt_path = None

    def rewrite_all(self, ordered_updates: list[TranscriptionUpdate]) -> None:
        if self.txt_path is None:
            return

        content_lines = []
        for u in ordered_updates:
            content_lines.append(f"{u.timestamp}\n{u.text}\n")
        self.txt_path.write_text("\n".join(content_lines), encoding="utf-8")

        if self.srt_path is not None:
            self._write_srt(ordered_updates)

    def _write_srt(self, updates: list[TranscriptionUpdate]) -> None:
        lines = []
        for idx, u in enumerate(updates, start=1):
            # No exact timestamp intervals available in GUI-overwrite mode; keep simple rolling placeholders.
            lines.append(str(idx))
            lines.append(f"00:00:{idx:02d},000 --> 00:00:{idx+1:02d},000")
            lines.append(u.text)
            lines.append("")
        self.srt_path.write_text("\n".join(lines), encoding="utf-8")
