"""Domain hotword dictionary loading and correction utilities."""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_HOTWORDS = {
    "法連想": "報連相",
    "外見先": "外勤先",
    "さて": "査定",
    "評価性度": "評価制度",
}


def load_hotwords(path: str | None) -> dict[str, str]:
    mapping = dict(DEFAULT_HOTWORDS)
    if not path:
        return mapping

    p = Path(path)
    if not p.exists():
        return mapping

    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            mapping.update({str(k): str(v) for k, v in data.items()})
        return mapping

    # txt: wrong=>correct per line
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=>" in line:
            wrong, right = line.split("=>", 1)
            mapping[wrong.strip()] = right.strip()
    return mapping


def apply_hotwords(text: str, mapping: dict[str, str]) -> str:
    out = text
    for wrong, right in mapping.items():
        out = out.replace(wrong, right)
    return out
