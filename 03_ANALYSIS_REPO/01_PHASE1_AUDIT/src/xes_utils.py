from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

XES_ATTR_TAGS = {"string", "date", "int", "float", "boolean", "id"}


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def find_dataset(repo_root: Path) -> Path:
    env = os.environ.get("BPI2019_XES")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.extend([
        repo_root / "data" / "raw" / "BPI_Challenge_2019.xes",
        repo_root / "BPI_Challenge_2019.xes",
    ])
    for parent in [repo_root, *repo_root.parents]:
        candidates.append(parent / "01_DATASET" / "BPI_Challenge_2019.xes")
        candidates.append(parent / "BPI_Challenge_2019.xes")
    seen = set()
    for path in candidates:
        path = path.expanduser()
        if str(path) in seen:
            continue
        seen.add(str(path))
        if path.exists() and path.is_file():
            return path.resolve()
    pretty = "\n".join(f" - {path}" for path in candidates)
    raise FileNotFoundError("Nu am gasit BPI_Challenge_2019.xes. Cautat in:\n" + pretty)



def parse_xes_value(attr_type: str, raw: Optional[str]) -> Any:
    if raw is None:
        return None
    try:
        if attr_type == "date":
            text = raw.replace("Z", "+00:00")
            return datetime.fromisoformat(text)
        if attr_type == "int":
            return int(raw)
        if attr_type == "float":
            return float(raw)
        if attr_type == "boolean":
            return raw.lower() == "true"
        return raw
    except Exception:
        return raw


def attr_dict(parent) -> Dict[str, Tuple[str, Any]]:
    out: Dict[str, Tuple[str, Any]] = {}
    for child in parent:
        tag = local_name(child.tag)
        if tag in XES_ATTR_TAGS:
            key = child.get("key")
            if key:
                out[key] = (tag, parse_xes_value(tag, child.get("value")))
    return out


def dt_to_iso(v: Any) -> str:
    if isinstance(v, datetime):
        return v.isoformat()
    return "" if v is None else str(v)
