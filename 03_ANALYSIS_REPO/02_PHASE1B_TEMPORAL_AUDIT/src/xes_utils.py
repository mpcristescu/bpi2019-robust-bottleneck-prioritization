from __future__ import annotations
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

XES_ATTR_TAGS = {"string", "date", "int", "float", "boolean", "id"}

def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag

def find_dataset(repo_root: Path) -> Path:
    env = os.environ.get("BPI2019_XES")
    candidates = []
    if env:
        candidates.append(Path(env))
    # local/repo locations
    candidates += [
        repo_root / "data" / "raw" / "BPI_Challenge_2019.xes",
        repo_root / "BPI_Challenge_2019.xes",
    ]
    # project-style locations: walk upward and look for 01_DATASET
    for parent in [repo_root, *repo_root.parents]:
        candidates.append(parent / "01_DATASET" / "BPI_Challenge_2019.xes")
        candidates.append(parent / "BPI_Challenge_2019.xes")
    seen = set()
    for p in candidates:
        p = p.expanduser()
        if str(p) in seen:
            continue
        seen.add(str(p))
        if p.exists() and p.is_file():
            return p.resolve()
    pretty = "\n".join(f" - {p}" for p in candidates)
    raise FileNotFoundError("Nu am gasit BPI_Challenge_2019.xes. Cautat in:\n" + pretty)

def parse_xes_value(attr_type: str, raw: Optional[str]) -> Any:
    if raw is None:
        return None
    try:
        if attr_type == "date":
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if attr_type == "int": return int(raw)
        if attr_type == "float": return float(raw)
        if attr_type == "boolean": return raw.lower() == "true"
        return raw
    except Exception:
        return raw

def attr_dict(parent) -> Dict[str, Tuple[str, Any, Optional[str]]]:
    out = {}
    for child in parent:
        tag = local_name(child.tag)
        if tag in XES_ATTR_TAGS:
            key = child.get("key")
            if key:
                raw = child.get("value")
                out[key] = (tag, parse_xes_value(tag, raw), raw)
    return out
