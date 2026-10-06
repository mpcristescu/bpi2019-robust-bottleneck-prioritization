"""Stream event logs without conditioning historical eligibility on later events."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from lxml import etree


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_log(path: Path, log: str, cache: Path) -> tuple[pd.DataFrame, dict]:
    """One row per recorded event, including events without an observed successor.

    Event timestamps are availability proxies. No ingestion timestamps are present.
    BPI 2019 case eligibility uses the first recorded timestamp, not a condition on
    the absence of future anomalies. Out-of-range events are excluded individually.
    BPI 2017 retains COMPLETE events only. Original order resolves timestamp ties.
    The cache is private and must not be committed to the public repository.
    """
    cache.mkdir(parents=True, exist_ok=True)
    pq_path = cache / f"{log}_origins.parquet"
    meta_path = cache / f"{log}_source.json"
    signature = {"file_size": path.stat().st_size, "source_sha256": checksum(path), "parser_version": "1.1.0"}
    if pq_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if all(meta.get(key) == value for key, value in signature.items()):
            return pd.read_parquet(pq_path), meta

    rows = []
    stats = Counter()
    activities = set()
    stamp_min, stamp_max = None, None
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as handle:
        for _, trace in etree.iterparse(handle, events=("end",), tag="{*}trace", huge_tree=True):
            stats["raw_cases"] += 1
            events = []
            for order, event in enumerate(trace.findall("{*}event")):
                stats["raw_events"] += 1
                attrs = {child.get("key"): child.get("value") for child in event}
                if log == "bpi2017" and (attrs.get("lifecycle:transition") or "").lower() != "complete":
                    stats["non_complete_excluded"] += 1
                    continue
                try:
                    stamp = datetime.fromisoformat(attrs["time:timestamp"].replace("Z", "+00:00")).astimezone(timezone.utc)
                except (KeyError, AttributeError, ValueError):
                    stats["invalid_timestamps"] += 1
                    continue
                name = attrs.get("concept:name")
                if not name:
                    stats["missing_activity"] += 1
                    continue
                events.append((stamp, order, name))
            events.sort(key=lambda x: (x[0], x[1]))
            if events and (log != "bpi2019" or events[0][0].year == 2018):
                if log == "bpi2019":
                    filtered = [e for e in events if 2017 <= e[0].year <= 2019]
                    stats["out_of_range_events_excluded"] += len(events) - len(filtered)
                    events = filtered
                if events:
                    case = stats["retained_cases"]
                    stats["retained_cases"] += 1
                    stats["retained_events"] += len(events)
                    for i, (origin, order, activity) in enumerate(events):
                        activities.add(activity)
                        stamp_min = origin if stamp_min is None else min(stamp_min, origin)
                        stamp_max = origin if stamp_max is None else max(stamp_max, origin)
                        if i + 1 < len(events):
                            dest, _, target = events[i + 1]
                            gap = (dest - origin).total_seconds() / 3600.0
                            stats["timestamp_ties"] += int(gap == 0)
                            transition = f"{activity} -> {target}"
                        else:
                            dest, gap, transition = None, float("nan"), None
                        rows.append((case, origin, dest, activity, transition, gap))
            trace.clear()
            while trace.getprevious() is not None:
                del trace.getparent()[0]
            if stats["raw_cases"] % 50000 == 0:
                print(log, "parsed cases", stats["raw_cases"], flush=True)
    frame = pd.DataFrame(rows, columns=["case", "origin", "destination", "origin_activity", "transition", "delay"])
    frame["case"] = frame["case"].astype("int32")
    frame["origin"] = pd.to_datetime(frame["origin"], utc=True)
    frame["destination"] = pd.to_datetime(frame["destination"], utc=True)
    meta = dict(signature, **dict(stats), file_name=path.name, activities=len(activities),
                observed_start=stamp_min.isoformat(), observed_end=stamp_max.isoformat())
    frame.to_parquet(pq_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(log, meta, flush=True)
    return frame, meta


def times(log: str) -> dict:
    year = 2018 if log == "bpi2019" else 2016
    cutoff = pd.Timestamp(f"{year}-07-01", tz="UTC")
    observation_end = (pd.Timestamp("2019-01-18 13:34:00", tz="UTC")
                       if log == "bpi2019" else pd.Timestamp("2017-02-01", tz="UTC"))
    return {"start": pd.Timestamp(f"{year}-01-01", tz="UTC"), "decision": cutoff,
            "q3_end": pd.Timestamp(f"{year}-10-01", tz="UTC"), "observation_end": observation_end}


def historical(frame: pd.DataFrame, start, end, decision) -> pd.DataFrame:
    return frame.loc[(frame.origin >= start) & (frame.origin < end)
                     & frame.destination.notna() & (frame.destination < decision)].copy()


def holdout(frame: pd.DataFrame, start, end, observation_end) -> pd.DataFrame:
    return frame.loc[(frame.origin >= start) & (frame.origin < end)
                     & frame.destination.notna() & (frame.destination <= observation_end)].copy()


def availability_audit(frame: pd.DataFrame, start, end, decision, label: str) -> dict:
    origins = frame.loc[(frame.origin >= start) & (frame.origin < end)]
    available = origins.destination.notna() & (origins.destination < decision)
    later = origins.destination.notna() & (origins.destination >= decision)
    no_successor = origins.destination.isna()
    return {"window": label, "origin_start": start.isoformat(), "origin_end_exclusive": end.isoformat(),
            "information_cutoff": decision.isoformat(), "origin_events": len(origins),
            "available_pairs": int(available.sum()), "later_recorded_successors": int(later.sum()),
            "no_recorded_successor": int(no_successor.sum()),
            "pending_at_cutoff_pct": 100 * float((~available).mean()) if len(origins) else None}
