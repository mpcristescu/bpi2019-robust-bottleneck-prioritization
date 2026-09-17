from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from lxml import etree


EXPECTED_CASES = 251_266
EXPECTED_EVENTS = 1_587_374
EVENT_BATCH_SIZE = 100_000
CASE_KEYS = [
    "Company",
    "Document Type",
    "Item Type",
    "Item Category",
    "Spend area text",
    "Sub spend area text",
    "Spend classification text",
    "Goods Receipt",
    "GR-Based Inv. Verif.",
    "Source",
]


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def attributes(element: etree._Element) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for child in element:
        key = child.attrib.get("key", local_name(child.tag))
        value: Any = child.attrib.get("value")
        tag = local_name(child.tag)
        if tag == "date" and value:
            try:
                value = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                pass
        elif tag == "int" and value:
            try:
                value = int(value)
            except ValueError:
                pass
        elif tag == "float" and value:
            try:
                value = float(value)
            except ValueError:
                pass
        elif tag == "boolean" and value:
            value = value.lower() == "true"
        result[key] = value
    return result


def resource_kind(resource: str) -> str:
    value = (resource or "").strip()
    if value.startswith("user_"):
        return "human_user"
    if value.startswith("batch_"):
        return "batch_user"
    return "no_recorded_user"


def scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def rebuild_phase2_pair(project_root: Path) -> tuple[Path, Path]:
    dataset = project_root / "01_DATASET" / "BPI_Challenge_2019.xes"
    if not dataset.exists():
        raise FileNotFoundError(f"Nu exista datasetul pentru reconstruirea cohortei: {dataset}")
    processed = project_root / "04_ANALYSIS_RESULTS" / "03_PHASE2_PROCESS_STRUCTURE" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    event_path = processed / "events_primary_cohort.parquet"
    case_path = processed / "cases_primary_cohort.parquet"
    if event_path.exists():
        event_path.unlink()
    if case_path.exists():
        case_path.unlink()

    trace_count = event_count = kept_cases = kept_events = 0
    event_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    writer = None

    def flush() -> None:
        nonlocal writer
        if not event_rows:
            return
        frame = pd.DataFrame(event_rows)
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(event_path, table.schema, compression="zstd")
        writer.write_table(table)
        event_rows.clear()

    context = etree.iterparse(str(dataset), events=("end",), tag="{*}trace", huge_tree=True, recover=True)
    for _, trace in context:
        trace_count += 1
        trace_attrs = attributes(trace)
        case_id = str(trace_attrs.get("concept:name", f"__missing_case_{trace_count}"))
        raw_events: list[dict[str, Any]] = []
        for child in trace:
            if local_name(child.tag) != "event":
                continue
            event_count += 1
            event_attrs = attributes(child)
            timestamp = event_attrs.get("time:timestamp")
            activity = str(event_attrs.get("concept:name", "__MISSING_ACTIVITY__"))
            resource = str(event_attrs.get("org:resource", "") or "")
            raw_events.append(
                {
                    "timestamp": timestamp,
                    "activity": activity,
                    "resource": resource,
                    "resource_kind": resource_kind(resource),
                    "cumulative_net_worth_eur": scalar(event_attrs.get("Cumulative net worth (EUR)")),
                }
            )
        timestamps = [event["timestamp"] for event in raw_events if isinstance(event["timestamp"], datetime)]
        if not timestamps:
            include = False
        else:
            case_min = min(timestamps)
            case_max = max(timestamps)
            has_extreme = any(ts.year < 2017 or ts.year > 2019 for ts in timestamps)
            include = not has_extreme and case_min.year == 2018
        if include:
            kept_cases += 1
            kept_events += len(raw_events)
            case_min = min(timestamps)
            case_max = max(timestamps)
            start_month = case_min.strftime("%Y-%m")
            activities = [event["activity"] for event in raw_events]
            resources = [event["resource"] for event in raw_events]
            kinds = [event["resource_kind"] for event in raw_events]
            case_rows.append(
                {
                    "case_id": case_id,
                    "case_start": case_min,
                    "case_end": case_max,
                    "case_start_month": start_month,
                    "case_end_year": case_max.year,
                    "has_2019_followup": bool(case_max.year == 2019),
                    "cycle_time_hours": (case_max - case_min).total_seconds() / 3600.0,
                    "events_per_case": len(raw_events),
                    "unique_activities": len(set(activities)),
                    "unique_recorded_resources": len({resource for resource in resources if resource_kind(resource) != "no_recorded_user"}),
                    "human_user_events": sum(kind == "human_user" for kind in kinds),
                    "batch_user_events": sum(kind == "batch_user" for kind in kinds),
                    "no_recorded_user_events": sum(kind == "no_recorded_user" for kind in kinds),
                    "start_activity": activities[0] if activities else "",
                    "end_activity": activities[-1] if activities else "",
                    "variant": " -> ".join(activities),
                    "temporal_order_ok": all(
                        raw_events[i]["timestamp"] <= raw_events[i + 1]["timestamp"]
                        for i in range(len(raw_events) - 1)
                        if isinstance(raw_events[i]["timestamp"], datetime) and isinstance(raw_events[i + 1]["timestamp"], datetime)
                    ),
                    **{f"case_{key}": scalar(trace_attrs.get(key)) for key in CASE_KEYS},
                }
            )
            previous = None
            for index, event in enumerate(raw_events):
                timestamp = event["timestamp"]
                gap = None
                transition = None
                previous_activity = None
                if previous is not None and isinstance(timestamp, datetime) and isinstance(previous["timestamp"], datetime):
                    gap = (timestamp - previous["timestamp"]).total_seconds() / 3600.0
                    previous_activity = previous["activity"]
                    transition = f"{previous_activity} -> {event['activity']}"
                event_rows.append(
                    {
                        "case_id": case_id,
                        "event_index": index,
                        "timestamp": timestamp,
                        "event_year": timestamp.year if isinstance(timestamp, datetime) else None,
                        "event_month": timestamp.strftime("%Y-%m") if isinstance(timestamp, datetime) else None,
                        "case_start_month": start_month,
                        "activity": event["activity"],
                        "resource": event["resource"],
                        "resource_kind": event["resource_kind"],
                        "cumulative_net_worth_eur": event["cumulative_net_worth_eur"],
                        "previous_activity": previous_activity,
                        "transition": transition,
                        "inter_event_gap_hours": gap,
                        "is_first_event": index == 0,
                        "is_last_event": index == len(raw_events) - 1,
                    }
                )
                previous = event
                if len(event_rows) >= EVENT_BATCH_SIZE:
                    flush()
        trace.clear()
        parent = trace.getparent()
        if parent is not None:
            while trace.getprevious() is not None:
                del parent[0]
    flush()
    if writer is not None:
        writer.close()
    pd.DataFrame(case_rows).to_parquet(case_path, index=False, compression="zstd")
    if kept_cases != EXPECTED_CASES or kept_events != EXPECTED_EVENTS:
        raise RuntimeError(
            f"Reconstruirea cohortei nu corespunde asteptarilor: {kept_cases} cazuri, {kept_events} evenimente"
        )
    return event_path.resolve(), case_path.resolve()
