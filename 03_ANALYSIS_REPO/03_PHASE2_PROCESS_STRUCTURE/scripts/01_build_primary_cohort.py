from __future__ import annotations
import csv, json, sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from lxml import etree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.xes_utils import attr_dict, find_dataset, local_name, resource_kind, val

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT = PROJECT_ROOT / "04_ANALYSIS_RESULTS" / "03_PHASE2_PROCESS_STRUCTURE"
PROCESSED = OUT / "processed"
OUT.mkdir(exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

PRIMARY_START_YEAR = 2018
VALID_MIN_YEAR = 2017
VALID_MAX_YEAR = 2019
EVENT_BATCH_SIZE = 100_000
EXPECTED_CASES = 251_266
EXPECTED_EVENTS = 1_587_374

CASE_KEYS = [
    "Company", "Document Type", "Item Type", "Item Category",
    "Spend area text", "Sub spend area text", "Spend classification text",
    "Goods Receipt", "GR-Based Inv. Verif.", "Source"
]

def clean_scalar(x):
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)

def flush_events(writer, rows, path):
    if not rows:
        return writer
    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table)
    rows.clear()
    return writer

def write_rows_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

def main():
    dataset = find_dataset(REPO)
    print(f"Dataset: {dataset}")
    print(f"Size: {dataset.stat().st_size/1024**2:.1f} MB")
    print("Phase 2A: building primary 2018-start cohort...")

    event_path = PROCESSED / "events_primary_cohort.parquet"
    case_path = PROCESSED / "cases_primary_cohort.parquet"
    if event_path.exists(): event_path.unlink()
    if case_path.exists(): case_path.unlink()

    trace_count = event_count = 0
    kept_cases = kept_events = 0
    excluded_cases = excluded_events = 0
    order_violation_cases = 0
    resource_counts = Counter()
    start_month_counts = Counter()
    exclusion_reasons = Counter()
    exclusion_rows = []
    case_rows = []
    event_rows = []
    writer = None

    context = etree.iterparse(str(dataset), events=("end",), tag="{*}trace", huge_tree=True, recover=True)
    for _, trace in context:
        trace_count += 1
        cattrs = attr_dict(trace)
        case_id = str(val(cattrs, "concept:name", f"__missing_case_{trace_count}"))
        raw_events = []
        timestamps = []
        temporal_order_ok = True
        prev_ts_check = None

        for child in trace:
            if local_name(child.tag) != "event":
                continue
            event_count += 1
            e = attr_dict(child)
            ts = val(e, "time:timestamp")
            act = str(val(e, "concept:name", "__MISSING_ACTIVITY__"))
            res = val(e, "org:resource")
            res = "" if res is None else str(res)
            worth = val(e, "Cumulative net worth (EUR)")
            if isinstance(ts, datetime):
                timestamps.append(ts)
                if prev_ts_check is not None and ts < prev_ts_check:
                    temporal_order_ok = False
                prev_ts_check = ts
            raw_events.append({
                "timestamp": ts,
                "activity": act,
                "resource": res,
                "resource_kind": resource_kind(res),
                "cumulative_net_worth_eur": clean_scalar(worth),
            })

        total_case_events = len(raw_events)
        if not timestamps:
            reason = "no_valid_timestamp"
            include = False
            case_min = case_max = None
        else:
            case_min = min(timestamps); case_max = max(timestamps)
            has_extreme = any(t.year < VALID_MIN_YEAR or t.year > VALID_MAX_YEAR for t in timestamps)
            if has_extreme:
                reason = "extreme_timestamp_before2017_or_after2019"
                include = False
            elif case_min.year < PRIMARY_START_YEAR:
                reason = "left_censored_case_started_before2018"
                include = False
            elif case_min.year > PRIMARY_START_YEAR:
                reason = "case_started_after2018"
                include = False
            else:
                reason = "included_2018_start_cohort"
                include = True

        if not temporal_order_ok:
            order_violation_cases += 1

        if not include:
            excluded_cases += 1; excluded_events += total_case_events
            exclusion_reasons[reason] += 1
            exclusion_rows.append({
                "case_id": case_id,
                "reason": reason,
                "case_min_timestamp": case_min.isoformat() if case_min else "",
                "case_max_timestamp": case_max.isoformat() if case_max else "",
                "total_events": total_case_events,
            })
        else:
            kept_cases += 1; kept_events += total_case_events
            start_month = case_min.strftime("%Y-%m")
            start_month_counts[start_month] += 1
            acts = [ev["activity"] for ev in raw_events]
            resources = [ev["resource"] for ev in raw_events]
            kinds = [ev["resource_kind"] for ev in raw_events]
            timestamps_in_order = [ev["timestamp"] for ev in raw_events if isinstance(ev["timestamp"], datetime)]
            cycle_hours = (case_max - case_min).total_seconds()/3600.0
            variant = " -> ".join(acts)
            case_data = {k: clean_scalar(val(cattrs, k)) for k in CASE_KEYS}
            case_rows.append({
                "case_id": case_id,
                "case_start": case_min,
                "case_end": case_max,
                "case_start_month": start_month,
                "case_end_year": case_max.year,
                "has_2019_followup": bool(case_max.year == 2019),
                "cycle_time_hours": cycle_hours,
                "events_per_case": total_case_events,
                "unique_activities": len(set(acts)),
                "unique_recorded_resources": len({r for r in resources if resource_kind(r) != "no_recorded_user"}),
                "human_user_events": sum(k == "human_user" for k in kinds),
                "batch_user_events": sum(k == "batch_user" for k in kinds),
                "no_recorded_user_events": sum(k == "no_recorded_user" for k in kinds),
                "start_activity": acts[0] if acts else "",
                "end_activity": acts[-1] if acts else "",
                "variant": variant,
                "temporal_order_ok": temporal_order_ok,
                **{f"case_{k}": v for k,v in case_data.items()},
            })

            prev_event = None
            for idx, ev in enumerate(raw_events):
                ts = ev["timestamp"]
                gap_hours = None
                transition = None
                prev_activity = None
                if prev_event is not None and isinstance(ts, datetime) and isinstance(prev_event["timestamp"], datetime):
                    gap_hours = (ts - prev_event["timestamp"]).total_seconds()/3600.0
                    prev_activity = prev_event["activity"]
                    transition = f"{prev_activity} -> {ev['activity']}"
                resource_counts[(ev["resource_kind"], ev["resource"])] += 1
                row = {
                    "case_id": case_id,
                    "event_index": idx,
                    "timestamp": ts,
                    "event_year": ts.year if isinstance(ts, datetime) else None,
                    "event_month": ts.strftime("%Y-%m") if isinstance(ts, datetime) else None,
                    "case_start_month": start_month,
                    "activity": ev["activity"],
                    "resource": ev["resource"],
                    "resource_kind": ev["resource_kind"],
                    "cumulative_net_worth_eur": ev["cumulative_net_worth_eur"],
                    "previous_activity": prev_activity,
                    "transition": transition,
                    "inter_event_gap_hours": gap_hours,
                    "is_first_event": idx == 0,
                    "is_last_event": idx == total_case_events-1,
                }
                event_rows.append(row)
                if len(event_rows) >= EVENT_BATCH_SIZE:
                    writer = flush_events(writer, event_rows, event_path)
                prev_event = ev

        if trace_count % 10000 == 0:
            print(f"  traces={trace_count:,} kept={kept_cases:,} kept_events={kept_events:,}")
        trace.clear()
        parent = trace.getparent()
        if parent is not None:
            while trace.getprevious() is not None:
                del parent[0]

    writer = flush_events(writer, event_rows, event_path)
    if writer is not None:
        writer.close()
    pd.DataFrame(case_rows).to_parquet(case_path, index=False, compression="zstd")

    write_rows_csv(OUT/"cohort_exclusions.csv", exclusion_rows,
                   ["case_id","reason","case_min_timestamp","case_max_timestamp","total_events"])
    write_rows_csv(OUT/"case_start_month_counts.csv",
                   [{"case_start_month":k,"case_count":v} for k,v in sorted(start_month_counts.items())],
                   ["case_start_month","case_count"])
    write_rows_csv(OUT/"resource_event_counts_primary_cohort.csv",
                   [{"resource_kind":k[0],"resource":k[1],"event_count":v} for k,v in resource_counts.most_common()],
                   ["resource_kind","resource","event_count"])

    summary = {
        "source_dataset": str(dataset),
        "source_trace_count": trace_count,
        "source_event_count": event_count,
        "primary_cohort_rule": "first recorded event in 2018; no timestamp <2017 or >2019; retain all events of included cases including 2019 follow-up",
        "kept_cases": kept_cases,
        "kept_events": kept_events,
        "excluded_cases": excluded_cases,
        "excluded_events": excluded_events,
        "kept_case_pct": round(100*kept_cases/trace_count,6) if trace_count else None,
        "kept_event_pct": round(100*kept_events/event_count,6) if event_count else None,
        "exclusion_reason_counts": dict(exclusion_reasons),
        "cases_with_temporal_order_violation": order_violation_cases,
        "expected_from_phase1b": {"cases": EXPECTED_CASES, "events": EXPECTED_EVENTS},
        "matches_phase1b_expected_cases": kept_cases == EXPECTED_CASES,
        "matches_phase1b_expected_events": kept_events == EXPECTED_EVENTS,
        "event_parquet": str(event_path),
        "case_parquet": str(case_path),
    }
    (OUT/"cohort_build_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    report = [
        "# Phase 2A — Primary Cohort Build", "",
        f"- Source cases: **{trace_count:,}**",
        f"- Source events: **{event_count:,}**",
        f"- Retained cases: **{kept_cases:,} ({summary['kept_case_pct']:.3f}%)**",
        f"- Retained events: **{kept_events:,} ({summary['kept_event_pct']:.3f}%)**",
        f"- Excluded cases: **{excluded_cases:,}**",
        f"- Excluded events: **{excluded_events:,}**",
        f"- Cases with temporal-order violations: **{order_violation_cases:,}**", "",
        "## Cohort rule", "",
        summary["primary_cohort_rule"], "",
        "## Important timing semantics", "",
        "The dataset has no lifecycle start/complete pairs. `inter_event_gap_hours` is elapsed time between consecutive recorded events, not activity service time and not pure queue waiting time.", "",
        "## Phase 1B cross-check", "",
        f"Expected cases: {EXPECTED_CASES:,}; observed: {kept_cases:,}; match: **{summary['matches_phase1b_expected_cases']}**",
        f"Expected events: {EXPECTED_EVENTS:,}; observed: {kept_events:,}; match: **{summary['matches_phase1b_expected_events']}**",
    ]
    (OUT/"PHASE2A_COHORT_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print("PHASE 2A COMPLETE")
    print(f"Processed data: {PROCESSED}")

if __name__ == "__main__":
    main()
