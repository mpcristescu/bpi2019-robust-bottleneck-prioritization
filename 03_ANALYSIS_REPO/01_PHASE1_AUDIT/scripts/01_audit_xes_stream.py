from __future__ import annotations

import csv
import json
import math
import random
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median

from lxml import etree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.xes_utils import attr_dict, dt_to_iso, find_dataset, local_name

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT = PROJECT_ROOT / "04_ANALYSIS_RESULTS" / "01_PHASE1_AUDIT"
OUT.mkdir(parents=True, exist_ok=True)
ERR = OUT / "phase1_error.log"

RNG = random.Random(20260903)
GAP_RESERVOIR_MAX = 200_000
SAMPLE_CASES_MAX = 250
TOP_N = 5000


def pct(n, d):
    return (100.0 * n / d) if d else 0.0


def q(sorted_vals, p):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    x = (len(sorted_vals) - 1) * p
    lo = int(math.floor(x)); hi = int(math.ceil(x))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] * (hi - x) + sorted_vals[hi] * (x - lo)


def write_counter(path: Path, counter: Counter, a: str, b: str, limit=None):
    rows = counter.most_common(limit)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([a, b])
        w.writerows(rows)


def main():
    dataset = find_dataset(REPO)
    print(f"Dataset: {dataset}")
    print(f"Size: {dataset.stat().st_size / 1024**2:.1f} MB")
    print("Streaming XES audit started...")

    trace_count = 0
    event_count = 0
    cases_missing_id = 0
    events_missing_activity = 0
    events_missing_time = 0
    events_missing_resource = 0
    traces_with_time_order_violation = 0

    case_key_present = Counter()
    case_key_types = defaultdict(Counter)
    event_key_present = Counter()
    event_key_types = defaultdict(Counter)
    sample_values_case = defaultdict(list)
    sample_values_event = defaultdict(list)

    activity_counts = Counter()
    resource_counts = Counter()
    lifecycle_counts = Counter()
    start_activity_counts = Counter()
    end_activity_counts = Counter()
    dfg_counts = Counter()
    variant_counts = Counter()

    case_sizes = []
    case_durations_hours = []
    gap_reservoir = []
    seen_gaps = 0

    global_min_time = None
    global_max_time = None
    sample_rows = []

    context = etree.iterparse(
        str(dataset), events=("end",), tag="{*}trace", huge_tree=True, recover=True
    )

    for _, trace in context:
        trace_count += 1
        cattrs = attr_dict(trace)
        case_id = cattrs.get("concept:name", (None, None))[1]
        if case_id is None:
            cases_missing_id += 1
            case_id = f"__missing_case_{trace_count}"

        for key, (typ, val) in cattrs.items():
            case_key_present[key] += 1
            case_key_types[key][typ] += 1
            if len(sample_values_case[key]) < 5 and val is not None:
                s = dt_to_iso(val)
                if s not in sample_values_case[key]:
                    sample_values_case[key].append(s[:200])

        activities = []
        times = []
        trace_event_rows = []

        for child in trace:
            if local_name(child.tag) != "event":
                continue
            event_count += 1
            eattrs = attr_dict(child)
            for key, (typ, val) in eattrs.items():
                event_key_present[key] += 1
                event_key_types[key][typ] += 1
                if len(sample_values_event[key]) < 5 and val is not None:
                    s = dt_to_iso(val)
                    if s not in sample_values_event[key]:
                        sample_values_event[key].append(s[:200])

            act = eattrs.get("concept:name", (None, None))[1]
            ts = eattrs.get("time:timestamp", (None, None))[1]
            res = eattrs.get("org:resource", (None, None))[1]
            lc = eattrs.get("lifecycle:transition", (None, None))[1]

            if act is None:
                events_missing_activity += 1
                act = "__MISSING_ACTIVITY__"
            activity_counts[str(act)] += 1
            activities.append(str(act))

            if isinstance(ts, datetime):
                times.append(ts)
                if global_min_time is None or ts < global_min_time:
                    global_min_time = ts
                if global_max_time is None or ts > global_max_time:
                    global_max_time = ts
            else:
                events_missing_time += 1
                times.append(None)

            if res is None or str(res).strip() == "":
                events_missing_resource += 1
            else:
                resource_counts[str(res)] += 1

            if lc is not None:
                lifecycle_counts[str(lc)] += 1

            if trace_count <= SAMPLE_CASES_MAX:
                row = {
                    "case_id": case_id,
                    "activity": act,
                    "timestamp": dt_to_iso(ts),
                    "resource": "" if res is None else str(res),
                    "lifecycle": "" if lc is None else str(lc),
                }
                for k, (_, v) in cattrs.items():
                    row[f"case:{k}"] = dt_to_iso(v)
                for k, (_, v) in eattrs.items():
                    if k not in {"concept:name", "time:timestamp", "org:resource", "lifecycle:transition"}:
                        row[f"event:{k}"] = dt_to_iso(v)
                trace_event_rows.append(row)

        n = len(activities)
        case_sizes.append(n)
        if activities:
            start_activity_counts[activities[0]] += 1
            end_activity_counts[activities[-1]] += 1
            variant_counts[tuple(activities)] += 1
            for a, b in zip(activities[:-1], activities[1:]):
                dfg_counts[(a, b)] += 1

        valid_times = [t for t in times if isinstance(t, datetime)]
        if valid_times:
            tmin, tmax = min(valid_times), max(valid_times)
            case_durations_hours.append((tmax - tmin).total_seconds() / 3600.0)

            ordered = True
            prev = None
            for t in times:
                if not isinstance(t, datetime):
                    continue
                if prev is not None:
                    gap = (t - prev).total_seconds() / 3600.0
                    if gap < 0:
                        ordered = False
                    seen_gaps += 1
                    if len(gap_reservoir) < GAP_RESERVOIR_MAX:
                        gap_reservoir.append(gap)
                    else:
                        j = RNG.randint(1, seen_gaps)
                        if j <= GAP_RESERVOIR_MAX:
                            gap_reservoir[j - 1] = gap
                prev = t
            if not ordered:
                traces_with_time_order_violation += 1

        if trace_count <= SAMPLE_CASES_MAX:
            sample_rows.extend(trace_event_rows)

        if trace_count % 10_000 == 0:
            print(f"  traces={trace_count:,} events={event_count:,}")

        trace.clear()
        parent = trace.getparent()
        if parent is not None:
            while trace.getprevious() is not None:
                del parent[0]

    case_sizes_sorted = sorted(case_sizes)
    dur_sorted = sorted(case_durations_hours)
    gap_sorted = sorted(gap_reservoir)

    summary = {
        "dataset_path": str(dataset),
        "dataset_size_bytes": dataset.stat().st_size,
        "trace_count": trace_count,
        "event_count": event_count,
        "cases_missing_id": cases_missing_id,
        "events_missing_activity": events_missing_activity,
        "events_missing_time": events_missing_time,
        "events_missing_resource": events_missing_resource,
        "resource_coverage_pct": round(100.0 - pct(events_missing_resource, event_count), 4),
        "traces_with_time_order_violation": traces_with_time_order_violation,
        "distinct_activities": len(activity_counts),
        "distinct_resources_observed": len(resource_counts),
        "distinct_variants": len(variant_counts),
        "distinct_directly_follows_pairs": len(dfg_counts),
        "global_min_timestamp": dt_to_iso(global_min_time),
        "global_max_timestamp": dt_to_iso(global_max_time),
        "lifecycle_values": dict(lifecycle_counts),
        "has_lifecycle_transition_attribute": bool(lifecycle_counts),
        "has_start_and_complete_lifecycle": (
            any(k.lower() == "start" for k in lifecycle_counts)
            and any(k.lower() == "complete" for k in lifecycle_counts)
        ),
        "case_size": {
            "min": case_sizes_sorted[0] if case_sizes_sorted else None,
            "p25": q(case_sizes_sorted, .25),
            "median": q(case_sizes_sorted, .5),
            "p75": q(case_sizes_sorted, .75),
            "p95": q(case_sizes_sorted, .95),
            "max": case_sizes_sorted[-1] if case_sizes_sorted else None,
            "mean": mean(case_sizes_sorted) if case_sizes_sorted else None,
        },
        "case_duration_hours": {
            "n": len(dur_sorted),
            "p25": q(dur_sorted, .25),
            "median": q(dur_sorted, .5),
            "p75": q(dur_sorted, .75),
            "p95": q(dur_sorted, .95),
            "p99": q(dur_sorted, .99),
            "max": dur_sorted[-1] if dur_sorted else None,
        },
        "inter_event_gap_hours_reservoir": {
            "n_seen": seen_gaps,
            "n_sampled": len(gap_sorted),
            "p25": q(gap_sorted, .25),
            "median": q(gap_sorted, .5),
            "p75": q(gap_sorted, .75),
            "p95": q(gap_sorted, .95),
            "p99": q(gap_sorted, .99),
            "min": gap_sorted[0] if gap_sorted else None,
            "max": gap_sorted[-1] if gap_sorted else None,
        },
        "feasibility_flags": {
            "can_measure_case_cycle_time_from_event_timestamps": bool(dur_sorted),
            "can_measure_inter_event_delay": bool(seen_gaps),
            "can_directly_measure_processing_time_from_lifecycle": (
                any(k.lower() == "start" for k in lifecycle_counts)
                and any(k.lower() == "complete" for k in lifecycle_counts)
            ),
            "has_resource_field_with_nonzero_coverage": bool(resource_counts),
        },
    }

    (OUT / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    def write_schema(path, keys_present, keys_types, samples, denom, kind):
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["scope", "key", "present_count", "coverage_pct", "xes_types", "sample_values"])
            for key, cnt in keys_present.most_common():
                types = ";".join(f"{k}:{v}" for k, v in keys_types[key].most_common())
                vals = " | ".join(samples[key])
                w.writerow([kind, key, cnt, round(pct(cnt, denom), 4), types, vals])

    write_schema(OUT / "schema_case_attributes.csv", case_key_present, case_key_types,
                 sample_values_case, trace_count, "case")
    write_schema(OUT / "schema_event_attributes.csv", event_key_present, event_key_types,
                 sample_values_event, event_count, "event")

    write_counter(OUT / "activity_counts.csv", activity_counts, "activity", "event_count")
    write_counter(OUT / "resource_counts.csv", resource_counts, "resource", "event_count", TOP_N)
    write_counter(OUT / "lifecycle_counts.csv", lifecycle_counts, "lifecycle", "event_count")
    write_counter(OUT / "start_activity_counts.csv", start_activity_counts, "activity", "case_count")
    write_counter(OUT / "end_activity_counts.csv", end_activity_counts, "activity", "case_count")

    with (OUT / "directly_follows_counts.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["activity_from", "activity_to", "count"])
        for (a,b), cnt in dfg_counts.most_common(TOP_N):
            w.writerow([a,b,cnt])

    with (OUT / "variant_counts.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["rank", "case_count", "variant_length", "variant"])
        for rank, (variant, cnt) in enumerate(variant_counts.most_common(TOP_N), 1):
            w.writerow([rank, cnt, len(variant), " -> ".join(variant)])

    with (OUT / "case_size_distribution.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["events_per_case", "case_count"])
        for k, cnt in sorted(Counter(case_sizes).items()):
            w.writerow([k, cnt])

    with (OUT / "case_duration_distribution.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["case_duration_hours"])
        for v in case_durations_hours:
            w.writerow([v])

    with (OUT / "inter_event_gap_sample.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["inter_event_gap_hours"])
        for v in gap_reservoir:
            w.writerow([v])

    if sample_rows:
        fieldnames = sorted({k for r in sample_rows for k in r.keys()},
                            key=lambda x: (0 if x in {"case_id","activity","timestamp","resource","lifecycle"} else 1, x))
        with (OUT / "sample_events.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader(); w.writerows(sample_rows)

    print("Audit complete.")
    print(json.dumps(summary["feasibility_flags"], indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        ERR.write_text(traceback.format_exc(), encoding="utf-8")
        traceback.print_exc()
        sys.exit(1)
