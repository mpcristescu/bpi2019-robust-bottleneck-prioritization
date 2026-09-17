from __future__ import annotations
import csv, json, sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from lxml import etree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.xes_utils import attr_dict, find_dataset, local_name

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT = PROJECT_ROOT / "04_ANALYSIS_RESULTS" / "02_PHASE1B_TEMPORAL_AUDIT"
OUT.mkdir(parents=True, exist_ok=True)

CORE_YEAR = 2018
EXTREME_LOW = 2017
EXTREME_HIGH = 2019
MAX_DETAIL_ROWS = 100_000


def write_counter(path, counter, headers):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(headers)
        for k,v in counter.most_common():
            if isinstance(k, tuple): w.writerow([*k,v])
            else: w.writerow([k,v])


def main():
    dataset = find_dataset(REPO)
    print(f"Dataset: {dataset}")
    print(f"Size: {dataset.stat().st_size/1024**2:.1f} MB")
    print("Phase 1B temporal/resource audit started...")

    event_count = trace_count = 0
    year_counts = Counter(); month_counts = Counter(); activity_year = Counter()
    resource_counts = Counter(); resource_kind_counts = Counter()
    extreme_year_counts = Counter(); extreme_activity_counts = Counter(); extreme_resource_counts = Counter()
    boundary_year_counts = Counter()
    case_anomaly_rows = []
    detail_rows = []
    detail_truncated = False
    global_min = global_max = None
    core_min = core_max = None

    context = etree.iterparse(str(dataset), events=("end",), tag="{*}trace", huge_tree=True, recover=True)
    for _, trace in context:
        trace_count += 1
        cattrs = attr_dict(trace)
        case_id = cattrs.get("concept:name", (None, None, None))[1]
        if case_id is None: case_id = f"__missing_case_{trace_count}"
        times=[]; anomaly_events=0; extreme_events=0; boundary_events=0
        case_min=case_max=None
        for child in trace:
            if local_name(child.tag) != "event": continue
            event_count += 1
            e = attr_dict(child)
            act = str(e.get("concept:name", (None,"__MISSING_ACTIVITY__",None))[1])
            ts = e.get("time:timestamp", (None,None,None))[1]
            raw_ts = e.get("time:timestamp", (None,None,None))[2]
            resv = e.get("org:resource", (None,None,None))[1]
            res = "" if resv is None else str(resv)
            resource_counts[res if res else "__EMPTY__"] += 1
            if res.startswith("user_"): kind="human_user"
            elif res.startswith("batch_"): kind="batch_user"
            elif res.strip().upper() in {"", "NONE", "NULL", "N/A", "NA"}: kind="no_recorded_user"
            else: kind="other_resource_value"
            resource_kind_counts[kind] += 1

            if not isinstance(ts, datetime):
                continue
            y=ts.year; ym=f"{y:04d}-{ts.month:02d}"
            year_counts[y]+=1; month_counts[ym]+=1; activity_year[(act,y)]+=1
            if global_min is None or ts<global_min: global_min=ts
            if global_max is None or ts>global_max: global_max=ts
            if y==CORE_YEAR:
                if core_min is None or ts<core_min: core_min=ts
                if core_max is None or ts>core_max: core_max=ts
            if case_min is None or ts<case_min: case_min=ts
            if case_max is None or ts>case_max: case_max=ts
            if y != CORE_YEAR:
                anomaly_events += 1
                category = "boundary" if y in {2017,2019} else "extreme"
                if category=="boundary":
                    boundary_events += 1; boundary_year_counts[y]+=1
                else:
                    extreme_events += 1; extreme_year_counts[y]+=1
                    extreme_activity_counts[act]+=1
                    extreme_resource_counts[res if res else "__EMPTY__"]+=1
                if len(detail_rows) < MAX_DETAIL_ROWS:
                    detail_rows.append({
                        "case_id": case_id, "activity": act, "timestamp_raw": raw_ts or "",
                        "timestamp_iso": ts.isoformat(), "year": y, "resource": res,
                        "category": category,
                        "case_item_category": str(cattrs.get("Item Category",(None,"",None))[1] or ""),
                        "case_company": str(cattrs.get("Company",(None,"",None))[1] or ""),
                        "case_document_type": str(cattrs.get("Document Type",(None,"",None))[1] or ""),
                    })
                else:
                    detail_truncated=True
        if anomaly_events:
            case_anomaly_rows.append({
                "case_id":case_id,
                "case_min_timestamp":case_min.isoformat() if case_min else "",
                "case_max_timestamp":case_max.isoformat() if case_max else "",
                "non_2018_events":anomaly_events,
                "boundary_2017_2019_events":boundary_events,
                "extreme_events_before2017_after2019":extreme_events,
                "total_events":sum(1 for ch in trace if local_name(ch.tag)=="event")
            })
        if trace_count % 10000 == 0:
            print(f"  traces={trace_count:,} events={event_count:,}")
        trace.clear()
        parent=trace.getparent()
        if parent is not None:
            while trace.getprevious() is not None:
                del parent[0]

    write_counter(OUT/"year_counts.csv", year_counts, ["year","event_count"])
    write_counter(OUT/"month_counts.csv", month_counts, ["year_month","event_count"])
    write_counter(OUT/"activity_year_counts.csv", activity_year, ["activity","year","event_count"])
    write_counter(OUT/"resource_counts_corrected.csv", resource_counts, ["resource","event_count"])
    write_counter(OUT/"resource_kind_counts.csv", resource_kind_counts, ["resource_kind","event_count"])
    write_counter(OUT/"extreme_year_counts.csv", extreme_year_counts, ["year","event_count"])
    write_counter(OUT/"extreme_activity_counts.csv", extreme_activity_counts, ["activity","event_count"])
    write_counter(OUT/"extreme_resource_counts.csv", extreme_resource_counts, ["resource","event_count"])

    def write_rows(path, rows, fields):
        with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    write_rows(OUT/"non_2018_event_details.csv", detail_rows,
               ["case_id","activity","timestamp_raw","timestamp_iso","year","resource","category","case_item_category","case_company","case_document_type"])
    write_rows(OUT/"cases_with_non_2018_timestamps.csv", case_anomaly_rows,
               ["case_id","case_min_timestamp","case_max_timestamp","non_2018_events","boundary_2017_2019_events","extreme_events_before2017_after2019","total_events"])

    distinct_all=len(resource_counts)
    distinct_human=sum(1 for r in resource_counts if r.startswith("user_"))
    distinct_batch=sum(1 for r in resource_counts if r.startswith("batch_"))
    no_user_events=resource_kind_counts["no_recorded_user"]
    semantic_recorded=event_count-no_user_events
    summary={
      "dataset_path":str(dataset), "trace_count":trace_count, "event_count":event_count,
      "global_min_timestamp":global_min.isoformat() if global_min else None,
      "global_max_timestamp":global_max.isoformat() if global_max else None,
      "core_2018_min_timestamp":core_min.isoformat() if core_min else None,
      "core_2018_max_timestamp":core_max.isoformat() if core_max else None,
      "events_in_2018":year_counts[2018],
      "events_outside_2018":event_count-year_counts[2018],
      "cases_with_any_non_2018_timestamp":len(case_anomaly_rows),
      "events_in_boundary_years_2017_2019":sum(boundary_year_counts.values()),
      "events_in_extreme_years_before2017_after2019":sum(extreme_year_counts.values()),
      "non_2018_detail_rows_written":len(detail_rows), "detail_truncated":detail_truncated,
      "distinct_resource_values_including_placeholders":distinct_all,
      "distinct_human_users":distinct_human, "distinct_batch_users":distinct_batch,
      "no_recorded_user_events":no_user_events,
      "semantic_resource_coverage_pct_excluding_NONE":round(100*semantic_recorded/event_count,4) if event_count else 0,
      "resource_kind_event_counts":dict(resource_kind_counts),
      "year_counts":dict(sorted(year_counts.items())),
    }
    (OUT/"phase1b_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")

    lines=[
      "# BPI Challenge 2019: Phase 1B Temporal and Resource Audit", "",
      f"- Cases: **{trace_count:,}**", f"- Events: **{event_count:,}**",
      f"- Raw timestamp range: **{summary['global_min_timestamp']} → {summary['global_max_timestamp']}**",
      f"- Events in 2018: **{summary['events_in_2018']:,}**",
      f"- Events outside 2018: **{summary['events_outside_2018']:,}**",
      f"- Cases with ≥1 non-2018 timestamp: **{summary['cases_with_any_non_2018_timestamp']:,}**",
      f"- Extreme events (<2017 or >2019): **{summary['events_in_extreme_years_before2017_after2019']:,}**", "",
      "## Resource semantics", "",
      f"- Distinct human users (`user_`): **{distinct_human}**",
      f"- Distinct batch users (`batch_`): **{distinct_batch}**",
      f"- Events with no recorded user (`NONE`/empty/etc.): **{no_user_events:,}**",
      f"- Semantic resource coverage excluding `NONE`: **{summary['semantic_resource_coverage_pct_excluding_NONE']:.2f}%**", "",
      "## Important interpretation", "",
      "The official BPI Challenge 2019 metadata reports time coverage **2018** and 627 users (607 human + 20 batch).",
      "Therefore, non-2018 timestamps are not silently discarded. This audit quantifies them first so the final temporal analysis can use an explicit, reproducible inclusion/exclusion rule.",
      "Likewise, the XES resource attribute is syntactically present on all events, but `NONE` means no user was recorded; the corrected semantic coverage is reported above.",
    ]
    (OUT/"PHASE1B_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    print("PHASE 1B COMPLETE")

if __name__=="__main__": main()
