from __future__ import annotations

import csv
import json
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT = PROJECT_ROOT / "04_ANALYSIS_RESULTS" / "01_PHASE1_AUDIT"
ERR = OUT / "phase1_error.log"


def read_top_csv(path: Path, n=10):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))[:n]


def main():
    s = json.loads((OUT / "audit_summary.json").read_text(encoding="utf-8"))
    flags = s["feasibility_flags"]
    lifecycle = s.get("lifecycle_values", {})
    top_acts = read_top_csv(OUT / "activity_counts.csv", 15)
    top_resources = read_top_csv(OUT / "resource_counts.csv", 10)
    top_variants = read_top_csv(OUT / "variant_counts.csv", 10)

    direct_processing = flags["can_directly_measure_processing_time_from_lifecycle"]
    resource_ok = flags["has_resource_field_with_nonzero_coverage"]

    recommendation = []
    if direct_processing:
        recommendation.append(
            "Logul conține atât lifecycle=start cât și lifecycle=complete; putem investiga durate de procesare pe activități, după validarea perechilor start-complete."
        )
    else:
        recommendation.append(
            "Nu există dovadă suficientă start+complete pentru a interpreta timestampurile drept processing/service time. În Faza 2 vom trata doar cycle time și inter-event delay ca observabile directe."
        )
    if resource_ok:
        recommendation.append(
            f"Câmpul de resursă este observat cu acoperire {s['resource_coverage_pct']:.2f}%. Putem analiza workload/resource assignment, dar nu vom presupune capacitate sau ore lucrate fără o reconstrucție explicită."
        )
    else:
        recommendation.append(
            "Nu avem resurse suficient observate; modelul final nu trebuie construit în jurul resource allocation individual."
        )

    md = []
    md.append("# BPI Challenge 2019: Phase 1 Feasibility Report\n")
    md.append("## 1. Dataset audit\n")
    md.append(f"- Traces/cases: **{s['trace_count']:,}**")
    md.append(f"- Events: **{s['event_count']:,}**")
    md.append(f"- Activities: **{s['distinct_activities']:,}**")
    md.append(f"- Observed resources: **{s['distinct_resources_observed']:,}**")
    md.append(f"- Resource coverage: **{s['resource_coverage_pct']:.2f}%**")
    md.append(f"- Variants: **{s['distinct_variants']:,}**")
    md.append(f"- Time range: **{s['global_min_timestamp']} → {s['global_max_timestamp']}**")
    md.append(f"- Cases with timestamp-order violations: **{s['traces_with_time_order_violation']:,}**")
    md.append("")

    md.append("## 2. Lifecycle evidence\n")
    md.append(f"Observed lifecycle values: `{json.dumps(lifecycle, ensure_ascii=False)}`")
    md.append(f"Direct start/complete processing-time measurement supported: **{direct_processing}**")
    md.append("")

    md.append("## 3. Case duration\n")
    d = s["case_duration_hours"]
    md.append(f"- median: {d['median']}")
    md.append(f"- P95: {d['p95']}")
    md.append(f"- P99: {d['p99']}")
    md.append("")

    md.append("## 4. Inter-event delay reservoir sample\n")
    g = s["inter_event_gap_hours_reservoir"]
    md.append(f"- gaps seen: {g['n_seen']:,}")
    md.append(f"- sampled: {g['n_sampled']:,}")
    md.append(f"- median: {g['median']}")
    md.append(f"- P95: {g['p95']}")
    md.append("")

    md.append("## 5. Top activities\n")
    for r in top_acts:
        md.append(f"- {r.get('activity')}: {r.get('event_count')}")
    md.append("")

    md.append("## 6. Top resources\n")
    for r in top_resources:
        md.append(f"- {r.get('resource')}: {r.get('event_count')}")
    md.append("")

    md.append("## 7. Dominant variants\n")
    for r in top_variants:
        md.append(f"- n={r.get('case_count')} | len={r.get('variant_length')} | {r.get('variant')}")
    md.append("")

    md.append("## 8. Automatic methodological recommendation\n")
    for x in recommendation:
        md.append(f"- {x}")
    md.append("")
    md.append("**Important:** acest raport decide doar ce observabile sunt legitime. Formularea finală a modelului Mathematics/DRO se fixează după revizuirea acestor outputs.")

    report = OUT / "PHASE1_FEASIBILITY_REPORT.md"
    report.write_text("\n".join(md), encoding="utf-8")

    print(f"Report written: {report}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        ERR.write_text(traceback.format_exc(), encoding="utf-8")
        traceback.print_exc()
        sys.exit(1)
