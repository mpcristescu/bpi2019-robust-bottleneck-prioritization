from __future__ import annotations
import json, math
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

REPO = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT = PROJECT_ROOT / "04_ANALYSIS_RESULTS" / "03_PHASE2_PROCESS_STRUCTURE"
PROCESSED = OUT / "processed"
OUT.mkdir(exist_ok=True)

MIN_TRANSITION_COUNT = 200
MIN_DRIFT_GROUP_COUNT = 100
MAX_WASSERSTEIN_SAMPLE = 20000
RNG = np.random.default_rng(20260903)

def qseries(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s)==0:
        return {"count":0,"mean":np.nan,"median":np.nan,"p75":np.nan,"p90":np.nan,"p95":np.nan,"p99":np.nan,"max":np.nan,"sum":0.0}
    q=s.quantile([.5,.75,.9,.95,.99])
    return {
        "count":int(len(s)),"mean":float(s.mean()),"median":float(q.loc[.5]),"p75":float(q.loc[.75]),
        "p90":float(q.loc[.9]),"p95":float(q.loc[.95]),"p99":float(q.loc[.99]),"max":float(s.max()),"sum":float(s.sum())
    }

def summarize_group(df, group_col, value_col, min_count=1):
    rows=[]
    for key,g in df.groupby(group_col, dropna=False, observed=True):
        stats=qseries(g[value_col])
        if stats["count"] < min_count: continue
        rows.append({group_col:key, **stats})
    return pd.DataFrame(rows)

def shannon_entropy(values):
    vc=pd.Series(values).value_counts(normalize=True)
    if len(vc)<=1: return 0.0
    return float(-(vc*np.log(vc)).sum())

def sample_values(s, n=MAX_WASSERSTEIN_SAMPLE):
    x=pd.to_numeric(s,errors="coerce").dropna().to_numpy(dtype=float)
    if len(x)>n:
        x=RNG.choice(x,size=n,replace=False)
    return x

def main():
    events_path=PROCESSED/"events_primary_cohort.parquet"
    cases_path=PROCESSED/"cases_primary_cohort.parquet"
    if not events_path.exists() or not cases_path.exists():
        raise FileNotFoundError("Ruleaza mai intai scripts/01_build_primary_cohort.py")
    print("Loading processed cohort...")
    event_columns=["case_id","timestamp","event_month","case_start_month","activity","resource","resource_kind","transition","inter_event_gap_hours"]
    events=pd.read_parquet(events_path, columns=event_columns)
    cases=pd.read_parquet(cases_path)
    print(f"Cases={len(cases):,} Events={len(events):,}")

    # Core case performance
    cycle_summary=qseries(cases["cycle_time_hours"])
    cycle_month=summarize_group(cases,"case_start_month","cycle_time_hours")
    cycle_month.to_csv(OUT/"cycle_time_by_case_start_month.csv",index=False,encoding="utf-8-sig")

    # Process variants
    vc=cases["variant"].value_counts(dropna=False)
    variant_df=vc.rename_axis("variant").reset_index(name="case_count")
    variant_df["rank"]=np.arange(1,len(variant_df)+1)
    variant_df["case_share_pct"]=100*variant_df.case_count/len(cases)
    variant_df["cumulative_case_share_pct"]=variant_df.case_share_pct.cumsum()
    variant_df["variant_length"]=variant_df["variant"].fillna("").map(lambda x: 0 if x=="" else x.count(" -> ")+1)
    variant_df.head(5000).to_csv(OUT/"variant_summary_top5000.csv",index=False,encoding="utf-8-sig")

    # Directly follows and gap semantics
    trans=events[events["transition"].notna()].copy()
    trans["inter_event_gap_hours"]=pd.to_numeric(trans["inter_event_gap_hours"],errors="coerce")
    trans=trans[trans.inter_event_gap_hours>=0]
    transition_summary=summarize_group(trans,"transition","inter_event_gap_hours",MIN_TRANSITION_COUNT)
    total_gap=transition_summary["sum"].sum() if len(transition_summary) else np.nan
    transition_summary["share_of_summarized_gap_hours_pct"]=100*transition_summary["sum"]/total_gap if total_gap else np.nan
    transition_summary["tail_ratio_p95_to_median"]=transition_summary["p95"]/transition_summary["median"].replace(0,np.nan)
    transition_summary=transition_summary.sort_values(["sum","p95"],ascending=False)
    transition_summary.to_csv(OUT/"transition_delay_summary.csv",index=False,encoding="utf-8-sig")
    transition_summary.head(50).to_csv(OUT/"top50_transition_bottleneck_candidates.csv",index=False,encoding="utf-8-sig")

    activity_gap=summarize_group(trans,"activity","inter_event_gap_hours",MIN_TRANSITION_COUNT)
    activity_gap=activity_gap.sort_values(["sum","p95"],ascending=False)
    activity_gap.to_csv(OUT/"activity_preceding_gap_summary.csv",index=False,encoding="utf-8-sig")

    # Case category heterogeneity
    category_cols=["case_Item Category","case_Document Type","case_Item Type","case_Company"]
    cat_rows=[]
    for c in category_cols:
        if c not in cases.columns: continue
        for key,g in cases.groupby(c,dropna=False,observed=True):
            st=qseries(g["cycle_time_hours"])
            cat_rows.append({"dimension":c.replace("case_",""),"category":key,"case_count":len(g),**st})
    pd.DataFrame(cat_rows).to_csv(OUT/"case_category_cycle_time_summary.csv",index=False,encoding="utf-8-sig")

    # Resource composition; counts are descriptive only
    rk=events["resource_kind"].value_counts().rename_axis("resource_kind").reset_index(name="event_count")
    rk["event_share_pct"]=100*rk.event_count/len(events)
    rk.to_csv(OUT/"resource_kind_summary.csv",index=False,encoding="utf-8-sig")

    human=events[events.resource_kind=="human_user"].copy()
    hr=[]
    for res,g in human.groupby("resource",observed=True):
        hr.append({
            "resource":res,
            "event_count":len(g),
            "distinct_activities":g.activity.nunique(),
            "activity_entropy":shannon_entropy(g.activity),
            "active_months":g.event_month.nunique(),
        })
    pd.DataFrame(hr).sort_values("event_count",ascending=False).to_csv(OUT/"human_resource_activity_profile.csv",index=False,encoding="utf-8-sig")

    # Temporal drift: H1 vs H2 case-start cohorts on transition gaps.
    # This measures change in recorded inter-event elapsed time, not service-time drift.
    trans["start_half"]=np.where(trans.case_start_month.astype(str).str.slice(5,7).astype(int)<=6,"H1_2018","H2_2018")
    drift_rows=[]
    for tr,g in trans.groupby("transition",observed=True):
        a=g[g.start_half=="H1_2018"].inter_event_gap_hours.dropna()
        b=g[g.start_half=="H2_2018"].inter_event_gap_hours.dropna()
        if len(a)<MIN_DRIFT_GROUP_COUNT or len(b)<MIN_DRIFT_GROUP_COUNT:
            continue
        xa=sample_values(a); xb=sample_values(b)
        med_a=float(np.median(xa)); med_b=float(np.median(xb))
        drift_rows.append({
            "transition":tr,"h1_count":len(a),"h2_count":len(b),
            "h1_median_hours":med_a,"h2_median_hours":med_b,
            "median_shift_hours":med_b-med_a,
            "median_ratio_h2_h1":(med_b/med_a if med_a>0 else np.nan),
            "wasserstein_hours":float(wasserstein_distance(xa,xb)),
        })
    drift=pd.DataFrame(drift_rows)
    if len(drift):
        drift["abs_median_shift_hours"]=drift.median_shift_hours.abs()
        drift=drift.sort_values(["wasserstein_hours","abs_median_shift_hours"],ascending=False)
    drift.to_csv(OUT/"transition_temporal_drift_H1_vs_H2.csv",index=False,encoding="utf-8-sig")

    # Monthly volume and follow-up
    volume=cases.groupby("case_start_month",observed=True).agg(
        case_count=("case_id","size"),
        median_cycle_hours=("cycle_time_hours","median"),
        p95_cycle_hours=("cycle_time_hours",lambda x: x.quantile(.95)),
        followup_2019_share=("has_2019_followup","mean"),
    ).reset_index()
    volume["followup_2019_share_pct"]=100*volume.pop("followup_2019_share")
    volume.to_csv(OUT/"case_start_month_performance.csv",index=False,encoding="utf-8-sig")

    summary={
        "cases":int(len(cases)),"events":int(len(events)),
        "distinct_activities":int(events.activity.nunique()),
        "distinct_transitions":int(trans.transition.nunique()),
        "distinct_variants":int(cases.variant.nunique()),
        "cases_with_2019_followup":int(cases.has_2019_followup.sum()),
        "cycle_time_hours":cycle_summary,
        "transition_summary_min_count":MIN_TRANSITION_COUNT,
        "transition_rows_analyzed":int(len(trans)),
        "human_resource_events":int((events.resource_kind=="human_user").sum()),
        "batch_resource_events":int((events.resource_kind=="batch_user").sum()),
        "no_recorded_resource_events":int((events.resource_kind=="no_recorded_user").sum()),
        "timing_semantics":"inter-event gaps are elapsed time between consecutive recorded events; not service time and not pure waiting time",
        "phase3_gate":"Use Phase 2 results to formulate robust process-improvement/bottleneck-intervention optimization. Literal resource-capacity optimization requires additional lifecycle-rich data.",
    }
    (OUT/"phase2_process_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")

    top_variants_80 = int((variant_df.cumulative_case_share_pct < 80).sum()+1)
    lines=[
        "# Phase 2 — Process Structure and Temporal Performance", "",
        f"- Cases: **{len(cases):,}**",
        f"- Events: **{len(events):,}**",
        f"- Activities: **{events.activity.nunique():,}**",
        f"- Observed directly-follow transitions: **{trans.transition.nunique():,}**",
        f"- Distinct process variants: **{cases.variant.nunique():,}**",
        f"- Top variants required for 80% case coverage: **{top_variants_80:,}**",
        f"- Median cycle time: **{cycle_summary['median']:.2f} h**",
        f"- P95 cycle time: **{cycle_summary['p95']:.2f} h**", "",
        "## Critical interpretation", "",
        "The log does not expose lifecycle start/complete pairs. All transition timing results are therefore reported as **elapsed time between consecutive recorded events**. They must not be interpreted as task processing times or pure queue waiting times.", "",
        "## Purpose of this phase", "",
        "This phase identifies process variants, transition-level elapsed-time concentration, tail delays, category heterogeneity, and H1-vs-H2 temporal drift. These are the empirical inputs for the next robust-optimization formulation.", "",
        "## Phase 3 decision", "",
        "The preferred next formulation is **distributionally robust process-improvement / bottleneck-intervention optimization under temporal shift**, not literal staff-capacity optimization from BPI 2019 alone.",
        "If literal resource allocation is retained as the main claim, an additional public dataset with reliable lifecycle start/complete information should be added.",
    ]
    (OUT/"PHASE2_PROCESS_STRUCTURE_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    print("PHASE 2B COMPLETE")

if __name__=="__main__":
    main()
