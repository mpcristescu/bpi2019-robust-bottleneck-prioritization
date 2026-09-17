
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from src.find_data import find_processed_pair

PROJECT_ROOT=Path(__file__).resolve().parents[3]
OUT=PROJECT_ROOT/"04_ANALYSIS_RESULTS"/"04_PHASE2B_MATURITY_CENSORING_AUDIT"; OUT.mkdir(parents=True, exist_ok=True)
HORIZONS_DAYS=[60,90,120,150,180]
MIN_TRANS_PER_PERIOD=500
CAP_HOURS=120*24
RNG=np.random.default_rng(20260903)

def q(s):
    s=pd.to_numeric(s,errors="coerce").dropna()
    if not len(s): return {}
    qq=s.quantile([.5,.75,.9,.95,.99])
    return dict(count=int(len(s)),mean=float(s.mean()),median=float(qq.loc[.5]),
                p75=float(qq.loc[.75]),p90=float(qq.loc[.9]),p95=float(qq.loc[.95]),
                p99=float(qq.loc[.99]),max=float(s.max()))

def sample(s,n=20000):
    x=pd.to_numeric(s,errors="coerce").dropna().to_numpy(float)
    if len(x)>n: x=RNG.choice(x,n,replace=False)
    return x

def main():
    events_path,cases_path=find_processed_pair(REPO)
    print("Events:",events_path)
    print("Cases :",cases_path)
    cases=pd.read_parquet(cases_path)
    events=pd.read_parquet(events_path, columns=[
        "case_id","timestamp","activity","transition","inter_event_gap_hours","case_start_month"
    ])
    events["timestamp"]=pd.to_datetime(events["timestamp"],utc=True,errors="coerce")
    cases["case_start"]=pd.to_datetime(cases["case_start"],utc=True,errors="coerce")
    cases["case_end"]=pd.to_datetime(cases["case_end"],utc=True,errors="coerce")

    # Empirical monthly density and operational cutoff.
    em=events.dropna(subset=["timestamp"]).copy()
    em["month"]=em["timestamp"].dt.to_period("M").astype(str)
    month_counts=em["month"].value_counts().sort_index().rename_axis("month").reset_index(name="events")
    peak=int(month_counts.events.max())
    dense_threshold=max(1000,int(round(peak*0.01)))
    dense=month_counts[month_counts.events>=dense_threshold]
    if dense.empty: raise RuntimeError("Nu s-a putut estima operational cutoff.")
    last_dense_month=pd.Period(dense.iloc[-1]["month"],freq="M")
    in_month=em[em["timestamp"].dt.to_period("M")==last_dense_month]
    cutoff=in_month["timestamp"].max()
    month_counts["dense_month"]=month_counts.events>=dense_threshold
    month_counts.to_csv(OUT/"01_month_event_density_and_cutoff.csv",index=False,encoding="utf-8-sig")

    # How much sparse data remains after the operational cutoff?
    post=em[em.timestamp>cutoff]
    post_summary={
        "operational_cutoff":cutoff.isoformat(),
        "dense_month_threshold_events":dense_threshold,
        "last_dense_month":str(last_dense_month),
        "events_after_operational_cutoff":int(len(post)),
        "max_timestamp_raw_primary_cohort":em.timestamp.max().isoformat()
    }

    # Case maturity audit
    cases["available_followup_hours"]=(cutoff-cases.case_start).dt.total_seconds()/3600
    cases["available_followup_days"]=cases.available_followup_hours/24
    cases["end_after_cutoff"]=cases.case_end>cutoff
    cases["case_start_month2"]=cases.case_start.dt.to_period("M").astype(str)
    monthly=cases.groupby("case_start_month2").agg(
        cases=("case_id","size"),
        median_cycle_hours=("cycle_time_hours","median"),
        p90_cycle_hours=("cycle_time_hours",lambda x:x.quantile(.90)),
        p95_cycle_hours=("cycle_time_hours",lambda x:x.quantile(.95)),
        median_available_followup_days=("available_followup_days","median"),
        end_after_cutoff_share=("end_after_cutoff","mean"),
    ).reset_index()
    monthly["p95_to_followup_ratio"]=monthly.p95_cycle_hours/(monthly.median_available_followup_days*24)
    monthly["end_after_cutoff_share_pct"]=100*monthly.pop("end_after_cutoff_share")
    monthly.to_csv(OUT/"02_case_maturity_by_start_month.csv",index=False,encoding="utf-8-sig")

    maturity_rows=[]
    for d in HORIZONS_DAYS:
        g=cases[cases.available_followup_days>=d]
        st=q(g.cycle_time_hours)
        maturity_rows.append({
            "min_followup_days":d,"case_count":len(g),
            "case_share_pct":100*len(g)/len(cases),
            **st
        })
    pd.DataFrame(maturity_rows).to_csv(OUT/"03_case_cycle_summary_by_maturity_threshold.csv",index=False,encoding="utf-8-sig")

    # End-activity mix by start quarter: flags potential trace truncation / changing endpoint mix.
    cases["start_quarter"]=cases.case_start.dt.to_period("Q").astype(str)
    endmix=(cases.groupby(["start_quarter","end_activity"]).size()
            .rename("cases").reset_index())
    endmix["quarter_total"]=endmix.groupby("start_quarter").cases.transform("sum")
    endmix["share_pct"]=100*endmix.cases/endmix.quarter_total
    endmix.sort_values(["start_quarter","cases"],ascending=[True,False]).to_csv(
        OUT/"04_end_activity_mix_by_start_quarter.csv",index=False,encoding="utf-8-sig")

    # Transition-origin timestamps. The event row is destination; infer predecessor/origin timestamp.
    trans=events[events.transition.notna()].copy()
    trans["gap_hours"]=pd.to_numeric(trans.inter_event_gap_hours,errors="coerce")
    trans=trans[(trans.gap_hours>=0)&trans.timestamp.notna()]
    trans["origin_timestamp"]=trans.timestamp-pd.to_timedelta(trans.gap_hours,unit="h")
    trans["origin_quarter"]=trans.origin_timestamp.dt.to_period("Q").astype(str)
    trans["origin_month"]=trans.origin_timestamp.dt.to_period("M").astype(str)
    trans["gap_capped_120d_hours"]=trans.gap_hours.clip(upper=CAP_HOURS)

    # Cutoff-safe design: origins through 2018-09-30 have >= about 120 days to dense cutoff.
    safe_origin_end=pd.Timestamp("2018-09-30 23:59:59",tz="UTC")
    safe=trans[trans.origin_timestamp<=safe_origin_end].copy()
    safe["analysis_period"]=np.select(
        [
            safe.origin_timestamp < pd.Timestamp("2018-04-01",tz="UTC"),
            safe.origin_timestamp < pd.Timestamp("2018-07-01",tz="UTC"),
            safe.origin_timestamp < pd.Timestamp("2018-10-01",tz="UTC")
        ],
        ["Q1_2018","Q2_2018","Q3_2018"],
        default="OTHER"
    )
    safe=safe[safe.analysis_period!="OTHER"]

    # Transition drift on fixed 120-day capped delay, Q1+Q2 nominal vs Q3 holdout.
    safe["nominal_holdout"]=np.where(safe.analysis_period.isin(["Q1_2018","Q2_2018"]),"H1_nominal","Q3_holdout")
    rows=[]
    for tr,g in safe.groupby("transition",observed=True):
        a=g[g.nominal_holdout=="H1_nominal"].gap_capped_120d_hours
        b=g[g.nominal_holdout=="Q3_holdout"].gap_capped_120d_hours
        if len(a)<MIN_TRANS_PER_PERIOD or len(b)<MIN_TRANS_PER_PERIOD: continue
        xa=sample(a); xb=sample(b)
        rows.append({
            "transition":tr,
            "h1_count":len(a),"q3_count":len(b),
            "h1_mean_capped_hours":float(a.mean()),"q3_mean_capped_hours":float(b.mean()),
            "h1_median_capped_hours":float(a.median()),"q3_median_capped_hours":float(b.median()),
            "h1_p95_capped_hours":float(a.quantile(.95)),"q3_p95_capped_hours":float(b.quantile(.95)),
            "median_shift_hours":float(b.median()-a.median()),
            "wasserstein_capped_hours":float(wasserstein_distance(xa,xb)),
            "h1_total_capped_gap_hours":float(a.sum()),
            "q3_total_capped_gap_hours":float(b.sum())
        })
    drift=pd.DataFrame(rows)
    if len(drift):
        drift["combined_capped_gap_hours"]=drift.h1_total_capped_gap_hours+drift.q3_total_capped_gap_hours
        drift["q3_to_h1_mean_ratio"]=drift.q3_mean_capped_hours/drift.h1_mean_capped_hours.replace(0,np.nan)
        drift=drift.sort_values(["combined_capped_gap_hours","wasserstein_capped_hours"],ascending=False)
    drift.to_csv(OUT/"05_cutoff_safe_transition_drift_H1_vs_Q3.csv",index=False,encoding="utf-8-sig")

    # Candidate set for future robust bottleneck prioritization.
    if len(drift):
        total=drift.combined_capped_gap_hours.sum()
        cand=drift.copy()
        cand["share_of_candidate_capped_gap_pct"]=100*cand.combined_capped_gap_hours/total
        cand["drift_weighted_delay_score"]=cand.share_of_candidate_capped_gap_pct*(1+np.log1p(cand.wasserstein_capped_hours))
        cand=cand.sort_values("drift_weighted_delay_score",ascending=False)
        cand.head(40).to_csv(OUT/"06_phase3_candidate_transitions.csv",index=False,encoding="utf-8-sig")
    else:
        cand=pd.DataFrame()

    # Decision report
    dec_month=monthly.loc[monthly.case_start_month2=="2018-12"]
    dec_med=float(dec_month.median_cycle_hours.iloc[0]) if len(dec_month) else None
    jan_month=monthly.loc[monthly.case_start_month2=="2018-01"]
    jan_med=float(jan_month.median_cycle_hours.iloc[0]) if len(jan_month) else None
    report=f"""# Phase 2B — Maturity and Right-Censoring Audit

## Operational observation edge
- Last dense month: **{last_dense_month}**
- Operational cutoff: **{cutoff.isoformat()}**
- Events after cutoff: **{len(post):,}** (sparse residual/anomalous tail)

## Why this audit was necessary
The Phase 2 case-start-month profile showed a strong decline in observed case span late in 2018.
January median observed span: **{jan_med:.2f} h**.
December median observed span: **{dec_med:.2f} h**.
This pattern is consistent with right-edge observation bias and must not be interpreted as a process improvement without additional evidence.

## Recommended temporal design for the next model
Use transition-origin observations from **2018-01-01 through 2018-09-30**.
Define:
- nominal period: **Q1 + Q2 2018**;
- out-of-time holdout: **Q3 2018**;
- transition-delay outcome: elapsed time between consecutive recorded events, **capped at 120 days**.

This keeps at least ~120 days of observation after every included transition origin and removes Q4 from the temporal-shift experiment.

## Interpretation rule
The outcome is a recorded inter-event elapsed time, not service time and not pure queue waiting time.

## Phase 3 gate
If the cutoff-safe H1-vs-Q3 candidate table is sufficiently populated, proceed to
**distributionally robust bottleneck prioritization under temporal shift**.
Do not use whole-case cycle time from Q4 as an optimization target.
"""
    (OUT/"PHASE2B_MATURITY_CENSORING_REPORT.md").write_text(report,encoding="utf-8")

    summary={
        **post_summary,
        "cases":int(len(cases)),
        "events":int(len(events)),
        "recommended_safe_origin_end":"2018-09-30T23:59:59Z",
        "nominal_period":"Q1+Q2 2018",
        "holdout_period":"Q3 2018",
        "fixed_delay_cap_days":120,
        "candidate_transition_count":int(len(drift)),
        "proposed_phase3":"distributionally robust bottleneck prioritization under temporal shift"
    }
    (OUT/"phase2b_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(report)
    print("PHASE 2B COMPLETE")

if __name__=="__main__":
    main()
