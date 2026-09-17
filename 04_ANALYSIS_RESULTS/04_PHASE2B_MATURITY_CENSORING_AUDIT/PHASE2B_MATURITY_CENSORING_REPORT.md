# Phase 2B : Maturity and Right-Censoring Audit

## Operational observation edge
- Last dense month: **2019-01**
- Operational cutoff: **2019-01-18T13:34:00+00:00**
- Events after cutoff: **3** (sparse residual/anomalous tail)

## Why this audit was necessary
The Phase 2 case-start-month profile showed a strong decline in observed case span late in 2018.
January median observed span: **1847.05 h**.
December median observed span: **516.92 h**.
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
