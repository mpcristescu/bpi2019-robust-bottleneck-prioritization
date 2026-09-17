# Phase 2A : Primary Cohort Build

- Source cases: **251,734**
- Source events: **1,595,923**
- Retained cases: **251,266 (99.814%)**
- Retained events: **1,587,374 (99.464%)**
- Excluded cases: **468**
- Excluded events: **8,549**
- Cases with temporal-order violations: **0**

## Cohort rule

first recorded event in 2018; no timestamp <2017 or >2019; retain all events of included cases including 2019 follow-up

## Important timing semantics

The dataset has no lifecycle start/complete pairs. `inter_event_gap_hours` is elapsed time between consecutive recorded events, not activity service time and not pure queue waiting time.

## Phase 1B cross-check

Expected cases: 251,266; observed: 251,266; match: **True**
Expected events: 1,587,374; observed: 1,587,374; match: **True**
