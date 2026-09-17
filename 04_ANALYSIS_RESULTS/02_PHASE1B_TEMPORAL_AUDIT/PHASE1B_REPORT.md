# BPI Challenge 2019 : Phase 1B Temporal & Resource Audit

- Cases: **251,734**
- Events: **1,595,923**
- Raw timestamp range: **1948-01-26T22:59:00+00:00 → 2020-04-09T21:59:00+00:00**
- Events in 2018: **1,550,468**
- Events outside 2018: **45,455**
- Cases with ≥1 non-2018 timestamp: **32,920**
- Extreme events (<2017 or >2019): **97**

## Resource semantics

- Distinct human users (`user_`): **607**
- Distinct batch users (`batch_`): **20**
- Events with no recorded user (`NONE`/empty/etc.): **399,090**
- Semantic resource coverage excluding `NONE`: **74.99%**

## Important interpretation

The official BPI Challenge 2019 metadata reports time coverage **2018** and 627 users (607 human + 20 batch).
Therefore, non-2018 timestamps are not silently discarded. This audit quantifies them first so the final temporal analysis can use an explicit, reproducible inclusion/exclusion rule.
Likewise, the XES resource attribute is syntactically present on all events, but `NONE` means no user was recorded; the corrected semantic coverage is reported above.
