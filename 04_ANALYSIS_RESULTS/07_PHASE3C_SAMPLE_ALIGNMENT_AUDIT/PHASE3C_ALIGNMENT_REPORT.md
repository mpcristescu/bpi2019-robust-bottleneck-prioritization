# Phase 3C, Exact Sample Alignment Audit

## Decision

The exact Phase 3 H1-origin convention was reproduced for all 63 frozen candidate transitions.
The reproduction gate passed. Phase 3 H1 observation counts and primary metrics match within the declared numeric tolerance.
No new candidate selection, parameter tuning, or out-of-time evaluation was performed.

## Input isolation

Q3 was NOT read or used.
The Parquet predicate read extends beyond 30 June only because a destination timestamp after June can belong to an H1-origin transition under the exact Phase 3 convention. Only rows with an inferred origin in H1 are retained. No Q3-origin row, Q3 result file, or Q3-derived statistic enters the analysis.
Events input: `04_ANALYSIS_RESULTS/03_PHASE2_PROCESS_STRUCTURE/processed/events_primary_cohort.parquet`.
Destination read window: `2018-01-01T00:00:00+00:00` through `2019-01-18T13:34:01+00:00` as an exclusive upper bound.
Reconstructed H1-origin observations: **659,827**.
Phase 3B strict destination-timestamp observations: **601,018**.

## Reproduction gate

The candidate set is frozen at 63 transitions. The primary configuration is `C = 2160` hours, `alpha = 0.95`, a 24-hour grid, and the original Phase 3 primary Wasserstein radius.
Rows passing every reproduction metric: **63 of 63**.
Rows with a reproduction mismatch: **0**.

Maximum absolute discrepancies

| Metric | Maximum absolute error |
|---|---:|
| h1_observations | 0 |
| h1_frequency_share | 9.97465998687e-17 |
| h1_mean_capped_elapsed_hours | 2.27373675443e-13 |
| h1_empirical_cvar95_hours | 4.54747350886e-13 |
| h1_grid_empirical_cvar95_hours | 4.54747350886e-13 |
| epsilon_primary_hours | 0 |
| h1_dro_worst_case_cvar95_hours | 1.11413100967e-11 |
| mean_burden_score | 1.42108547152e-14 |
| tail_score | 5.68434188608e-14 |
| dro_score | 5.68434188608e-14 |

## H1 boundary bridge rows

Bridge rows are H1-origin rows with a destination timestamp on or after 1 July 2018. The aligned sample contains **58,809** bridge rows across **214** transitions.
Bridge rows represent **8.9128%** of the exact H1-origin sample.

The four Phase 3B classification discrepancies are inspected below.

| Transition | Phase 3 H1 | Phase 3B strict | Bridge | m_cap before | m_cap after | epsilon_sat before | epsilon_sat after | Empirical before | Empirical after | Predicted before | Predicted after | Observed before | Observed after | Total before | Total after |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Record Invoice Receipt -> Record Invoice Receipt | 2,196 | 2,132 | 64 | 0.00093809 | 0.00683060 | 67.400375 | 50.083060 | False | False | False | True | True | True | False | True |
| Change Quantity -> Change Quantity | 1,181 | 1,147 | 34 | 0.00174368 | 0.00169348 | 60.946818 | 54.140220 | False | False | False | True | True | True | False | True |
| Record Goods Receipt -> Cancel Goods Receipt | 532 | 524 | 8 | 0.00381679 | 0.00563910 | 66.906870 | 53.332331 | False | False | False | True | True | True | False | True |
| Change Quantity -> Change Delivery Indicator | 511 | 506 | 5 | 0.00000000 | 0.00391389 | 66.720949 | 53.812133 | False | False | False | True | True | True | False | True |

## Aligned saturation classification

Empirical saturation count: **8**.
DRO-induced saturation count: **28**.
Total saturation count: **36 of 63**.
Unsaturated count: **27**.
Theoretical-vs-observed saturation mismatch count: **0**.

The quantity `epsilon_sat` is the bounded-support Wasserstein-CVaR saturation threshold diagnostic. It is an analytical consequence used here as a diagnostic and verified on the aligned empirical sample.
For positive thresholds, the computation moves the required probability mass to the cap from the largest sub-cap grid point downward. When the empirical cap mass already reaches the upper-tail mass, `epsilon_sat = 0` is handled explicitly.

## Linear-program validation

Validation passed for **63 of 63** transitions.
Positive epsilon_sat cases: **55**.
Zero epsilon_sat cases: **8**.
Maximum absolute error at epsilon = 0: **2.52384779742e-11** hours.
Maximum absolute error at epsilon = epsilon_sat: **9.36779542826e-11** hours.

At zero radius, the LP agrees with grid empirical CVaR. At the saturation threshold, it reaches the cap within tolerance. At `0.99 * epsilon_sat`, every positive-threshold case remains below the cap outside tolerance.

## Aligned ranking diagnostics

Tail versus DRO Spearman correlation: **0.9641284129107152**.
Tail versus DRO Kendall correlation: **0.8634384090730585**.
Mean burden versus DRO Spearman correlation: **0.9163396340293859**.
Mean burden versus DRO Kendall correlation: **0.7701768123660575**.

| K | Tail versus DRO Jaccard | Overlap count | DRO portfolio saturated | DRO portfolio unsaturated |
|---:|---:|---:|---:|---:|
| 3 | 1.00000000 | 3 | 3 | 0 |
| 5 | 1.00000000 | 5 | 4 | 1 |
| 10 | 0.81818182 | 9 | 6 | 4 |

## Phase 3B versus Phase 3C

Transitions whose total saturation classification changed: **4**.
The comparison table retains the observation count, cap mass, saturation threshold, finite saturation ratio, and empirical, predicted, observed, and total classifications for all 63 transitions.

## Integrity statement

Phases 1 through 3B were not modified. The repository contains the audit code, report, summary, and six CSV tables. It does not contain the source Parquet file.
