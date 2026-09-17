# Phase 3: Distributionally Robust Bottleneck Prioritization

## Data and temporal split

- Primary cohort: 251,266 cases and 1,587,374 events.
- H1 training window: 2018-01-01T00:00:00+00:00 through 2018-06-30 23:59:59 UTC.
- Q3 holdout: 2018-07-01T00:00:00+00:00 through 2018-09-30 23:59:59 UTC.
- H1-only candidate transitions: 63.
- Phase 2B diagnostic transitions used only for descriptive comparison: 39.

The outcome is elapsed time between two consecutive recorded events. It is not interpreted as a task execution duration or as pure queue delay because the event log has no lifecycle start/complete pairs.

## H1-only candidate rule

Candidate transitions have at least 500 H1 observations, at least 8 valid 14-day H1 blocks, and at least 20 observations in every valid block. Candidate selection was completed before Q3-origin observations were loaded.

## Primary DRO specification

- Delay cap: 90 days = 2160 hours.
- Tail level: alpha = 0.95.
- Common support grid: 24 hours.
- Epsilon: the H1 valid-block Wasserstein-1 q90 for each transition.
- Worst-case CVaR: linear programs solved with scipy.optimize.linprog using absolute CDF differences and auxiliary variables.
- Sensitivity configurations use an exact one-dimensional fractional-transport shortcut for the same W1 hinge problem; the explicit LP remains authoritative for the primary model and epsilon = 0 QA.

The epsilon = 0 QA passed for all H1 candidates within the discretization tolerance. The maximum absolute difference between the zero-radius LP and the grid empirical CVaR is 0.0000 hours.

## Primary DRO top-5

| Rank | Transition | H1 frequency share | H1 mean burden | H1 empirical CVaR95 | H1 DRO CVaR95 | DRO score |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Record Invoice Receipt -> Clear Invoice | 10.135% | 1156.68 | 2160.00 | 2160.00 | 218.92 |
| 2 | Create Purchase Order Item -> Vendor creates invoice | 9.090% | 304.08 | 1608.05 | 2160.00 | 196.34 |
| 3 | Record Goods Receipt -> Record Invoice Receipt | 8.743% | 476.82 | 2160.00 | 2160.00 | 188.85 |
| 4 | Vendor creates invoice -> Record Invoice Receipt | 7.461% | 283.08 | 1827.82 | 2160.00 | 161.16 |
| 5 | Create Purchase Order Item -> Record Goods Receipt | 6.179% | 363.90 | 1425.17 | 2139.85 | 132.23 |

## Q3 K = 5 evaluation

| Method | Q3 captured mean-burden share | Q3 captured tail-burden share | Mean-burden regret | Jaccard vs Q3 mean oracle |
|---|---:|---:|---:|---:|
| Frequency | 55.548% | 46.363% | 16.426 pp | 0.250 |
| Mean burden | 70.229% | 59.235% | 1.745 pp | 0.429 |
| Tail baseline | 66.972% | 60.943% | 5.002 pp | 0.429 |
| DRO | 66.972% | 60.943% | 5.002 pp | 0.429 |

## Q3 block-bootstrap comparisons at K = 5

| Comparison | Mean difference | 95% CI | CI includes zero |
|---|---:|---:|---|
| DRO minus Frequency | 11.419 pp | [7.906, 14.761] pp | False |
| DRO minus Mean burden | -3.219 pp | [-6.441, -0.233] pp | False |
| DRO minus Tail baseline | 0.027 pp | [-3.681, 3.805] pp | True |

## Decision

Defensible out-of-time DRO advantage: **no**.
A positive point estimate is not treated as superiority when the corresponding percentile bootstrap interval includes zero.

## Sensitivity

The sensitivity grid includes 90-day and 120-day caps, alpha values 0.95 and 0.90, 12-hour and 24-hour discretizations, epsilon multipliers 0, 0.5, 1.0, 1.5, and 2.0, and K values 3, 5, and 10.
The 120-day cap equals 2880 hours. The primary 90-day cap equals 2160 hours.

## Reproducibility and leakage controls

- Q3 was not used before evaluation for candidate selection, epsilon calibration, method choice, hyperparameter choice, or portfolio construction.
- H1 candidate selection is H1-only.
- Epsilon calibration is H1-only.
- The random seed is 20260914.
- Phases 1, 1B, 2, and 2B were not modified.
- The results exclude the original XES file and the large Parquet files.

## Warnings
- The outcome is inter-event elapsed time reconstructed from consecutive recorded events. It is not a task execution duration or pure queue delay.
- The Phase 2B 39-transition list is diagnostic only; the final candidate set was selected from H1 with fixed thresholds.
- For H1 bootstrap stability, the full-H1 calibrated DRO CVaR and epsilon values are held fixed while 14-day blocks are resampled. This measures portfolio stability around the frozen H1 model rather than refitting the ambiguity radius in every replicate.
- The Q3 oracle is a retrospective benchmark and is not used to build the model.
- The 95% Q3 bootstrap interval for DRO minus Tail baseline includes zero; no superiority claim is made for that comparison.
