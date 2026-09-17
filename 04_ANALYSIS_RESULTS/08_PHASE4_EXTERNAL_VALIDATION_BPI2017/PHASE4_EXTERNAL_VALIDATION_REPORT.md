# Phase 4, External Validation on BPI Challenge 2017

## Dataset and provenance

The external event log is the official BPI Challenge 2017 dataset from Figshare article 12696884.
Official page: `https://figshare.com/articles/dataset/BPI_Challenge_2017/12696884`.
DOI: `10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b`.
Time coverage in public metadata: `2016-01-01/2017-02-01`.
License: `4TU General Terms of Use`.
Original file: `BPI Challenge 2017.xes.gz`.
Official download URL: `https://ndownloader.figshare.com/files/24044117`.
File size: **29,658,747 bytes**.
SHA-256: `183c5e5189282779c811c78c33ff936351b3dd201165d612211fc220936f8249`.

## Preprocessing

The primary preprocessing retained only events with `lifecycle:transition == COMPLETE`, after case-insensitive normalization. START and SCHEDULE events were excluded. The analytical label was `concept:name`, and the case identifier was the trace-level `concept:name`.
Complete events were ordered chronologically within each case. The original XES order was used as the stable tie-break for equal timestamps. The outcome is inter-event elapsed time between two consecutive COMPLETE events.

| Preprocessing item | Count |
|---|---:|
| Traces | 31,509 |
| Raw events | 1,202,267 |
| COMPLETE events | 475,306 |
| Excluded non-COMPLETE events | 726,961 |
| COMPLETE events missing required fields | 0 |
| COMPLETE events missing case id | 0 |
| COMPLETE events missing activity | 0 |
| COMPLETE events with invalid timestamp | 0 |
| Valid COMPLETE events | 475,306 |
| Direct-follow pairs before negative elapsed filter | 443,797 |
| Rows removed for negative elapsed time | 0 |
| Final direct-follow rows | 193,194 |
| Consecutive timestamp tie pairs | 0 |
| Cases containing timestamp ties | 0 |
| Timestamp tie groups | 0 |

No activity duration was created from lifecycle pairs. No negative elapsed row was retained.

## Temporal design and leakage control

H1 was fixed as origins from 1 January 2016 inclusive to 1 July 2016 exclusive. Q3 was fixed as origins from 1 July 2016 inclusive to 1 October 2016 exclusive.
Q3 was not accessed before portfolio freeze. Before freeze, the extraction returned H1-origin rows only, all candidate selection used H1 only, all epsilon calibration used H1 only, and kappa = 0.90 was fixed. The Q3-origin holdout was opened only after the frozen selection manifest was written and checksummed.
Frozen portfolio manifest SHA-256: `9b56a20f976afabac60c5f83e0a50ec6c5a13c8056a24deced6f57e43194e21f`.

## H1 feasibility gate

H1 direct-follow rows: **193,194**.
Distinct H1 activities: **24**.
Distinct H1 transitions: **145**.
Frozen H1-only candidate transitions: **41**.
The fixed candidate rule was 500 H1 observations, at least 8 valid 14-day blocks, and at least 20 observations in each valid block. The feasibility gate passed before Q3 evaluation.

## Primary H1 model

The primary cap was 2160 hours, alpha = 0.95, and the grid step was 24 hours. Temporal ambiguity radius was the q90 of H1 block Wasserstein-1 distances from each transition to its pooled H1 distribution.
The saturation-specific quantity is used as a bounded-support diagnostic. Original DRO uses the full temporal radius. SA-DRO uses epsilon_SA = min(epsilon_temporal, 0.90 * epsilon_sat) when epsilon_sat is positive, and epsilon_SA = 0 when epsilon_sat is zero. The value kappa = 0.90 was fixed before Q3.
Candidate transitions: **41**.
Empirical saturation: **0**.
Original-DRO induced saturation: **2**.
Original-DRO total saturation: **2 of 41**.
SA-DRO total saturation: **0 of 41**.
Median finite saturation_ratio: **0.0064719260**.
Original-DRO theoretical-vs-observed saturation mismatches: **0**.

## LP validation

LP validation passed for **41 of 41** candidates.
epsilon = 0 QA passed: **True**.
epsilon = epsilon_sat QA passed: **True**.
epsilon = 0.99 * epsilon_sat QA passed outside tolerance for positive thresholds: **True**.
Maximum zero-radius absolute error: **4.32009983342e-12** hours.
Maximum saturation-threshold absolute error: **6.36646291241e-11** hours.

## External Q3 evaluation

Q3-origin direct-follow rows: **129,191**.
Q3 transition universe: **140**.
The Q3 oracle is retrospective and was not used for model construction, candidate selection, epsilon calibration, kappa selection, or portfolio freeze.

### Primary K = 5 captured mean-burden share

| Method | Q3 captured mean-burden share | Mean-burden regret |
|---|---:|---:|
| Frequency | 7.327012% | 63.321974 pp |
| Mean burden | 70.237306% | 0.411680 pp |
| Empirical Tail | 70.237306% | 0.411680 pp |
| Original DRO | 70.237306% | 0.411680 pp |
| Saturation-Aware DRO | 70.237306% | 0.411680 pp |

## Q3 bootstrap uncertainty

The block bootstrap used **1000** replicates over **14** nonempty weekly blocks with seed **20260914**.

| Comparison | Difference mean | 95% CI | Includes zero |
|---|---:|---:|:---:|
| SA-DRO minus Original DRO | 0.000000 pp | [0.000000, 0.000000] | True |
| SA-DRO minus Mean burden | 0.000000 pp | [0.000000, 0.000000] | True |
| SA-DRO minus Empirical Tail | 0.000000 pp | [0.000000, 0.000000] | True |
| SA-DRO minus Frequency | 62.878904 pp | [61.025452, 64.688415] | False |

Defensible external SA-DRO advantage under the pre-specified rule: **False**.
The rule requires the lower endpoint of every listed SA-DRO minus baseline interval to be strictly positive. When an interval includes zero, no superiority claim is made for that comparison.

## H1 stability

| Method | Mean Jaccard versus full H1 | Transitions with inclusion probability at least 0.8 |
|---|---:|---:|
| Mean burden | 0.910000 | 4 |
| Empirical Tail | 1.000000 | 5 |
| Original DRO | 0.999333 | 5 |
| Saturation-Aware DRO | 0.999333 | 5 |

## Sensitivity

Sensitivity configurations were run without selecting a preferred configuration from Q3. They vary the cap, alpha, grid step, and SA-DRO kappa according to the pre-specified primary and sensitivity values. K values 3, 5, and 10 were retained in every configuration.
Sensitivity rows: **120**.

## Integrity

Random seed: **20260914**.
The outcome is inter-event elapsed time. The primary preprocessing is COMPLETE-only. H1 origins are January to June 2016. Q3 origins are July to September 2016.
Phases 1 through 3C were not modified. The results exclude the XES.GZ dataset and contain code, output tables, and provenance metadata.

## Warnings

- The 95% Q3 weekly block bootstrap interval for SA-DRO minus Original DRO includes zero.
- The 95% Q3 weekly block bootstrap interval for SA-DRO minus Mean burden includes zero.
- The 95% Q3 weekly block bootstrap interval for SA-DRO minus Empirical Tail includes zero.
- No superiority claim is made for SA-DRO because at least one pre-specified interval includes zero.
