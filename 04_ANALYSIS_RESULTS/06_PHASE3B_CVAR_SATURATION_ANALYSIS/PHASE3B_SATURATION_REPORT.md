# Phase 3B: CVaR Saturation Analysis

## Scope and data

Q3 was NOT read or used in Phase 3B.

The analysis uses 601,018 event rows whose recorded timestamps satisfy the strict H1 interval `[2018-01-01 00:00:00 UTC, 2018-07-01 00:00:00 UTC)`. The candidate universe and primary radii were read only from the three permitted H1 Phase 3 CSV files.

The analysis contains `candidate_count = 63`, `C = 2160 hours`, `alpha = 0.95`, and `grid step = 24 hours`. The random seed is `seed = 20260914`.

The response variable is elapsed time between consecutive recorded events. No activity-level duration is inferred from this interval.

## Saturation construction

For each candidate transition, values are capped at `C` and mapped to the same 24-hour support grid used in Phase 3. If `m_C` is the empirical probability at the cap, the extra mass required to fill the upper CVaR tail is `delta = max(0, 1 - alpha - m_C)`. The implementation transports this mass from the nearest sub-cap grid points to `C` and sums `mass_moved * (C - x)`.

The resulting quantity is `epsilon_sat`. For positive thresholds, `saturation_ratio = epsilon_primary / epsilon_sat`. The ratio is left undefined for distributions already saturated at zero radius.

## H1 empirical findings

| Quantity | Count | Percentage |
|---|---:|---:|
| Empirically saturated | 3 | 4.762% |
| DRO-induced saturation | 29 | 46.032% |
| Total saturated under primary radius | 32 | 50.794% |
| Unsaturated | 31 | 49.206% |

Finite positive saturation ratios have the following distribution:

`median = 1.014843`, `q75 = 3.267240`, `q90 = 8.200766`, `max = 485.222635`.

## Rank association and portfolios

Tail score versus DRO score gives Spearman `rho = 0.964128` and Kendall `tau = 0.863438`. Mean burden score versus DRO score gives Spearman `rho = 0.916340` and Kendall `tau = 0.770177`.

| K | Tail versus DRO Jaccard | Overlap | DRO saturated | DRO unsaturated |
|---:|---:|---:|---:|---:|
| 3 | 1.000000 | 3 | 3 | 0 |
| 5 | 1.000000 | 5 | 4 | 1 |
| 10 | 0.818182 | 9 | 6 | 4 |

## LP validation

The epsilon=0 QA passed for 63 of 63 transition rows under the declared tolerance of `1e-05` hours. The maximum zero-radius absolute error was `2.02362571144e-11` hours.

The epsilon_sat cap test passed for all rows with positive epsilon_sat: `True`. The `0.99 * epsilon_sat` sub-cap test passed for all applicable rows: `True`.

The theoretical condition `epsilon_primary >= epsilon_sat` was compared with the permitted observed Phase 3 saturation flag. Mismatch count: `4`.

The mismatches are retained in `01_transition_saturation_thresholds.csv` and are not hidden. They are classified as follows:

| Transition | Predicted | Observed | Reason |
|---|---:|---:|---|
| Record Invoice Receipt -> Record Invoice Receipt | False | True | Phase 3 observed metric uses its H1-origin extraction convention, while Phase 3B uses the strict H1 timestamp window |
| Change Quantity -> Change Quantity | False | True | Phase 3 observed metric uses its H1-origin extraction convention, while Phase 3B uses the strict H1 timestamp window |
| Record Goods Receipt -> Cancel Goods Receipt | False | True | Phase 3 observed metric uses its H1-origin extraction convention, while Phase 3B uses the strict H1 timestamp window |
| Change Quantity -> Change Delivery Indicator | False | True | Phase 3 observed metric uses its H1-origin extraction convention, while Phase 3B uses the strict H1 timestamp window |

## Synthetic saturation study

The deterministic synthetic study used 2001 support points on `[0, 1]`, four Beta distributions, and two mixtures with point mass at one. It evaluated alpha values `[0.9, 0.95, 0.99]` and the requested epsilon ratios. The condition at ratio greater than or equal to one passed for every alpha=0.95 synthetic row: `True`.

The threshold transition is therefore visible independently of the BPI 2019 event log. Distributions with a cap mass already covering the upper-tail probability are represented as saturated at zero radius.

## Diagnostic radius sensitivity

The kappa analysis keeps the primary radius capped at `min(epsilon_primary, kappa * epsilon_sat)` for kappa equal to 0.50, 0.75, and 0.90. It reports the resulting H1 CVaR scores, ranks, portfolios, and saturation counts. No kappa is selected as best. No Q3 data is used.

- kappa = 0.50, saturated transitions = 3, top-5 = Record Invoice Receipt -> Clear Invoice, Record Goods Receipt -> Record Invoice Receipt, Create Purchase Order Item -> Vendor creates invoice, Vendor creates invoice -> Record Invoice Receipt, Create Purchase Order Item -> Record Goods Receipt
- kappa = 0.75, saturated transitions = 3, top-5 = Record Invoice Receipt -> Clear Invoice, Create Purchase Order Item -> Vendor creates invoice, Record Goods Receipt -> Record Invoice Receipt, Vendor creates invoice -> Record Invoice Receipt, Create Purchase Order Item -> Record Goods Receipt
- kappa = 0.90, saturated transitions = 3, top-5 = Record Invoice Receipt -> Clear Invoice, Create Purchase Order Item -> Vendor creates invoice, Record Goods Receipt -> Record Invoice Receipt, Vendor creates invoice -> Record Invoice Receipt, Create Purchase Order Item -> Record Goods Receipt

## Reproducibility and integrity checks

| Check | Status |
|---|---|
| Q3 was NOT read or used in Phase 3B | PASS |
| candidate_count = 63 | PASS |
| C = 2160 hours | PASS |
| alpha = 0.95 | PASS |
| grid step = 24 hours | PASS |
| epsilon_sat formula implemented | PASS |
| epsilon=0 QA passed | True |
| Theoretical saturation condition compared with LP observation | PASS |
| seed = 20260914 | PASS |
| Phases 1 to 3 were not modified | PASS |

The full derivation is in `THEORETICAL_NOTE.md`. The outputs exclude the XES file and large Parquet files.

The empirical saturation result explains why bounded-support worst-case CVaR values can collapse to the cap when the calibrated radius crosses the transport threshold. It does not establish out-of-time predictive superiority for a new method.
