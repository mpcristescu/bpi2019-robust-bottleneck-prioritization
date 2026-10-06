# Robust Bottleneck Prioritization from Event Logs under Temporal Shift: A Wasserstein-CVaR Diagnostic Framework

Analysis repository for the BPI Challenge 2019 development log and the BPI Challenge 2017 external validation log.

Author: Marian Pompiliu Cristescu
Affiliation: Lucian Blaga University of Sibiu, Romania
Contact: marian.cristescu@ulbsibiu.ro
Public repository URL: https://github.com/mpcristescu/bpi2019-robust-bottleneck-prioritization

## Scope

The repository contains the scripts used for event-log auditing, cohort construction, process-structure analysis, temporal calibration, Wasserstein-CVaR evaluation, saturation diagnosis, sample-alignment checks, and external validation. It also contains derived CSV, JSON, and Markdown outputs.

Raw event logs, processed Parquet files, manuscript files, and image outputs are not included. Official source links and checksums are recorded in DATASET_LINKS.md.

The BPI 2019 outcome is inter-event elapsed time between consecutive recorded events. It is not treated as service time or pure waiting time because the log does not provide paired activity start and complete timestamps.

## Workflow

### Availability-aware analysis version 1.1.0

The manuscript tables use the workflow in `analysis/` and the aggregate results in `results/availability_analysis/`. Historical observations require both event timestamps to precede the 1 July decision cutoff. BPI 2019 uses the recorded events of cases first observed in 2018 without excluding a case because of a later anomaly. BPI 2017 uses COMPLETE events. Event timestamps are availability proxies, not verified ingestion times.

Run from the repository root after downloading the official event logs:

```powershell
python -m pip install -r requirements.txt
python -m analysis.run_analysis --bpi2019 "PATH_TO_BPI_Challenge_2019.xes" --bpi2017 "PATH_TO_BPI Challenge 2017.xes.gz" --cache "PRIVATE_CACHE_OUTSIDE_REPOSITORY"
python -m analysis.publication_tables
python -m unittest discover -s tests -v
```

Keep the cache outside the repository. The analysis exports only aggregate tables and source checksums. Figure preparation and manuscript-editing scripts are not included.

The primary model uses the original empirical support, a 90-day cap, alpha 0.95, a 14-day calibration block, at least 500 historical observations, and valid blocks covering at least 60% of the calendar blocks. A valid block contains at least 20 observations. The coverage rule requires eight of the thirteen H1 blocks and adapts coherently to the 7-day, 28-day and maturity-lag specifications. Calibration sensitivity outputs retain the primary candidate set wherever a qualifying block exists and also report strict eligibility counts.

Worst-case CVaR is computed exactly as `min(C, empirical_CVaR + epsilon/(1-alpha))`. The dual linear program and greedy transport calculation are independent numerical checks. Radius calibration uses the historical sample only. Every method uses identical Q3 resamples. CVaR is recalculated in every resample. Identical portfolios must have exactly zero paired differences.

These are retrospective decision-snapshot evaluations, not new blind tests. Outcomes are conditional on a successor recorded by the observation cutoff. The availability audit distinguishes later-recorded successors from events without any recorded successor. It does not equate the latter with censoring or verified termination.

### Table and code mapping

| Manuscript material | Aggregate output in each log directory |
| --- | --- |
| Tables 3 and 5, mean and tail results and oracles | `holdout_evaluation.csv` |
| Table 4, saturation regimes | `historical_scores.csv`, `summary.json` |
| Tables 6 and 7, calibration and cap sensitivity | `sensitivity_summary.csv` |
| Table 8, Kendall tau and oracle overlap | `holdout_evaluation.csv` |
| Availability and maturation audit | `availability_audit.csv`, `availability_by_month.csv` |
| Radius multipliers and selection boundaries | `sensitivity_summary.csv`, `radius_boundary_audit.csv` |
| Clipping-factor sensitivity | `clipping_diagnostics.csv`, `sensitivity_summary.csv` |
| Original-support and grid comparison | `grid_comparison.csv`, `sensitivity_transition_risks.csv` |
| Paired confidence intervals | `paired_bootstrap.csv` |
| Full historical selection refits | `selection_stability.csv`, `selection_probabilities.csv` |
| Exact selected sets and within-set ranks | `portfolios.csv`, `sensitivity_portfolios.csv` |
| Formula, transport and LP agreement | `mathematical_checks.csv` |
| Finite-sample permutation benchmark | `sampling_variability_benchmark.csv` |

`analysis_manifest.json` records the parameters, runtime versions, source checksums, code checksums, portfolio checksums and all aggregate-table checksums. No local absolute data paths are embedded in that manifest.

### Historical exploratory phases

The phase directories below preserve the earlier exploratory workflow. Their origin-only analyses and derived tables are not the source of version 1.1.0 manuscript results. Use the availability-aware workflow above for the current manuscript.

1. 01_PHASE1_AUDIT: audit the BPI 2019 XES structure.
2. 02_PHASE1B_TEMPORAL_AUDIT: quantify timestamp and resource semantics.
3. 03_PHASE2_PROCESS_STRUCTURE: build the primary cohort and processed Parquet files.
4. 04_PHASE2B_MATURITY_CENSORING_AUDIT: assess observation-edge maturity.
5. 05_PHASE3_DRO_BOTTLENECK_PRIORITIZATION: calibrate transition-specific Wasserstein radii and evaluate frozen rankings.
6. 06_PHASE3B_CVAR_SATURATION_ANALYSIS: compute the bounded-support saturation threshold and validate it with linear programs and synthetic distributions.
7. 07_PHASE3C_SAMPLE_ALIGNMENT_AUDIT: reproduce the exact H1-origin sample used by the primary model.
8. 08_PHASE4_EXTERNAL_VALIDATION_BPI2017: apply the exploratory rules to the external log.
9. 09_PHASE4B_EXTERNAL_VALIDATION_QA: reproduce the external primary result and evaluate frozen sensitivity portfolios.

Each phase writes its tables and reports to the matching directory under 04_ANALYSIS_RESULTS. The phase scripts use the standard paths in this repository. Environment variables BPI2019_XES, BPI2019_EVENTS_PARQUET, and BPI2019_CASES_PARQUET can be used when local files are stored elsewhere.

## Environment

Python 3.11 or later is recommended. Each phase includes setup_windows.bat and requirements.txt. The requirements contain only libraries used by the analysis.
