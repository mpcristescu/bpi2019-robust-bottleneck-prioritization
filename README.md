# Robust Bottleneck Prioritization from Event Logs under Temporal Shift: A Wasserstein-CVaR Diagnostic Framework

Analysis repository for the BPI Challenge 2019 development log and the BPI Challenge 2017 external validation log.

Author: Marian Pompiliu Cristescu
Affiliation: Lucian Blaga University of Sibiu, Romania
Contact: marian.cristescu@ulbsibiu.ro
Public repository URL: https://github.com/REPLACE_WITH_GITHUB_USERNAME/bpi2019-robust-bottleneck-prioritization

## Scope

The repository contains the scripts used for event-log auditing, cohort construction, process-structure analysis, temporal calibration, Wasserstein-CVaR evaluation, saturation diagnosis, sample-alignment checks, and external validation. It also contains derived CSV, JSON, and Markdown outputs.

Raw event logs, processed Parquet files, manuscript files, and image outputs are not included. Official source links and checksums are recorded in DATASET_LINKS.md.

The BPI 2019 outcome is inter-event elapsed time between consecutive recorded events. It is not treated as service time or pure waiting time because the log does not provide paired activity start and complete timestamps.

## Workflow

Run the phases in order from their respective folders under 03_ANALYSIS_REPO.

1. 01_PHASE1_AUDIT: audit the BPI 2019 XES structure.
2. 02_PHASE1B_TEMPORAL_AUDIT: quantify timestamp and resource semantics.
3. 03_PHASE2_PROCESS_STRUCTURE: build the primary cohort and processed Parquet files.
4. 04_PHASE2B_MATURITY_CENSORING_AUDIT: assess observation-edge maturity and define the cutoff-safe temporal design.
5. 05_PHASE3_DRO_BOTTLENECK_PRIORITIZATION: calibrate transition-specific Wasserstein radii and evaluate frozen rankings.
6. 06_PHASE3B_CVAR_SATURATION_ANALYSIS: compute the bounded-support saturation threshold and validate it with linear programs and synthetic distributions.
7. 07_PHASE3C_SAMPLE_ALIGNMENT_AUDIT: reproduce the exact H1-origin sample used by the primary model.
8. 08_PHASE4_EXTERNAL_VALIDATION_BPI2017: apply the frozen rules to the untouched external log.
9. 09_PHASE4B_EXTERNAL_VALIDATION_QA: reproduce the external primary result and evaluate frozen sensitivity portfolios.

Each phase writes its tables and reports to the matching directory under 04_ANALYSIS_RESULTS. The phase scripts use the standard paths in this repository. Environment variables BPI2019_XES, BPI2019_EVENTS_PARQUET, and BPI2019_CASES_PARQUET can be used when local files are stored elsewhere.

## Environment

Python 3.11 or later is recommended. Each phase includes setup_windows.bat and requirements.txt. The requirements contain only libraries used by the analysis.
