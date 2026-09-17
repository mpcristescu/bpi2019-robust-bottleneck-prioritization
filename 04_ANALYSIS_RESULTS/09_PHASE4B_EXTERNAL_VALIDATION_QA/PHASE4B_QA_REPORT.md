# Phase 4B, External Validation QA

## Scope

This QA stage reads the completed Phase 4 outputs, corrects the H1 score-correlation labels, and evaluates only the sensitivity portfolios that were frozen before Q3 access. No primary selection was changed and no new Q3-driven selection was created.

## Primary reproduction gate

The Phase 4 primary reproduction gate passed: **True**.
Candidate transitions: **41**.
Primary configuration: C = **2160 h**, alpha = **0.95**, grid = **24 h**, SA kappa = **0.9**, confirmatory K = **5**.
Primary K=5 captured mean-burden shares were reproduced from the Phase 4 Q3 evaluation output.

| Method | Captured Q3 mean-burden share |
|---|---:|
| Frequency | 7.3270119674% |
| Mean burden | 70.2373056645% |
| Empirical Tail | 70.2373056645% |
| Original DRO | 70.2373056645% |
| Saturation-Aware DRO | 70.2373056645% |

Original-DRO total saturation: **2/41**.
SA-DRO total saturation: **0/41**.
The original frozen manifest checksum is unchanged: 9b56a20f976afabac60c5f83e0a50ec6c5a13c8056a24deced6f57e43194e21f.

## Direct-follow row labels

The full XES preprocessing produced 443,797 COMPLETE direct-follow pair candidates. No negative elapsed rows were removed. The H1 and Q3 counts are window-specific retained direct-follow rows, not the global total.

| Quantity | Count |
|---|---:|
| All COMPLETE direct-follow pair candidates in the full XES | 443,797 |
| Negative elapsed rows removed | 0 |
| H1-origin window-specific retained direct-follow rows | 193,194 |
| Q3-origin window-specific retained direct-follow rows | 129,191 |

## Corrected H1 score correlations

The four Tail and Mean-burden comparisons with the two DRO scores were calculated directly from the 41 rows in 04_external_transition_scores.csv. The H1-to-Q3 correlations from Phase 4 are retained separately with their scope stated explicitly.

| Scope | Comparison | Method | Type | Value | n |
|---|---|---|---|---:|---:|
| H1 score-to-score | Tail score vs Original-DRO score |  | Spearman | 0.9871080139 | 41 |
| H1 score-to-score | Tail score vs Original-DRO score |  | Kendall | 0.9268292683 | 41 |
| H1 score-to-score | Tail score vs SA-DRO score |  | Spearman | 0.9871080139 | 41 |
| H1 score-to-score | Tail score vs SA-DRO score |  | Kendall | 0.9268292683 | 41 |
| H1 score-to-score | Mean-burden score vs Original-DRO score |  | Spearman | 0.9876306620 | 41 |
| H1 score-to-score | Mean-burden score vs Original-DRO score |  | Kendall | 0.9219512195 | 41 |
| H1 score-to-score | Mean-burden score vs SA-DRO score |  | Spearman | 0.9876306620 | 41 |
| H1 score-to-score | Mean-burden score vs SA-DRO score |  | Kendall | 0.9219512195 | 41 |

## Frozen sensitivity Q3 evaluation

The manifest contains **8** pre-frozen combined sensitivity configurations. Their **120** portfolios were evaluated on Q3 using the Phase 4 primary Q3 metric definition, with C = 2160 h and alpha = 0.95.
The configurations combine changes in cap, alpha, grid step, and SA-DRO kappa. The results are not one-factor-at-a-time effects.

SA-DRO Q3 captured mean-burden share across the frozen sensitivity configurations and K values ranged from **32.7903705440%** to **84.5676150877%**.
At K=5, the corresponding range was **70.2373056645%** to **70.2373056645%**.

## K=5 portfolio stability

SA-DRO had the same K=5 set as the primary portfolio in **8 of 8** configurations.

| Configuration | Method | Overlap | Jaccard | Same set | Same order | Membership differs |
|---|---|---:|---:|:---:|:---:|:---:|
| primary | Frequency | 5 | 1.000000 | True | True | False |
| primary | Mean burden | 5 | 1.000000 | True | True | False |
| primary | Empirical Tail | 5 | 1.000000 | True | True | False |
| primary | Original DRO | 5 | 1.000000 | True | True | False |
| primary | Saturation-Aware DRO | 5 | 1.000000 | True | True | False |
| cap120 | Frequency | 5 | 1.000000 | True | True | False |
| cap120 | Mean burden | 5 | 1.000000 | True | True | False |
| cap120 | Empirical Tail | 5 | 1.000000 | True | True | False |
| cap120 | Original DRO | 5 | 1.000000 | True | True | False |
| cap120 | Saturation-Aware DRO | 5 | 1.000000 | True | True | False |
| alpha90 | Frequency | 5 | 1.000000 | True | True | False |
| alpha90 | Mean burden | 5 | 1.000000 | True | True | False |
| alpha90 | Empirical Tail | 5 | 1.000000 | True | False | False |
| alpha90 | Original DRO | 5 | 1.000000 | True | True | False |
| alpha90 | Saturation-Aware DRO | 5 | 1.000000 | True | True | False |
| grid12 | Frequency | 5 | 1.000000 | True | True | False |
| grid12 | Mean burden | 5 | 1.000000 | True | True | False |
| grid12 | Empirical Tail | 5 | 1.000000 | True | True | False |
| grid12 | Original DRO | 5 | 1.000000 | True | True | False |
| grid12 | Saturation-Aware DRO | 5 | 1.000000 | True | False | False |
| cap120_alpha90 | Frequency | 5 | 1.000000 | True | True | False |
| cap120_alpha90 | Mean burden | 5 | 1.000000 | True | True | False |
| cap120_alpha90 | Empirical Tail | 5 | 1.000000 | True | False | False |
| cap120_alpha90 | Original DRO | 5 | 1.000000 | True | True | False |
| cap120_alpha90 | Saturation-Aware DRO | 5 | 1.000000 | True | True | False |
| cap120_grid12 | Frequency | 5 | 1.000000 | True | True | False |
| cap120_grid12 | Mean burden | 5 | 1.000000 | True | True | False |
| cap120_grid12 | Empirical Tail | 5 | 1.000000 | True | True | False |
| cap120_grid12 | Original DRO | 5 | 1.000000 | True | True | False |
| cap120_grid12 | Saturation-Aware DRO | 5 | 1.000000 | True | True | False |
| alpha90_grid12 | Frequency | 5 | 1.000000 | True | True | False |
| alpha90_grid12 | Mean burden | 5 | 1.000000 | True | True | False |
| alpha90_grid12 | Empirical Tail | 5 | 1.000000 | True | False | False |
| alpha90_grid12 | Original DRO | 5 | 1.000000 | True | True | False |
| alpha90_grid12 | Saturation-Aware DRO | 5 | 1.000000 | True | True | False |
| cap120_alpha90_grid12 | Frequency | 5 | 1.000000 | True | True | False |
| cap120_alpha90_grid12 | Mean burden | 5 | 1.000000 | True | True | False |
| cap120_alpha90_grid12 | Empirical Tail | 5 | 1.000000 | True | False | False |
| cap120_alpha90_grid12 | Original DRO | 5 | 1.000000 | True | True | False |
| cap120_alpha90_grid12 | Saturation-Aware DRO | 5 | 1.000000 | True | True | False |

## Primary bootstrap verification

The Phase 4 primary bootstrap output was read without rerunning it. The intervals for SA-DRO minus Original DRO, SA-DRO minus Mean burden, and SA-DRO minus Empirical Tail are [0, 0]. The interval for SA-DRO minus Frequency is approximately [61.02545, 64.68841] percentage points.
The primary conclusion is unchanged. The external validation does not support a defensible superiority claim for SA-DRO over the strongest baselines.

## External saturation diagnostic

BPI 2017 external H1 has empirical saturation **0/41**, Original-DRO induced saturation **2/41**, SA-DRO saturation **0/41**, and median finite epsilon_temporal/epsilon_sat **0.0064719260**.
The aligned BPI 2019 H1 values from Phase 3C are empirical saturation **8/63**, DRO-induced saturation **28/63**, total saturation **36/63**, and median finite saturation ratio **1.0182147911**.
The external validation therefore does not show that saturation is universal. It separates a low-saturation regime in BPI 2017 from the high-saturation regime observed in aligned BPI 2019 H1.
No Q3 data from BPI 2019 were read or used.

## Integrity and warnings

Dataset SHA-256 verification passed: 183c5e5189282779c811c78c33ff936351b3dd201165d612211fc220936f8249.
Only pre-frozen sensitivity portfolios were evaluated on Q3. Phase 4 primary output was not changed. Phases 1 through 4 were not modified.
- The corrected H1 correlations replace the mislabeled Tail-versus-DRO values in phase4_summary.json without changing Phase 4 outputs.
- Sensitivity configurations are pre-frozen combined configurations and are not one-factor-at-a-time effects.
- No superiority claim is made for SA-DRO because the primary K=5 portfolios coincide with the strongest baselines and the corresponding bootstrap intervals include zero.

## Reproduction checks

| Check | Expected | Observed | Passed |
|---|---|---|:---:|
| candidate_count | 41 | 41 | True |
| H1 direct-follow rows | 193194 | 193194 | True |
| Q3 direct-follow rows | 129191 | 129191 | True |
| primary cap hours | 2160 | 2160 | True |
| primary alpha | 0.95 | 0.95 | True |
| primary grid hours | 24 | 24 | True |
| primary SA kappa | 0.9 | 0.9 | True |
| primary K | 5 | 5 | True |
| primary K=5 Q3 share, Frequency | 7.3270119674 | 7.327011967434773 | True |
| primary K=5 Q3 share, Mean burden | 70.2373056645 | 70.2373056644679 | True |
| primary K=5 Q3 share, Empirical Tail | 70.2373056645 | 70.2373056644679 | True |
| primary K=5 Q3 share, Original DRO | 70.2373056645 | 70.2373056644679 | True |
| primary K=5 Q3 share, Saturation-Aware DRO | 70.2373056645 | 70.2373056644679 | True |
| Original-DRO saturated count | 2 | 2 | True |
| SA-DRO saturated count | 0 | 0 | True |
| candidate score rows | 41 | 41 | True |
| all score rows are H1-only candidates | True | True | True |
| Q3 accessed before freeze | False | False | True |
| manifest Q3 accessed before freeze | False | False | True |
| manifest checksum | 9b56a20f976afabac60c5f83e0a50ec6c5a13c8056a24deced6f57e43194e21f | 9b56a20f976afabac60c5f83e0a50ec6c5a13c8056a24deced6f57e43194e21f | True |
| summary manifest checksum | 9b56a20f976afabac60c5f83e0a50ec6c5a13c8056a24deced6f57e43194e21f | 9b56a20f976afabac60c5f83e0a50ec6c5a13c8056a24deced6f57e43194e21f | True |
| sensitivity rows | 120 | 120 | True |
| all sensitivity selections marked H1-only | True | True | True |
| Phases 1 to 3C modified | False | False | True |
| Original-DRO theoretical-observed mismatch count | 0 | 0 | True |
