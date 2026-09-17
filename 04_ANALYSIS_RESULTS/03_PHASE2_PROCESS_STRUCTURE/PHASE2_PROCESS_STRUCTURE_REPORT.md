# Phase 2 : Process Structure and Temporal Performance

- Cases: **251,266**
- Events: **1,587,374**
- Activities: **42**
- Observed directly-follow transitions: **494**
- Distinct process variants: **11,910**
- Top variants required for 80% case coverage: **45**
- Median cycle time: **1537.27 h**
- P95 cycle time: **3411.25 h**

## Critical interpretation

The log does not expose lifecycle start/complete pairs. All transition timing results are therefore reported as **elapsed time between consecutive recorded events**. They must not be interpreted as task processing times or pure queue waiting times.

## Purpose of this phase

This phase identifies process variants, transition-level elapsed-time concentration, tail delays, category heterogeneity, and H1-vs-H2 temporal drift. These are the empirical inputs for the next robust-optimization formulation.

## Phase 3 decision

The preferred next formulation is **distributionally robust process-improvement / bottleneck-intervention optimization under temporal shift**, not literal staff-capacity optimization from BPI 2019 alone.
If literal resource allocation is retained as the main claim, an additional public dataset with reliable lifecycle start/complete information should be added.
