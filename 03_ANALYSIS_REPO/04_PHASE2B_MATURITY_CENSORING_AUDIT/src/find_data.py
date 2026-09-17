from __future__ import annotations

import os
from pathlib import Path


def find_processed_pair(repo_root: Path) -> tuple[Path, Path]:
    env_events = os.environ.get("BPI2019_EVENTS_PARQUET")
    env_cases = os.environ.get("BPI2019_CASES_PARQUET")
    if env_events and env_cases:
        events = Path(env_events).expanduser()
        cases = Path(env_cases).expanduser()
        if events.exists() and cases.exists():
            return events.resolve(), cases.resolve()
    project_root = repo_root.parents[1]
    processed = project_root / "04_ANALYSIS_RESULTS" / "03_PHASE2_PROCESS_STRUCTURE" / "processed"
    events = processed / "events_primary_cohort.parquet"
    cases = processed / "cases_primary_cohort.parquet"
    if events.exists() and cases.exists():
        return events.resolve(), cases.resolve()
    raise FileNotFoundError(
        "Nu am gasit events_primary_cohort.parquet si cases_primary_cohort.parquet. "
        "Ruleaza Phase 2 sau seteaza BPI2019_EVENTS_PARQUET si BPI2019_CASES_PARQUET."
    )
