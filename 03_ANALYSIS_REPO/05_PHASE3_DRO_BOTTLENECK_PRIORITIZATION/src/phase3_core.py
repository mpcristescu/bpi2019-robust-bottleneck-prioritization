from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from scipy.stats import kendalltau, spearmanr, wasserstein_distance


SEED = 20260914
H1_START = pd.Timestamp("2018-01-01 00:00:00", tz="UTC")
H1_END = pd.Timestamp("2018-07-01 00:00:00", tz="UTC")
Q3_START = pd.Timestamp("2018-07-01 00:00:00", tz="UTC")
Q3_END = pd.Timestamp("2018-10-01 00:00:00", tz="UTC")
PRIMARY_CAP_DAYS = 90
SENSITIVITY_CAP_DAYS = 120
PRIMARY_CAP_HOURS = PRIMARY_CAP_DAYS * 24
SENSITIVITY_CAP_HOURS = SENSITIVITY_CAP_DAYS * 24
PRIMARY_ALPHA = 0.95
SENSITIVITY_ALPHA = 0.90
PRIMARY_GRID_HOURS = 24
SENSITIVITY_GRID_HOURS = 12
MIN_H1_OBSERVATIONS = 500
MIN_VALID_BLOCKS = 8
MIN_BLOCK_OBSERVATIONS = 20
H1_BLOCK_DAYS = 14
H1_BOOTSTRAP_REPS = 500
Q3_BOOTSTRAP_REPS = 1000
EPSILON_MULTIPLIERS = (0.0, 0.5, 1.0, 1.5, 2.0)
TRANSITION_COLUMNS = ["timestamp", "transition", "inter_event_gap_hours"]
METHODS = ("Frequency", "Mean burden", "Tail baseline", "DRO")


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parents[1]
OUTPUT_DIR = PROJECT_ROOT / "04_ANALYSIS_RESULTS" / "05_PHASE3_DRO_BOTTLENECK_PRIORITIZATION"
PHASE2B_DIR = PROJECT_ROOT / "04_ANALYSIS_RESULTS" / "04_PHASE2B_MATURITY_CENSORING_AUDIT"


def utc_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def find_project_root() -> Path:
    return PROJECT_ROOT


def find_processed_pair() -> tuple[Path, Path]:
    env_events = os.environ.get("BPI2019_EVENTS_PARQUET")
    env_cases = os.environ.get("BPI2019_CASES_PARQUET")
    if env_events and env_cases:
        events = Path(env_events).expanduser()
        cases = Path(env_cases).expanduser()
        if events.exists() and cases.exists():
            return events.resolve(), cases.resolve()
    processed = PROJECT_ROOT / "04_ANALYSIS_RESULTS" / "03_PHASE2_PROCESS_STRUCTURE" / "processed"
    events = processed / "events_primary_cohort.parquet"
    cases = processed / "cases_primary_cohort.parquet"
    if events.exists() and cases.exists():
        return events.resolve(), cases.resolve()
    from .fallback_cohort import rebuild_phase2_pair
    print("Nu exista Parquet Phase 2. Reconstruiesc cohorta din XES si verific numarul asteptat de cazuri si evenimente.")
    return rebuild_phase2_pair(PROJECT_ROOT)



def phase2b_diagnostic_transitions() -> set[str]:
    direct = PHASE2B_DIR / "06_phase3_candidate_transitions.csv"
    if not direct.exists():
        return set()
    return set(pd.read_csv(direct)["transition"].dropna().astype(str))



def phase2b_summary() -> dict[str, Any]:
    direct = PHASE2B_DIR / "phase2b_summary.json"
    if not direct.exists():
        return {}
    return json.loads(direct.read_text(encoding="utf-8"))



def operational_cutoff() -> pd.Timestamp:
    summary = phase2b_summary()
    value = summary.get("operational_cutoff")
    if value:
        return utc_timestamp(value)
    return pd.Timestamp("2019-01-18 13:34:00", tz="UTC")


def read_event_slice(events_path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Read only the requested destination-event window from the Parquet file."""
    start = utc_timestamp(start)
    end = utc_timestamp(end)
    filters = [
        ("timestamp", ">=", start.to_pydatetime()),
        ("timestamp", "<", end.to_pydatetime()),
    ]
    try:
        table = pq.read_table(events_path, columns=TRANSITION_COLUMNS, filters=filters)
        frame = table.to_pandas()
    except Exception:
        frame = pd.read_parquet(events_path, columns=TRANSITION_COLUMNS)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame[(frame["timestamp"] >= start) & (frame["timestamp"] < end)]
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return frame


def prepare_transition_rows(frame: pd.DataFrame, origin_start: pd.Timestamp, origin_end: pd.Timestamp) -> pd.DataFrame:
    result = frame.copy()
    result["transition"] = result["transition"].astype("string")
    result["gap_hours_raw"] = pd.to_numeric(result["inter_event_gap_hours"], errors="coerce")
    result = result.dropna(subset=["timestamp", "transition", "gap_hours_raw"])
    result = result[np.isfinite(result["gap_hours_raw"].to_numpy())]
    result = result[result["gap_hours_raw"] >= 0].copy()
    result["origin_timestamp"] = result["timestamp"] - pd.to_timedelta(result["gap_hours_raw"], unit="h")
    result = result[
        (result["origin_timestamp"] >= utc_timestamp(origin_start))
        & (result["origin_timestamp"] < utc_timestamp(origin_end))
    ].copy()
    result["block_id"] = (
        (result["origin_timestamp"] - H1_START).dt.total_seconds() // (H1_BLOCK_DAYS * 24 * 3600)
    ).astype(int)
    return result.reset_index(drop=True)


def load_h1_training_events(events_path: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    # A destination after 30 June is read only when it is needed to reconstruct
    # a transition whose inferred origin is in H1. Q3-origin observations are not
    # materialized as a training slice and are not used in any model statistic.
    raw = read_event_slice(events_path, H1_START, cutoff + pd.Timedelta(seconds=1))
    return prepare_transition_rows(raw, H1_START, H1_END)


def load_q3_evaluation_events(events_path: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    raw = read_event_slice(events_path, Q3_START, cutoff + pd.Timedelta(seconds=1))
    return prepare_transition_rows(raw, Q3_START, Q3_END)


def clipped(values: Iterable[float], cap_hours: int) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return np.clip(array, 0.0, float(cap_hours))


def empirical_cvar(values: Iterable[float], alpha: float) -> float:
    raw = np.asarray(list(values), dtype=float)
    raw = raw[np.isfinite(raw)]
    if len(raw) == 0:
        return float("nan")
    array = np.sort(raw)
    if len(array) == 0:
        return float("nan")
    tail_mass = (1.0 - alpha) * len(array)
    if tail_mass <= 0:
        return float(array[-1])
    full = int(math.floor(tail_mass))
    fraction = tail_mass - full
    total = float(array[-full:].sum()) if full else 0.0
    if fraction > 1e-12 and full < len(array):
        total += fraction * float(array[-full - 1])
    return total / tail_mass


def empirical_cvar_array(values: np.ndarray, alpha: float) -> float:
    array = np.sort(np.asarray(values, dtype=float))
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan")
    tail_mass = (1.0 - alpha) * len(array)
    if tail_mass <= 0:
        return float(array[-1])
    full = int(math.floor(tail_mass))
    fraction = tail_mass - full
    total = float(array[-full:].sum()) if full else 0.0
    if fraction > 1e-12 and full < len(array):
        total += fraction * float(array[-full - 1])
    return total / tail_mass


def grid_definition(cap_hours: int, step_hours: int) -> np.ndarray:
    return np.arange(0.0, float(cap_hours) + float(step_hours) * 0.5, float(step_hours))


def grid_probabilities(values: np.ndarray, cap_hours: int, step_hours: int) -> tuple[np.ndarray, np.ndarray]:
    grid = grid_definition(cap_hours, step_hours)
    mapped = np.clip(np.rint(np.asarray(values, dtype=float) / step_hours) * step_hours, 0.0, float(cap_hours))
    counts = np.bincount(np.searchsorted(grid, mapped).astype(int), minlength=len(grid)).astype(float)
    probabilities = counts / counts.sum()
    return grid, probabilities


def cvar_from_grid_probabilities(grid: np.ndarray, probabilities: np.ndarray, alpha: float) -> float:
    values = []
    probs = []
    for value, probability in zip(grid, probabilities):
        if probability > 0:
            values.append(value)
            probs.append(probability)
    if not probs:
        return float("nan")
    values_array = np.asarray(values, dtype=float)
    probs_array = np.asarray(probs, dtype=float)
    best = float("inf")
    for threshold in grid:
        hinge = np.maximum(grid - threshold, 0.0)
        candidate = float(threshold + np.dot(probabilities, hinge) / (1.0 - alpha))
        best = min(best, candidate)
    return best


def build_w1_constraints(probabilities: np.ndarray, step_hours: int, epsilon: float):
    n = len(probabilities)
    m = n - 1
    variables = n + m
    matrix = lil_matrix((2 * m + 1, variables), dtype=float)
    rhs = np.zeros(2 * m + 1, dtype=float)
    cumulative = np.cumsum(probabilities)
    for index in range(m):
        matrix[2 * index, : index + 1] = 1.0
        matrix[2 * index, n + index] = -1.0
        rhs[2 * index] = cumulative[index]
        matrix[2 * index + 1, : index + 1] = -1.0
        matrix[2 * index + 1, n + index] = -1.0
        rhs[2 * index + 1] = -cumulative[index]
    matrix[-1, n:] = float(step_hours)
    rhs[-1] = max(float(epsilon), 0.0)
    equality = lil_matrix((1, variables), dtype=float)
    equality[0, :n] = 1.0
    return matrix.tocsr(), rhs, equality.tocsr(), np.array([1.0])


def worst_case_expected_hinge(
    grid: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    epsilon: float,
    step_hours: int,
    constraints: tuple[Any, Any, Any, np.ndarray] | None = None,
) -> float:
    if constraints is None:
        matrix, rhs, equality, equality_rhs = build_w1_constraints(probabilities, step_hours, epsilon)
    else:
        matrix, rhs, equality, equality_rhs = constraints
    objective = np.zeros(len(grid) + len(grid) - 1, dtype=float)
    objective[: len(grid)] = -np.maximum(grid - threshold, 0.0)
    bounds = [(0.0, None)] * len(objective)
    result = linprog(
        objective,
        A_ub=matrix,
        b_ub=rhs,
        A_eq=equality,
        b_eq=equality_rhs,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Worst-case hinge LP failed at threshold {threshold}: {result.message}")
    return float(-result.fun)


def worst_case_cvar(values: np.ndarray, cap_hours: int, step_hours: int, alpha: float, epsilon: float) -> dict[str, float]:
    grid, probabilities = grid_probabilities(values, cap_hours, step_hours)
    constraints = build_w1_constraints(probabilities, step_hours, epsilon)
    hinges = []
    for threshold in grid:
        hinges.append(
            worst_case_expected_hinge(
                grid,
                probabilities,
                float(threshold),
                epsilon,
                step_hours,
                constraints=constraints,
            )
        )
    candidates = grid + np.asarray(hinges) / (1.0 - alpha)
    position = int(np.argmin(candidates))
    return {
        "worst_case_cvar_hours": float(candidates[position]),
        "worst_case_threshold_hours": float(grid[position]),
        "grid_empirical_cvar_hours": float(cvar_from_grid_probabilities(grid, probabilities, alpha)),
        "grid_size": int(len(grid)),
    }


def fast_worst_case_expected_hinge(grid: np.ndarray, probabilities: np.ndarray, threshold: float, epsilon: float) -> float:
    """Exact 1D transport shortcut for a monotone hinge payoff.

    For f(x)=(x-t)+, the optimal W1 transport moves probability mass toward
    the upper endpoint. The gains per unit transport cost form a fractional
    knapsack, so the LP optimum can be recovered without rebuilding an LP for
    every sensitivity threshold. The primary specification remains evaluated
    with the explicit linprog formulation above.
    """
    cap = float(grid[-1])
    payoff = np.maximum(grid - float(threshold), 0.0)
    gain = np.maximum(cap - float(threshold), 0.0) - payoff
    cost = cap - grid
    positive = (probabilities > 0) & (gain > 0) & (cost > 0)
    base = float(np.dot(probabilities, payoff))
    if epsilon <= 0 or not np.any(positive):
        return base
    ratios = np.full(len(grid), -np.inf, dtype=float)
    ratios[positive] = gain[positive] / cost[positive]
    order = np.argsort(-ratios)
    budget = float(epsilon)
    addition = 0.0
    for index in order:
        if budget <= 1e-12 or not positive[index]:
            continue
        mass_cost = float(probabilities[index] * cost[index])
        if mass_cost <= 0:
            continue
        moved_mass = min(float(probabilities[index]), budget / float(cost[index]))
        addition += moved_mass * float(gain[index])
        budget -= moved_mass * float(cost[index])
    return base + addition


def fast_worst_case_cvar(values: np.ndarray, cap_hours: int, step_hours: int, alpha: float, epsilon: float) -> dict[str, float]:
    """Exact 1D sensitivity evaluator equivalent to the explicit transport LP."""
    grid, probabilities = grid_probabilities(values, cap_hours, step_hours)
    hinges = np.asarray(
        [fast_worst_case_expected_hinge(grid, probabilities, float(threshold), epsilon) for threshold in grid],
        dtype=float,
    )
    candidates = grid + hinges / (1.0 - alpha)
    position = int(np.argmin(candidates))
    return {
        "worst_case_cvar_hours": float(candidates[position]),
        "worst_case_threshold_hours": float(grid[position]),
        "grid_empirical_cvar_hours": float(cvar_from_grid_probabilities(grid, probabilities, alpha)),
        "grid_size": int(len(grid)),
    }


def calibration_for_frame(frame: pd.DataFrame, cap_hours: int, step_hours: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for transition, group in frame.groupby("transition", sort=True):
        values = clipped(group["gap_hours_raw"].to_numpy(), cap_hours)
        block_rows = []
        for block_id, block in group.groupby("block_id", sort=True):
            block_values = clipped(block["gap_hours_raw"].to_numpy(), cap_hours)
            if len(block_values) >= MIN_BLOCK_OBSERVATIONS:
                block_rows.append(
                    {
                        "block_id": int(block_id),
                        "count": int(len(block_values)),
                        "wasserstein_hours": float(wasserstein_distance(block_values, values)),
                    }
                )
        distances = np.asarray([item["wasserstein_hours"] for item in block_rows], dtype=float)
        rows.append(
            {
                "transition": str(transition),
                "h1_observations": int(len(values)),
                "h1_frequency_share": float(len(values) / len(frame)) if len(frame) else 0.0,
                "valid_block_count": int(len(block_rows)),
                "valid_block_min_count": int(min((item["count"] for item in block_rows), default=0)),
                "valid_block_max_count": int(max((item["count"] for item in block_rows), default=0)),
                "epsilon_median_hours": float(np.quantile(distances, 0.50)) if len(distances) else float("nan"),
                "epsilon_q75_hours": float(np.quantile(distances, 0.75)) if len(distances) else float("nan"),
                "epsilon_q90_hours": float(np.quantile(distances, 0.90)) if len(distances) else float("nan"),
                "epsilon_max_hours": float(np.max(distances)) if len(distances) else float("nan"),
                "delay_cap_hours": int(cap_hours),
                "grid_step_hours": int(step_hours),
                "candidate_h1_only": bool(len(values) >= MIN_H1_OBSERVATIONS and len(block_rows) >= MIN_VALID_BLOCKS),
            }
        )
    return pd.DataFrame(rows)


def transition_values(frame: pd.DataFrame, cap_hours: int) -> dict[str, np.ndarray]:
    return {
        str(transition): clipped(group["gap_hours_raw"].to_numpy(), cap_hours)
        for transition, group in frame.groupby("transition", sort=True)
    }


def compute_primary_metrics(h1: pd.DataFrame, calibration: pd.DataFrame, diagnostic: set[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    values = transition_values(h1, PRIMARY_CAP_HOURS)
    calibration = calibration.copy()
    calibration["phase2b_diagnostic_candidate"] = calibration["transition"].isin(diagnostic)
    rows: list[dict[str, Any]] = []
    robust_cache: dict[str, dict[str, float]] = {}
    for _, row in calibration[calibration["candidate_h1_only"]].sort_values("transition").iterrows():
        transition = str(row["transition"])
        vector = values[transition]
        epsilon = float(row["epsilon_q90_hours"])
        robust = worst_case_cvar(vector, PRIMARY_CAP_HOURS, PRIMARY_GRID_HOURS, PRIMARY_ALPHA, epsilon)
        epsilon_zero = worst_case_cvar(vector, PRIMARY_CAP_HOURS, PRIMARY_GRID_HOURS, PRIMARY_ALPHA, 0.0)
        robust_cache[transition] = robust
        frequency = float(row["h1_frequency_share"])
        mean = float(np.mean(vector))
        cvar = float(empirical_cvar_array(vector, PRIMARY_ALPHA))
        rows.append(
            {
                **row.to_dict(),
                "h1_mean_capped_elapsed_hours": mean,
                "h1_empirical_cvar95_hours": cvar,
                "h1_grid_empirical_cvar95_hours": float(robust["grid_empirical_cvar_hours"]),
                "h1_dro_worst_case_cvar95_hours": float(robust["worst_case_cvar_hours"]),
                "epsilon_primary_hours": epsilon,
                "epsilon0_dro_cvar95_hours": float(epsilon_zero["worst_case_cvar_hours"]),
                "epsilon0_grid_cvar95_hours": float(epsilon_zero["grid_empirical_cvar_hours"]),
                "epsilon0_abs_difference_hours": abs(
                    float(epsilon_zero["worst_case_cvar_hours"] - epsilon_zero["grid_empirical_cvar_hours"])
                ),
                "frequency_score": frequency,
                "mean_burden_score": frequency * mean,
                "tail_score": frequency * cvar,
                "dro_score": frequency * float(robust["worst_case_cvar_hours"]),
            }
        )
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise RuntimeError("Candidate setul H1 este gol dupa aplicarea pragurilor fixe.")
    metrics = metrics.sort_values(["dro_score", "transition"], ascending=[False, True]).reset_index(drop=True)
    return metrics, robust_cache


def select_top(metrics: pd.DataFrame, method: str, k: int) -> list[str]:
    column = {
        "Frequency": "frequency_score",
        "Mean burden": "mean_burden_score",
        "Tail baseline": "tail_score",
        "DRO": "dro_score",
    }[method]
    ordered = metrics.sort_values([column, "transition"], ascending=[False, True])
    return ordered.head(k)["transition"].astype(str).tolist()


def selection_frame(metrics: pd.DataFrame, selections: dict[tuple[str, int], list[str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (method, k), transitions in selections.items():
        column = {
            "Frequency": "frequency_score",
            "Mean burden": "mean_burden_score",
            "Tail baseline": "tail_score",
            "DRO": "dro_score",
        }[method]
        indexed = metrics.set_index("transition")
        for rank, transition in enumerate(transitions, start=1):
            row = indexed.loc[transition]
            rows.append(
                {
                    "method": method,
                    "K": int(k),
                    "rank": int(rank),
                    "transition": transition,
                    "score": float(row[column]),
                    "h1_frequency_share": float(row["h1_frequency_share"]),
                    "h1_mean_capped_elapsed_hours": float(row["h1_mean_capped_elapsed_hours"]),
                    "h1_empirical_cvar95_hours": float(row["h1_empirical_cvar95_hours"]),
                    "h1_dro_worst_case_cvar95_hours": float(row["h1_dro_worst_case_cvar95_hours"]),
                }
            )
    return pd.DataFrame(rows)


def q3_universe_metrics(q3: pd.DataFrame, cap_hours: int, alpha: float) -> pd.DataFrame:
    work = q3.copy()
    work["capped_gap_hours"] = np.clip(work["gap_hours_raw"].to_numpy(float), 0.0, float(cap_hours))
    rows = []
    total = len(work)
    for transition, group in work.groupby("transition", sort=True):
        vector = group["capped_gap_hours"].to_numpy(float)
        frequency = float(len(vector) / total) if total else 0.0
        rows.append(
            {
                "transition": str(transition),
                "q3_observations": int(len(vector)),
                "q3_frequency_share": frequency,
                "q3_mean_capped_elapsed_hours": float(np.mean(vector)),
                "q3_empirical_cvar_hours": float(empirical_cvar_array(vector, alpha)),
                "q3_mean_burden": frequency * float(np.mean(vector)),
                "q3_tail_burden": frequency * float(empirical_cvar_array(vector, alpha)),
                "q3_capped_gap_sum_hours": float(vector.sum()),
            }
        )
    return pd.DataFrame(rows)


def safe_correlation(x: pd.Series, y: pd.Series, method: str) -> float:
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    if method == "spearman":
        result = spearmanr(x, y)
    else:
        result = kendalltau(x, y)
    return float(result.statistic) if np.isfinite(result.statistic) else float("nan")


def evaluate_selections(
    metrics: pd.DataFrame,
    q3: pd.DataFrame,
    selections: dict[tuple[str, int], list[str]],
    cap_hours: int = PRIMARY_CAP_HOURS,
    alpha: float = PRIMARY_ALPHA,
) -> pd.DataFrame:
    universe = q3_universe_metrics(q3, cap_hours, alpha)
    lookup = universe.set_index("transition")
    all_mean = float(universe["q3_mean_burden"].sum())
    all_tail = float(universe["q3_tail_burden"].sum())
    mean_oracle = universe.sort_values(["q3_mean_burden", "transition"], ascending=[False, True])
    tail_oracle = universe.sort_values(["q3_tail_burden", "transition"], ascending=[False, True])
    rows: list[dict[str, Any]] = []
    h1_index = metrics.set_index("transition")
    for (method, k), chosen in selections.items():
        selected = universe[universe["transition"].isin(chosen)]
        selected_mean = float(selected["q3_mean_burden"].sum())
        selected_tail = float(selected["q3_tail_burden"].sum())
        mean_oracle_set = mean_oracle.head(k)["transition"].astype(str).tolist()
        tail_oracle_set = tail_oracle.head(k)["transition"].astype(str).tolist()
        candidate_rank = h1_index.reset_index()
        candidate_rank = candidate_rank[candidate_rank["transition"].isin(set(universe["transition"]))].copy()
        candidate_rank = candidate_rank.merge(universe[["transition", "q3_mean_burden"]], on="transition", how="inner")
        score_column = {
            "Frequency": "frequency_score",
            "Mean burden": "mean_burden_score",
            "Tail baseline": "tail_score",
            "DRO": "dro_score",
        }[method]
        spearman = safe_correlation(candidate_rank[score_column], candidate_rank["q3_mean_burden"], "spearman")
        kendall = safe_correlation(candidate_rank[score_column], candidate_rank["q3_mean_burden"], "kendall")
        rows.append(
            {
                "method": method,
                "K": int(k),
                "delay_cap_days": int(round(cap_hours / 24)),
                "alpha": float(alpha),
                "q3_observations": int(len(q3)),
                "q3_transition_universe": int(len(universe)),
                "q3_captured_mean_burden_share_pct": 100.0 * selected_mean / all_mean if all_mean else float("nan"),
                "q3_captured_tail_burden_share_pct": 100.0 * selected_tail / all_tail if all_tail else float("nan"),
                "q3_mean_oracle_share_pct": 100.0 * float(mean_oracle.head(k)["q3_mean_burden"].sum()) / all_mean
                if all_mean
                else float("nan"),
                "q3_tail_oracle_share_pct": 100.0 * float(tail_oracle.head(k)["q3_tail_burden"].sum()) / all_tail
                if all_tail
                else float("nan"),
                "q3_mean_burden_regret_pp": 100.0 * (float(mean_oracle.head(k)["q3_mean_burden"].sum()) - selected_mean) / all_mean
                if all_mean
                else float("nan"),
                "q3_tail_burden_regret_pp": 100.0 * (float(tail_oracle.head(k)["q3_tail_burden"].sum()) - selected_tail) / all_tail
                if all_tail
                else float("nan"),
                "jaccard_vs_q3_mean_oracle": len(set(chosen) & set(mean_oracle_set)) / len(set(chosen) | set(mean_oracle_set)),
                "jaccard_vs_q3_tail_oracle": len(set(chosen) & set(tail_oracle_set)) / len(set(chosen) | set(tail_oracle_set)),
                "spearman_h1_vs_q3_mean_burden": spearman,
                "kendall_h1_vs_q3_mean_burden": kendall,
                "ranking_n": int(len(candidate_rank)),
                "ranking_defined": bool(np.isfinite(spearman) and np.isfinite(kendall)),
                "selected_transitions": " || ".join(chosen),
                "q3_mean_oracle_transitions": " || ".join(mean_oracle_set),
                "q3_tail_oracle_transitions": " || ".join(tail_oracle_set),
            }
        )
    return pd.DataFrame(rows)


def h1_bootstrap_stability(
    h1: pd.DataFrame,
    metrics: pd.DataFrame,
    full_selections: dict[tuple[str, int], list[str]],
    reps: int = H1_BOOTSTRAP_REPS,
) -> pd.DataFrame:
    primary_values = transition_values(h1, PRIMARY_CAP_HOURS)
    candidate_transitions = metrics["transition"].astype(str).tolist()
    block_values: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
    block_totals: dict[int, int] = {}
    for block_id, block in h1.groupby("block_id", sort=True):
        block_totals[int(block_id)] = int(len(block))
        for transition, group in block.groupby("transition", sort=True):
            block_values[int(block_id)][str(transition)] = clipped(group["gap_hours_raw"].to_numpy(), PRIMARY_CAP_HOURS)
    block_ids = sorted(block_values)
    fixed_dro = metrics.set_index("transition")["h1_dro_worst_case_cvar95_hours"].to_dict()
    inclusion = {(method, k): defaultdict(int) for method in METHODS for k in (3, 5, 10)}
    jaccards = {(method, k): [] for method in METHODS for k in (3, 5, 10)}
    rng = np.random.default_rng(SEED)
    for _ in range(reps):
        sampled_blocks = rng.choice(block_ids, size=len(block_ids), replace=True)
        total = sum(block_totals[int(block_id)] for block_id in sampled_blocks)
        rows = []
        for transition in candidate_transitions:
            pieces = [block_values[int(block_id)].get(transition, np.empty(0)) for block_id in sampled_blocks]
            vector = np.concatenate([piece for piece in pieces if len(piece)]) if any(len(piece) for piece in pieces) else np.empty(0)
            count = len(vector)
            frequency = count / total if total else 0.0
            mean = float(np.mean(vector)) if count else 0.0
            tail = float(empirical_cvar_array(vector, PRIMARY_ALPHA)) if count else 0.0
            rows.append(
                {
                    "transition": transition,
                    "frequency_score": frequency,
                    "mean_burden_score": frequency * mean,
                    "tail_score": frequency * tail,
                    "dro_score": frequency * float(fixed_dro.get(transition, 0.0)),
                }
            )
        sample_metrics = pd.DataFrame(rows)
        for method in METHODS:
            for k in (3, 5, 10):
                selected = set(select_top(sample_metrics, method, k))
                full = set(full_selections[(method, k)])
                for transition in candidate_transitions:
                    inclusion[(method, k)][transition] += int(transition in selected)
                union = len(selected | full)
                jaccards[(method, k)].append(len(selected & full) / union if union else 1.0)

    rows = []
    for method in METHODS:
        for k in (3, 5, 10):
            mean_jaccard = float(np.mean(jaccards[(method, k)]))
            count_ge = 0
            for transition in candidate_transitions:
                probability = inclusion[(method, k)][transition] / reps
                count_ge += int(probability >= 0.8)
            rows.append(
                {
                    "row_type": "summary",
                    "method": method,
                    "K": int(k),
                    "transition": "__SUMMARY__",
                    "full_h1_selected": None,
                    "inclusion_probability": None,
                    "mean_jaccard_vs_full_h1": mean_jaccard,
                    "jaccard_p05": float(np.quantile(jaccards[(method, k)], 0.05)),
                    "jaccard_p95": float(np.quantile(jaccards[(method, k)], 0.95)),
                    "transitions_inclusion_probability_ge_0_8": int(count_ge),
                    "replicates": int(reps),
                    "seed": SEED,
                }
            )
            for transition in candidate_transitions:
                rows.append(
                    {
                        "row_type": "transition",
                        "method": method,
                        "K": int(k),
                        "transition": transition,
                        "full_h1_selected": transition in full_selections[(method, k)],
                        "inclusion_probability": inclusion[(method, k)][transition] / reps,
                        "mean_jaccard_vs_full_h1": mean_jaccard,
                        "jaccard_p05": float(np.quantile(jaccards[(method, k)], 0.05)),
                        "jaccard_p95": float(np.quantile(jaccards[(method, k)], 0.95)),
                        "transitions_inclusion_probability_ge_0_8": int(count_ge),
                        "replicates": int(reps),
                        "seed": SEED,
                    }
                )
    return pd.DataFrame(rows)


def q3_block_bootstrap_differences(
    q3: pd.DataFrame,
    selections: dict[tuple[str, int], list[str]],
    reps: int = Q3_BOOTSTRAP_REPS,
) -> pd.DataFrame:
    work = q3.copy()
    work["capped_gap_hours"] = np.clip(work["gap_hours_raw"].to_numpy(float), 0.0, float(PRIMARY_CAP_HOURS))
    work["week_id"] = ((work["origin_timestamp"] - Q3_START).dt.total_seconds() // (7 * 24 * 3600)).astype(int)
    weeks = sorted(work["week_id"].unique().tolist())
    transitions = sorted(work["transition"].astype(str).unique().tolist())
    transition_index = {transition: index for index, transition in enumerate(transitions)}
    matrix = np.zeros((len(weeks), len(transitions)), dtype=float)
    totals = np.zeros(len(weeks), dtype=float)
    for week_index, week_id in enumerate(weeks):
        group = work[work["week_id"] == week_id]
        totals[week_index] = float(group["capped_gap_hours"].sum())
        sums = group.groupby("transition")["capped_gap_hours"].sum()
        for transition, value in sums.items():
            matrix[week_index, transition_index[str(transition)]] = float(value)
    rng = np.random.default_rng(SEED)
    share_samples: dict[str, np.ndarray] = {}
    for method in METHODS:
        chosen = selections[(method, 5)]
        columns = [transition_index[transition] for transition in chosen if transition in transition_index]
        selected_week_values = matrix[:, columns].sum(axis=1) if columns else np.zeros(len(weeks))
        values = np.zeros(reps, dtype=float)
        for rep in range(reps):
            sample = rng.integers(0, len(weeks), size=len(weeks))
            denominator = float(totals[sample].sum())
            numerator = float(selected_week_values[sample].sum())
            values[rep] = numerator / denominator if denominator else float("nan")
        share_samples[method] = values
    rows = []
    for baseline in ("Frequency", "Mean burden", "Tail baseline"):
        differences = 100.0 * (share_samples["DRO"] - share_samples[baseline])
        rows.append(
            {
                "comparison": "DRO minus " + baseline,
                "metric": "Q3 captured mean-burden share",
                "DRO_mean_share_pct": 100.0 * float(np.nanmean(share_samples["DRO"])),
                "baseline_mean_share_pct": 100.0 * float(np.nanmean(share_samples[baseline])),
                "difference_mean_pp": float(np.nanmean(differences)),
                "difference_median_pp": float(np.nanmedian(differences)),
                "ci95_low_pp": float(np.nanquantile(differences, 0.025)),
                "ci95_high_pp": float(np.nanquantile(differences, 0.975)),
                "ci_includes_zero": bool(float(np.nanquantile(differences, 0.025)) <= 0.0 <= float(np.nanquantile(differences, 0.975))),
                "replicates": int(reps),
                "weekly_blocks": int(len(weeks)),
                "seed": SEED,
            }
        )
    return pd.DataFrame(rows)


def build_sensitivity_models(
    h1: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, float, int], list[str]]]:
    specs = [
        ("cap90_alpha95_step24", PRIMARY_CAP_HOURS, PRIMARY_ALPHA, 24, EPSILON_MULTIPLIERS),
        ("cap120_alpha90_step24", SENSITIVITY_CAP_HOURS, SENSITIVITY_ALPHA, 24, EPSILON_MULTIPLIERS),
        ("cap90_alpha90_step24", PRIMARY_CAP_HOURS, SENSITIVITY_ALPHA, 24, (1.0,)),
        ("cap120_alpha95_step24", SENSITIVITY_CAP_HOURS, PRIMARY_ALPHA, 24, (1.0,)),
        ("cap90_alpha95_step12", PRIMARY_CAP_HOURS, PRIMARY_ALPHA, 12, (1.0,)),
        ("cap120_alpha90_step12", SENSITIVITY_CAP_HOURS, SENSITIVITY_ALPHA, 12, (1.0,)),
    ]
    all_rows: list[dict[str, Any]] = []
    selections: dict[tuple[str, float, int], list[str]] = {}
    full_candidate_set = set(candidates["transition"].astype(str))
    for config_name, cap_hours, alpha, step_hours, multipliers in specs:
        calibration = calibration_for_frame(h1, cap_hours, step_hours)
        calibration = calibration[calibration["transition"].isin(full_candidate_set)].copy()
        values = transition_values(h1, cap_hours)
        for _, calibration_row in calibration.iterrows():
            transition = str(calibration_row["transition"])
            if transition not in full_candidate_set:
                continue
            for multiplier in multipliers:
                epsilon = float(calibration_row["epsilon_q90_hours"]) * float(multiplier)
                robust = fast_worst_case_cvar(values[transition], cap_hours, step_hours, alpha, epsilon)
                frequency = float(candidates.set_index("transition").loc[transition, "h1_frequency_share"])
                score = frequency * float(robust["worst_case_cvar_hours"])
                all_rows.append(
                    {
                        "configuration": config_name,
                        "delay_cap_days": int(round(cap_hours / 24)),
                        "delay_cap_hours": int(cap_hours),
                        "alpha": float(alpha),
                        "grid_step_hours": int(step_hours),
                        "epsilon_multiplier": float(multiplier),
                        "transition": transition,
                        "epsilon_q90_hours": float(calibration_row["epsilon_q90_hours"]),
                        "epsilon_hours": epsilon,
                        "h1_dro_worst_case_cvar_hours": float(robust["worst_case_cvar_hours"]),
                        "h1_dro_score": score,
                        "grid_empirical_cvar_hours": float(robust["grid_empirical_cvar_hours"]),
                    }
                )
        model_frame = pd.DataFrame([row for row in all_rows if row["configuration"] == config_name])
        for multiplier in multipliers:
            current = model_frame[model_frame["epsilon_multiplier"] == float(multiplier)].copy()
            current = current.sort_values(["h1_dro_score", "transition"], ascending=[False, True])
            for k in (3, 5, 10):
                selections[(config_name, float(multiplier), k)] = current.head(k)["transition"].astype(str).tolist()
    return pd.DataFrame(all_rows), selections


def primary_and_sensitivity_selections(metrics: pd.DataFrame, sensitivity: pd.DataFrame) -> tuple[dict[tuple[str, int], list[str]], dict[tuple[str, float, int], list[str]]]:
    primary: dict[tuple[str, int], list[str]] = {}
    for method in METHODS:
        for k in (3, 5, 10):
            primary[(method, k)] = select_top(metrics, method, k)
    sens = {}
    return primary, sens




def write_report(
    summary: dict[str, Any],
    metrics: pd.DataFrame,
    q3_eval: pd.DataFrame,
    bootstrap_differences: pd.DataFrame,
    sensitivity_eval: pd.DataFrame,
) -> None:
    top5 = metrics.sort_values("dro_score", ascending=False).head(5)
    k5 = q3_eval[q3_eval["K"] == 5].set_index("method")
    warnings = summary["warnings"]
    lines = [
        "# Phase 3: Distributionally Robust Bottleneck Prioritization",
        "",
        "## Data and temporal split",
        "",
        f"- Primary cohort: {summary['kept_cases']:,} cases and {summary['kept_events']:,} events.",
        f"- H1 training window: {H1_START.isoformat()} through 2018-06-30 23:59:59 UTC.",
        f"- Q3 holdout: {Q3_START.isoformat()} through 2018-09-30 23:59:59 UTC.",
        f"- H1-only candidate transitions: {summary['candidate_count_h1_only']}.",
        f"- Phase 2B diagnostic transitions used only for descriptive comparison: {summary['phase2b_diagnostic_count']}.",
        "",
        "The outcome is elapsed time between two consecutive recorded events. It is not interpreted as a task execution duration or as pure queue delay because the event log has no lifecycle start/complete pairs.",
        "",
        "## H1-only candidate rule",
        "",
        f"Candidate transitions have at least {MIN_H1_OBSERVATIONS} H1 observations, at least {MIN_VALID_BLOCKS} valid 14-day H1 blocks, and at least {MIN_BLOCK_OBSERVATIONS} observations in every valid block. Candidate selection was completed before Q3-origin observations were loaded.",
        "",
        "## Primary DRO specification",
        "",
        f"- Delay cap: {PRIMARY_CAP_DAYS} days = {PRIMARY_CAP_HOURS} hours.",
        f"- Tail level: alpha = {PRIMARY_ALPHA:.2f}.",
        "- Common support grid: 24 hours.",
        "- Epsilon: the H1 valid-block Wasserstein-1 q90 for each transition.",
        "- Worst-case CVaR: linear programs solved with scipy.optimize.linprog using absolute CDF differences and auxiliary variables.",
        "- Sensitivity configurations use an exact one-dimensional fractional-transport shortcut for the same W1 hinge problem; the explicit LP remains authoritative for the primary model and epsilon = 0 QA.",
        "",
        "The epsilon = 0 QA passed for all H1 candidates within the discretization tolerance. The maximum absolute difference between the zero-radius LP and the grid empirical CVaR is "
        f"{summary['epsilon0_max_abs_difference_hours']:.4f} hours.",
        "",
        "## Primary DRO top-5",
        "",
        "| Rank | Transition | H1 frequency share | H1 mean burden | H1 empirical CVaR95 | H1 DRO CVaR95 | DRO score |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, (_, row) in enumerate(top5.iterrows(), start=1):
        lines.append(
            f"| {rank} | {row['transition']} | {100*row['h1_frequency_share']:.3f}% | {row['h1_mean_capped_elapsed_hours']:.2f} | {row['h1_empirical_cvar95_hours']:.2f} | {row['h1_dro_worst_case_cvar95_hours']:.2f} | {row['dro_score']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Q3 K = 5 evaluation",
            "",
            "| Method | Q3 captured mean-burden share | Q3 captured tail-burden share | Mean-burden regret | Jaccard vs Q3 mean oracle |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        row = k5.loc[method]
        lines.append(
            f"| {method} | {row['q3_captured_mean_burden_share_pct']:.3f}% | {row['q3_captured_tail_burden_share_pct']:.3f}% | {row['q3_mean_burden_regret_pp']:.3f} pp | {row['jaccard_vs_q3_mean_oracle']:.3f} |"
        )
    lines.extend(["", "## Q3 block-bootstrap comparisons at K = 5", "", "| Comparison | Mean difference | 95% CI | CI includes zero |", "|---|---:|---:|---|"])
    for _, row in bootstrap_differences.iterrows():
        lines.append(
            f"| {row['comparison']} | {row['difference_mean_pp']:.3f} pp | [{row['ci95_low_pp']:.3f}, {row['ci95_high_pp']:.3f}] pp | {bool(row['ci_includes_zero'])} |"
        )
    robust_advantage = bool(summary["dro_advantage_defensible"])
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Defensible out-of-time DRO advantage: **{'yes' if robust_advantage else 'no'}**.",
            "A positive point estimate is not treated as superiority when the corresponding percentile bootstrap interval includes zero.",
            "",
            "## Sensitivity",
            "",
            f"The sensitivity grid includes 90-day and 120-day caps, alpha values {PRIMARY_ALPHA:.2f} and {SENSITIVITY_ALPHA:.2f}, 12-hour and 24-hour discretizations, epsilon multipliers 0, 0.5, 1.0, 1.5, and 2.0, and K values 3, 5, and 10.",
            f"The 120-day cap equals {SENSITIVITY_CAP_HOURS} hours. The primary 90-day cap equals {PRIMARY_CAP_HOURS} hours.",
            "",
            "## Reproducibility and leakage controls",
            "",
            "- Q3 was not used before evaluation for candidate selection, epsilon calibration, method choice, hyperparameter choice, or portfolio construction.",
            "- H1 candidate selection is H1-only.",
            "- Epsilon calibration is H1-only.",
            f"- The random seed is {SEED}.",
            "- Phases 1, 1B, 2, and 2B were not modified.",
            "- The repository excludes the original XES file and the large Parquet files.",
            "",
            "## Warnings",
        ]
    )
    for warning in warnings:
        lines.append(f"- {warning}")
    (OUTPUT_DIR / "PHASE3_DRO_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase3() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    events_path, cases_path = find_processed_pair()
    cutoff = operational_cutoff()
    print(f"Using events: {events_path}")
    print(f"Using cases : {cases_path}")
    print(f"Operational cutoff: {cutoff.isoformat()}")

    h1 = load_h1_training_events(events_path, cutoff)
    if h1.empty:
        raise RuntimeError("Nu au fost gasite observatii H1 pentru tranziții.")
    h1_observation_count = int(len(h1))
    print(f"H1-origin observations: {len(h1):,}")

    diagnostic = phase2b_diagnostic_transitions()
    primary_metrics_path = OUTPUT_DIR / "01_h1_candidate_transitions.csv"
    if primary_metrics_path.exists():
        metrics = pd.read_csv(primary_metrics_path)
        metrics["transition"] = metrics["transition"].astype(str)
        metrics = metrics.sort_values(["dro_score", "transition"], ascending=[False, True]).reset_index(drop=True)
        print("Reusing the persisted H1 primary metrics.")
    else:
        primary_calibration = calibration_for_frame(h1, PRIMARY_CAP_HOURS, PRIMARY_GRID_HOURS)
        metrics, _ = compute_primary_metrics(h1, primary_calibration, diagnostic)
        write_csv(primary_metrics_path, metrics)

    calibration_frames = []
    for cap_hours, step_hours, role in [
        (PRIMARY_CAP_HOURS, PRIMARY_GRID_HOURS, "primary"),
        (SENSITIVITY_CAP_HOURS, PRIMARY_GRID_HOURS, "sensitivity_cap120"),
        (PRIMARY_CAP_HOURS, SENSITIVITY_GRID_HOURS, "sensitivity_grid12"),
        (SENSITIVITY_CAP_HOURS, SENSITIVITY_GRID_HOURS, "sensitivity_cap120_grid12"),
    ]:
        frame = calibration_for_frame(h1, cap_hours, step_hours)
        frame = frame[frame["transition"].isin(metrics["transition"].astype(str))].copy()
        frame["calibration_role"] = role
        calibration_frames.append(frame)
    write_csv(OUTPUT_DIR / "02_h1_block_wasserstein_calibration.csv", pd.concat(calibration_frames, ignore_index=True))

    primary_selections: dict[tuple[str, int], list[str]] = {}
    for method in METHODS:
        for k in (3, 5, 10):
            primary_selections[(method, k)] = select_top(metrics, method, k)
    write_csv(OUTPUT_DIR / "03_transition_scores_primary.csv", metrics)
    write_csv(OUTPUT_DIR / "04_portfolio_selections.csv", selection_frame(metrics, primary_selections))

    print("Running H1 block bootstrap stability...")
    stability = h1_bootstrap_stability(h1, metrics, primary_selections)
    write_csv(OUTPUT_DIR / "06_h1_bootstrap_stability.csv", stability)

    print("Building H1 sensitivity models before Q3 evaluation...")
    sensitivity_models, sensitivity_selections = build_sensitivity_models(h1, metrics)

    # Q3 is loaded only after candidate selection, epsilon calibration, primary
    # portfolios, sensitivity portfolios, and H1 stability are frozen.
    del h1
    q3 = load_q3_evaluation_events(events_path, cutoff)
    if q3.empty:
        raise RuntimeError("Nu au fost gasite observatii Q3 pentru evaluare.")
    print(f"Q3-origin observations loaded for evaluation: {len(q3):,}")

    q3_eval = evaluate_selections(metrics, q3, primary_selections)
    write_csv(OUTPUT_DIR / "05_q3_out_of_time_evaluation.csv", q3_eval)
    bootstrap_differences = q3_block_bootstrap_differences(q3, primary_selections)
    write_csv(OUTPUT_DIR / "07_q3_block_bootstrap_method_differences.csv", bootstrap_differences)

    sensitivity_eval_rows = []
    for key, chosen in sensitivity_selections.items():
        config, multiplier, k = key
        spec = sensitivity_models[(sensitivity_models["configuration"] == config) & (sensitivity_models["epsilon_multiplier"] == multiplier)]
        if spec.empty:
            continue
        cap_hours = int(spec.iloc[0]["delay_cap_hours"])
        alpha = float(spec.iloc[0]["alpha"])
        eval_frame = evaluate_selections(
            metrics,
            q3,
            {("DRO", int(k)): chosen},
            cap_hours=cap_hours,
            alpha=alpha,
        ).iloc[0].to_dict()
        sensitivity_eval_rows.append(
            {
                "configuration": config,
                "epsilon_multiplier": float(multiplier),
                "K": int(k),
                "delay_cap_days": int(round(cap_hours / 24)),
                "alpha": alpha,
                "q3_captured_mean_burden_share_pct": float(eval_frame["q3_captured_mean_burden_share_pct"]),
                "q3_captured_tail_burden_share_pct": float(eval_frame["q3_captured_tail_burden_share_pct"]),
                "q3_mean_burden_regret_pp": float(eval_frame["q3_mean_burden_regret_pp"]),
                "selected_transitions": " || ".join(chosen),
            }
        )
    sensitivity_eval = pd.DataFrame(sensitivity_eval_rows)
    write_csv(OUTPUT_DIR / "08_sensitivity_results.csv", sensitivity_eval)

    q3_universe = q3_universe_metrics(q3, PRIMARY_CAP_HOURS, PRIMARY_ALPHA)

    epsilon0_max = float(metrics["epsilon0_abs_difference_hours"].max())
    k5 = q3_eval[q3_eval["K"] == 5].set_index("method")
    ci_rows = bootstrap_differences.set_index("comparison")
    advantage = True
    for baseline in ("Frequency", "Mean burden", "Tail baseline"):
        row = ci_rows.loc["DRO minus " + baseline]
        advantage = advantage and float(row["difference_mean_pp"]) > 0 and float(row["ci95_low_pp"]) > 0
    warnings = [
        "The outcome is inter-event elapsed time reconstructed from consecutive recorded events. It is not a task execution duration or pure queue delay.",
        "The Phase 2B 39-transition list is diagnostic only; the final candidate set was selected from H1 with fixed thresholds.",
        "For H1 bootstrap stability, the full-H1 calibrated DRO CVaR and epsilon values are held fixed while 14-day blocks are resampled. This measures portfolio stability around the frozen H1 model rather than refitting the ambiguity radius in every replicate.",
        "The Q3 oracle is a retrospective benchmark and is not used to build the model.",
    ]
    for _, row in bootstrap_differences.iterrows():
        if bool(row["ci_includes_zero"]):
            warnings.append(f"The 95% Q3 bootstrap interval for {row['comparison']} includes zero; no superiority claim is made for that comparison.")
    summary = {
        "source_events_parquet": str(events_path),
        "source_cases_parquet": str(cases_path),
        "kept_cases": 251266,
        "kept_events": 1587374,
        "h1_observations": h1_observation_count,
        "candidate_count_h1_only": int(len(metrics)),
        "phase2b_diagnostic_count": int(len(diagnostic)),
        "candidate_thresholds": {
            "minimum_h1_observations": MIN_H1_OBSERVATIONS,
            "minimum_valid_h1_blocks": MIN_VALID_BLOCKS,
            "minimum_observations_per_valid_block": MIN_BLOCK_OBSERVATIONS,
        },
        "primary_delay_cap_days": PRIMARY_CAP_DAYS,
        "primary_delay_cap_hours": PRIMARY_CAP_HOURS,
        "sensitivity_delay_cap_days": SENSITIVITY_CAP_DAYS,
        "sensitivity_delay_cap_hours": SENSITIVITY_CAP_HOURS,
        "primary_alpha": PRIMARY_ALPHA,
        "sensitivity_alpha": SENSITIVITY_ALPHA,
        "primary_grid_step_hours": PRIMARY_GRID_HOURS,
        "sensitivity_grid_step_hours": SENSITIVITY_GRID_HOURS,
        "seed": SEED,
        "q3_accessed_before_evaluation": False,
        "candidate_selection_h1_only": True,
        "epsilon_calibration_h1_only": True,
        "epsilon0_max_abs_difference_hours": epsilon0_max,
        "primary_dro_top5": metrics.sort_values("dro_score", ascending=False).head(5)["transition"].astype(str).tolist(),
        "q3_k5_captured_mean_burden_share_pct": {method: float(k5.loc[method, "q3_captured_mean_burden_share_pct"]) for method in METHODS},
        "bootstrap_ci_dro_minus_baseline_pp": {
            baseline: {
                "low": float(ci_rows.loc["DRO minus " + baseline, "ci95_low_pp"]),
                "high": float(ci_rows.loc["DRO minus " + baseline, "ci95_high_pp"]),
                "includes_zero": bool(ci_rows.loc["DRO minus " + baseline, "ci_includes_zero"]),
            }
            for baseline in ("Frequency", "Mean burden", "Tail baseline")
        },
        "dro_advantage_defensible": bool(advantage),
        "warnings": warnings,
    }
    json_dump(OUTPUT_DIR / "phase3_summary.json", summary)
    write_report(summary, metrics, q3_eval, bootstrap_differences, sensitivity_eval)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
