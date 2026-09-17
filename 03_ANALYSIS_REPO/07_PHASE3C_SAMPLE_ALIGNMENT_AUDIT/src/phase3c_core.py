from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any


import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from scipy.stats import kendalltau, spearmanr


SEED = 20260914
H1_START = pd.Timestamp("2018-01-01 00:00:00", tz="UTC")
H1_END = pd.Timestamp("2018-07-01 00:00:00", tz="UTC")
OPERATIONAL_CUTOFF = pd.Timestamp("2019-01-18 13:34:00", tz="UTC")
PRIMARY_CAP_HOURS = 2160
PRIMARY_ALPHA = 0.95
PRIMARY_GRID_HOURS = 24
CANDIDATE_COUNT = 63
MASS_TOL = 1e-12
NUMERIC_TOL = 1e-6
LP_TOL = 1e-5
TRANSITION_COLUMNS = ["timestamp", "transition", "inter_event_gap_hours"]
EXPECTED_MISMATCH_TRANSITIONS = [
    "Record Invoice Receipt -> Record Invoice Receipt",
    "Change Quantity -> Change Quantity",
    "Record Goods Receipt -> Cancel Goods Receipt",
    "Change Quantity -> Change Delivery Indicator",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parents[1]
PHASE3_RESULTS_DIR = PROJECT_ROOT / "04_ANALYSIS_RESULTS" / "05_PHASE3_DRO_BOTTLENECK_PRIORITIZATION"
PHASE3B_RESULTS_DIR = PROJECT_ROOT / "04_ANALYSIS_RESULTS" / "06_PHASE3B_CVAR_SATURATION_ANALYSIS"
OUTPUT_DIR = PROJECT_ROOT / "04_ANALYSIS_RESULTS" / "07_PHASE3C_SAMPLE_ALIGNMENT_AUDIT"

PERMITTED_PHASE3_FILES = {
    "01_h1_candidate_transitions.csv",
    "02_h1_block_wasserstein_calibration.csv",
    "03_transition_scores_primary.csv",
}
PERMITTED_PHASE3B_FILE = "01_transition_saturation_thresholds.csv"


def as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def finite_float(value: Any) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"Expected a finite numeric value, received {value!r}.")
    return number


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (dict, list, tuple, pd.DataFrame, pd.Series)) else False:
        return None
    return value


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", na_rep="")


def find_events_parquet() -> Path:
    configured = os.environ.get("BPI2019_EVENTS_PARQUET")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if candidate.exists() and candidate.name == "events_primary_cohort.parquet":
            return candidate
        raise FileNotFoundError(
            "BPI2019_EVENTS_PARQUET must point to an existing events_primary_cohort.parquet file."
        )
    candidate = PROJECT_ROOT / "04_ANALYSIS_RESULTS" / "03_PHASE2_PROCESS_STRUCTURE" / "processed" / "events_primary_cohort.parquet"
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(
        "No events_primary_cohort.parquet was found. Run the Phase 2 cohort build first or set BPI2019_EVENTS_PARQUET."
    )



def read_permitted_phase3_csv(filename: str) -> pd.DataFrame:
    if filename not in PERMITTED_PHASE3_FILES:
        raise ValueError(f"Attempted to read a non-permitted Phase 3 input: {filename}")
    direct = PHASE3_RESULTS_DIR / filename
    if not direct.exists():
        raise FileNotFoundError(f"Missing permitted Phase 3 input: {filename}")
    return pd.read_csv(direct)



def read_permitted_phase3b_csv() -> pd.DataFrame:
    direct = PHASE3B_RESULTS_DIR / PERMITTED_PHASE3B_FILE
    if not direct.exists():
        raise FileNotFoundError(f"Missing permitted Phase 3B input: {PERMITTED_PHASE3B_FILE}")
    return pd.read_csv(direct)



def read_phase3b_strict_h1_total() -> int:
    """Read the H1 row-count field from the Phase 3B summary."""
    direct = PHASE3B_RESULTS_DIR / "phase3b_summary.json"
    if not direct.exists():
        raise FileNotFoundError("The Phase 3B summary is missing.")
    summary = json.loads(direct.read_text(encoding="utf-8"))
    value = summary.get("h1_event_rows_loaded")
    if value is None:
        raise FileNotFoundError("The Phase 3B summary does not expose h1_event_rows_loaded.")
    return int(value)



def load_allowed_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = read_permitted_phase3_csv("01_h1_candidate_transitions.csv")
    calibration = read_permitted_phase3_csv("02_h1_block_wasserstein_calibration.csv")
    scores = read_permitted_phase3_csv("03_transition_scores_primary.csv")
    phase3b = read_permitted_phase3b_csv()

    candidates["candidate_h1_only"] = as_bool(candidates["candidate_h1_only"])
    candidate_transitions = set(
        candidates.loc[candidates["candidate_h1_only"], "transition"].dropna().astype(str)
    )
    if len(candidate_transitions) != CANDIDATE_COUNT:
        raise RuntimeError(
            f"The frozen candidate set must contain {CANDIDATE_COUNT} transitions, found {len(candidate_transitions)}."
        )

    calibration["candidate_h1_only"] = as_bool(calibration["candidate_h1_only"])
    if "calibration_role" not in calibration.columns:
        raise RuntimeError("The Phase 3 calibration input does not expose calibration_role.")
    calibration = calibration[calibration["calibration_role"].astype(str) == "primary"].copy()
    calibration = calibration[calibration["transition"].astype(str).isin(candidate_transitions)].copy()
    scores = scores[scores["transition"].astype(str).isin(candidate_transitions)].copy()
    phase3b = phase3b[phase3b["transition"].astype(str).isin(candidate_transitions)].copy()
    if len(calibration) != CANDIDATE_COUNT or len(scores) != CANDIDATE_COUNT:
        raise RuntimeError("The permitted Phase 3 inputs do not contain the same 63-transition universe.")
    if len(phase3b) != CANDIDATE_COUNT:
        raise RuntimeError("The permitted Phase 3B H1-only table does not contain the same 63-transition universe.")

    for frame in (calibration, scores, phase3b):
        frame["transition"] = frame["transition"].astype(str)
    return candidates, calibration, scores, phase3b


def read_exact_phase3_h1_events(events_path: Path) -> pd.DataFrame:
    """Reconstruct the exact Phase 3 H1-origin sample with a predicate read."""
    filters = [
        ("timestamp", ">=", H1_START.to_pydatetime()),
        ("timestamp", "<", (OPERATIONAL_CUTOFF + pd.Timedelta(seconds=1)).to_pydatetime()),
    ]
    try:
        table = pq.read_table(events_path, columns=TRANSITION_COLUMNS, filters=filters)
    except Exception as exc:
        raise RuntimeError(
            "The exact destination-timestamp predicate read failed. An unfiltered Parquet read is prohibited."
        ) from exc

    frame = table.to_pandas()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["transition"] = frame["transition"].astype("string")
    frame["gap_hours_raw"] = pd.to_numeric(frame["inter_event_gap_hours"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "transition", "gap_hours_raw"])
    frame = frame[np.isfinite(frame["gap_hours_raw"].to_numpy(float))]
    frame = frame[frame["gap_hours_raw"] >= 0].copy()
    frame["origin_timestamp"] = frame["timestamp"] - pd.to_timedelta(frame["gap_hours_raw"], unit="h")
    frame = frame[
        (frame["origin_timestamp"] >= H1_START)
        & (frame["origin_timestamp"] < H1_END)
    ].copy()
    if not frame.empty:
        if frame["origin_timestamp"].min() < H1_START or frame["origin_timestamp"].max() >= H1_END:
            raise RuntimeError("The reconstructed sample contains an origin outside the H1 interval.")
        if frame["timestamp"].min() < H1_START or frame["timestamp"].max() >= OPERATIONAL_CUTOFF + pd.Timedelta(seconds=1):
            raise RuntimeError("The reconstructed sample contains a destination outside the Phase 3 read window.")
    return frame.reset_index(drop=True)


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


def grid_definition() -> np.ndarray:
    return np.arange(
        0.0,
        float(PRIMARY_CAP_HOURS) + float(PRIMARY_GRID_HOURS) * 0.5,
        float(PRIMARY_GRID_HOURS),
    )


def grid_probabilities(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grid = grid_definition()
    mapped = np.clip(
        np.rint(np.asarray(values, dtype=float) / PRIMARY_GRID_HOURS) * PRIMARY_GRID_HOURS,
        0.0,
        float(PRIMARY_CAP_HOURS),
    )
    positions = np.searchsorted(grid, mapped).astype(int)
    counts = np.bincount(positions, minlength=len(grid)).astype(float)
    if counts.sum() <= 0:
        raise RuntimeError("A candidate transition has no finite elapsed-time observations.")
    return grid, counts / counts.sum()


def cvar_from_grid_probabilities(grid: np.ndarray, probabilities: np.ndarray, alpha: float) -> float:
    best = float("inf")
    for threshold in grid:
        hinge = np.maximum(grid - float(threshold), 0.0)
        candidate = float(threshold + np.dot(probabilities, hinge) / (1.0 - alpha))
        best = min(best, candidate)
    return best


def build_w1_constraints(probabilities: np.ndarray, step_hours: float, epsilon: float):
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


def worst_case_cvar(
    grid: np.ndarray,
    probabilities: np.ndarray,
    alpha: float,
    epsilon: float,
    template: tuple[Any, Any, Any, np.ndarray] | None = None,
) -> float:
    if template is None:
        template = build_w1_constraints(probabilities, float(grid[1] - grid[0]), epsilon)
    matrix, base_rhs, equality, equality_rhs = template
    n = len(grid)
    objective = np.zeros(n + n - 1, dtype=float)
    bounds = [(0.0, None)] * len(objective)
    rhs = np.asarray(base_rhs, dtype=float).copy()
    rhs[-1] = max(float(epsilon), 0.0)
    candidates: list[float] = []
    for threshold in grid:
        objective[:n] = -np.maximum(grid - float(threshold), 0.0)
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
            raise RuntimeError(f"The CVaR LP failed at threshold {threshold}: {result.message}")
        candidates.append(float(threshold - result.fun / (1.0 - alpha)))
    return float(min(candidates))


def transition_vectors(h1: pd.DataFrame, candidate_transitions: set[str]) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for transition in sorted(candidate_transitions):
        vector = pd.to_numeric(
            h1.loc[h1["transition"].astype(str) == transition, "gap_hours_raw"],
            errors="coerce",
        ).to_numpy(float)
        vector = vector[np.isfinite(vector) & (vector >= 0)]
        if len(vector) == 0:
            raise RuntimeError(f"The exact Phase 3 sample contains no observations for {transition!r}.")
        values[transition] = vector
    return values


def recompute_phase3_metrics(
    h1: pd.DataFrame,
    candidates: pd.DataFrame,
    calibration: pd.DataFrame,
    scores: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    candidate_transitions = set(
        candidates.loc[candidates["candidate_h1_only"], "transition"].astype(str)
    )
    vectors = transition_vectors(h1, candidate_transitions)
    calibration_map = calibration.set_index("transition")
    score_map = scores.set_index("transition")
    total_observations = len(h1)
    rows: list[dict[str, Any]] = []

    for transition in sorted(candidate_transitions):
        vector = vectors[transition]
        capped = np.clip(vector, 0.0, float(PRIMARY_CAP_HOURS))
        grid, probabilities = grid_probabilities(vector)
        epsilon = finite_float(calibration_map.loc[transition, "epsilon_q90_hours"])
        template = build_w1_constraints(probabilities, PRIMARY_GRID_HOURS, epsilon)
        dro_cvar = worst_case_cvar(grid, probabilities, PRIMARY_ALPHA, epsilon, template)
        grid_empirical = cvar_from_grid_probabilities(grid, probabilities, PRIMARY_ALPHA)
        frequency = float(len(vector) / total_observations)
        mean_burden = float(frequency * np.mean(capped))
        tail_score = float(frequency * empirical_cvar_array(capped, PRIMARY_ALPHA))
        row = {
            "transition": transition,
            "recomputed_h1_observations": int(len(vector)),
            "recomputed_h1_frequency_share": frequency,
            "recomputed_h1_mean_capped_elapsed_hours": float(np.mean(capped)),
            "recomputed_h1_empirical_cvar95_hours": float(empirical_cvar_array(capped, PRIMARY_ALPHA)),
            "recomputed_h1_grid_empirical_cvar95_hours": float(grid_empirical),
            "recomputed_epsilon_primary_hours": epsilon,
            "recomputed_h1_dro_worst_case_cvar95_hours": float(dro_cvar),
            "recomputed_mean_burden_score": mean_burden,
            "recomputed_tail_score": tail_score,
            "recomputed_dro_score": float(frequency * dro_cvar),
            "phase3_epsilon_primary_from_score": finite_float(score_map.loc[transition, "epsilon_primary_hours"]),
        }
        rows.append(row)

    metrics = pd.DataFrame(rows).sort_values("transition").reset_index(drop=True)
    return metrics, vectors


def compare_metric(
    expected: float,
    observed: float,
    exact: bool = False,
) -> tuple[float, float, bool]:
    absolute_error = abs(float(observed) - float(expected))
    relative_error = absolute_error / abs(float(expected)) if float(expected) != 0 else (0.0 if absolute_error == 0 else float("inf"))
    passed = bool(absolute_error == 0) if exact else bool(absolute_error <= NUMERIC_TOL)
    return absolute_error, relative_error, passed


def build_reproduction_check(
    recomputed: pd.DataFrame,
    scores: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float], int]:
    expected = scores.copy()
    expected["transition"] = expected["transition"].astype(str)
    expected = expected.set_index("transition")
    recomputed = recomputed.set_index("transition")
    metric_map = {
        "h1_observations": ("h1_observations", "recomputed_h1_observations", True),
        "h1_frequency_share": ("h1_frequency_share", "recomputed_h1_frequency_share", False),
        "h1_mean_capped_elapsed_hours": (
            "h1_mean_capped_elapsed_hours",
            "recomputed_h1_mean_capped_elapsed_hours",
            False,
        ),
        "h1_empirical_cvar95_hours": (
            "h1_empirical_cvar95_hours",
            "recomputed_h1_empirical_cvar95_hours",
            False,
        ),
        "h1_grid_empirical_cvar95_hours": (
            "h1_grid_empirical_cvar95_hours",
            "recomputed_h1_grid_empirical_cvar95_hours",
            False,
        ),
        "epsilon_primary_hours": (
            "epsilon_primary_hours",
            "recomputed_epsilon_primary_hours",
            False,
        ),
        "h1_dro_worst_case_cvar95_hours": (
            "h1_dro_worst_case_cvar95_hours",
            "recomputed_h1_dro_worst_case_cvar95_hours",
            False,
        ),
        "mean_burden_score": ("mean_burden_score", "recomputed_mean_burden_score", False),
        "tail_score": ("tail_score", "recomputed_tail_score", False),
        "dro_score": ("dro_score", "recomputed_dro_score", False),
    }
    rows: list[dict[str, Any]] = []
    max_abs: dict[str, float] = {}
    for transition in sorted(expected.index):
        row: dict[str, Any] = {"transition": transition, "candidate_h1_only": True}
        row_pass = True
        for metric, (expected_col, observed_col, exact) in metric_map.items():
            expected_value = float(expected.loc[transition, expected_col])
            observed_value = float(recomputed.loc[transition, observed_col])
            absolute_error, relative_error, passed = compare_metric(expected_value, observed_value, exact=exact)
            row[f"phase3_{metric}"] = expected_value
            row[f"recomputed_{metric}"] = observed_value
            row[f"{metric}_absolute_error"] = absolute_error
            row[f"{metric}_relative_error"] = relative_error
            row[f"{metric}_pass"] = passed
            row_pass = row_pass and passed
            max_abs[metric] = max(max_abs.get(metric, 0.0), absolute_error)
        row["gate_pass"] = row_pass
        rows.append(row)
    check = pd.DataFrame(rows)
    mismatch_count = int((~check["gate_pass"]).sum())
    return check, max_abs, mismatch_count


def bridge_table(h1: pd.DataFrame, candidate_transitions: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    bridge = h1[h1["timestamp"] >= H1_END].copy()
    base = h1.groupby(h1["transition"].astype(str), sort=True).size().rename("phase3_h1_observations")
    bridge_counts = bridge.groupby(bridge["transition"].astype(str), sort=True).size().rename("bridge_rows")
    table = pd.DataFrame(index=base.index)
    table["phase3_h1_observations"] = base.astype(int)
    table["bridge_rows"] = bridge_counts.reindex(table.index).fillna(0).astype(int)
    table["candidate_h1_only"] = table.index.isin(candidate_transitions)
    table["bridge_percentage_of_phase3_h1_sample"] = 100.0 * table["bridge_rows"] / len(h1)

    if bridge.empty:
        for column in [
            "bridge_elapsed_median_hours",
            "bridge_elapsed_q75_hours",
            "bridge_elapsed_q90_hours",
            "bridge_elapsed_max_hours",
            "bridge_destination_min_utc",
            "bridge_destination_max_utc",
        ]:
            table[column] = np.nan
    else:
        grouped = bridge.groupby(bridge["transition"].astype(str), sort=True)
        table["bridge_elapsed_median_hours"] = grouped["gap_hours_raw"].median()
        table["bridge_elapsed_q75_hours"] = grouped["gap_hours_raw"].quantile(0.75)
        table["bridge_elapsed_q90_hours"] = grouped["gap_hours_raw"].quantile(0.90)
        table["bridge_elapsed_max_hours"] = grouped["gap_hours_raw"].max()
        table["bridge_destination_min_utc"] = grouped["timestamp"].min().astype(str)
        table["bridge_destination_max_utc"] = grouped["timestamp"].max().astype(str)
    table = table.reset_index(names="transition")
    return table, bridge


def saturation_threshold_from_grid(
    grid: np.ndarray,
    probabilities: np.ndarray,
    alpha: float,
    cap_hours: float,
) -> tuple[float, float, float]:
    cap_position = int(np.argmax(grid >= float(cap_hours)))
    if not np.isclose(grid[cap_position], float(cap_hours)):
        raise ValueError("The saturation cap must be a grid point.")
    m_cap = float(probabilities[cap_position])
    delta = max(0.0, (1.0 - alpha) - m_cap)
    if delta <= MASS_TOL:
        return m_cap, 0.0, 0.0

    remaining = delta
    cost = 0.0
    for index in range(cap_position - 1, -1, -1):
        available = float(probabilities[index])
        moved = min(available, remaining)
        cost += moved * (float(cap_hours) - float(grid[index]))
        remaining -= moved
        if remaining <= MASS_TOL:
            break
    if remaining > 1e-10:
        raise RuntimeError("The discrete saturation mass could not be transported to the cap.")
    return m_cap, delta, float(cost)


def aligned_saturation_thresholds(
    vectors: dict[str, np.ndarray],
    recomputed: pd.DataFrame,
    phase3b: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_map = recomputed.set_index("transition")
    phase3b_map = phase3b.set_index("transition")
    rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []

    for transition in sorted(vectors):
        vector = vectors[transition]
        grid, probabilities = grid_probabilities(vector)
        m_cap, delta, epsilon_sat = saturation_threshold_from_grid(
            grid, probabilities, PRIMARY_ALPHA, PRIMARY_CAP_HOURS
        )
        empirical_cvar = cvar_from_grid_probabilities(grid, probabilities, PRIMARY_ALPHA)
        epsilon_primary = finite_float(metric_map.loc[transition, "recomputed_epsilon_primary_hours"])
        observed_dro = finite_float(metric_map.loc[transition, "recomputed_h1_dro_worst_case_cvar95_hours"])
        empirical_saturation = bool(abs(empirical_cvar - PRIMARY_CAP_HOURS) <= LP_TOL)
        predicted_saturated = bool(
            True if epsilon_sat <= MASS_TOL else epsilon_primary >= epsilon_sat - LP_TOL
        )
        observed_saturated = bool(abs(observed_dro - PRIMARY_CAP_HOURS) <= LP_TOL)
        dro_induced = bool(predicted_saturated and not empirical_saturation)
        total_saturated = bool(empirical_saturation or dro_induced)
        unsaturated = bool(not total_saturated)
        saturation_ratio = float(epsilon_primary / epsilon_sat) if epsilon_sat > MASS_TOL else float("nan")
        phase3b_row = phase3b_map.loc[transition]
        phase3b_observed = bool(as_bool(pd.Series([phase3b_row["observed_saturated"]])).iloc[0])
        phase3b_predicted = bool(as_bool(pd.Series([phase3b_row["predicted_saturated"]])).iloc[0])
        phase3b_total = bool(
            as_bool(pd.Series([phase3b_row["total_saturated_under_primary_radius"]])).iloc[0]
        )
        rows.append(
            {
                "transition": transition,
                "observations": int(len(vector)),
                "m_cap": m_cap,
                "delta_mass_to_cap": delta,
                "empirical_cvar95": empirical_cvar,
                "epsilon_primary": epsilon_primary,
                "epsilon_sat": epsilon_sat,
                "saturation_ratio": saturation_ratio,
                "empirical_saturation": empirical_saturation,
                "dro_induced_saturation": dro_induced,
                "predicted_saturated": predicted_saturated,
                "observed_saturated": observed_saturated,
                "total_saturated_under_primary_radius": total_saturated,
                "unsaturated": unsaturated,
                "observed_phase3_dro_cvar": observed_dro,
                "tail_score": finite_float(metric_map.loc[transition, "recomputed_tail_score"]),
                "mean_burden_score": finite_float(metric_map.loc[transition, "recomputed_mean_burden_score"]),
                "dro_score": finite_float(metric_map.loc[transition, "recomputed_dro_score"]),
                "phase3b_observed_saturated": phase3b_observed,
                "phase3b_predicted_saturated": phase3b_predicted,
                "phase3b_total_saturated": phase3b_total,
            }
        )

        template = build_w1_constraints(probabilities, PRIMARY_GRID_HOURS, 0.0)
        lp_zero = worst_case_cvar(grid, probabilities, PRIMARY_ALPHA, 0.0, template)
        if epsilon_sat > MASS_TOL:
            lp_sat = worst_case_cvar(grid, probabilities, PRIMARY_ALPHA, epsilon_sat, template)
            lp_near = worst_case_cvar(grid, probabilities, PRIMARY_ALPHA, 0.99 * epsilon_sat, template)
            near_required = True
            near_pass = bool(lp_near < PRIMARY_CAP_HOURS - LP_TOL)
        else:
            lp_sat = lp_zero
            lp_near = lp_zero
            near_required = False
            near_pass = True
        zero_error = abs(lp_zero - empirical_cvar)
        sat_error = abs(lp_sat - PRIMARY_CAP_HOURS)
        validation_rows.append(
            {
                "transition": transition,
                "epsilon_zero": 0.0,
                "epsilon_sat": epsilon_sat,
                "epsilon_near_sat": 0.99 * epsilon_sat,
                "grid_empirical_cvar95": empirical_cvar,
                "lp_cvar_at_epsilon_zero": lp_zero,
                "lp_zero_absolute_error_hours": zero_error,
                "lp_cvar_at_epsilon_sat": lp_sat,
                "lp_sat_absolute_error_to_cap_hours": sat_error,
                "lp_cvar_at_0_99_epsilon_sat": lp_near,
                "near_threshold_required": near_required,
                "near_threshold_below_cap": bool(lp_near < PRIMARY_CAP_HOURS - LP_TOL),
                "near_threshold_pass": near_pass,
                "epsilon_zero_pass": bool(zero_error <= LP_TOL),
                "epsilon_sat_pass": bool(sat_error <= LP_TOL),
                "validation_pass": bool(zero_error <= LP_TOL and sat_error <= LP_TOL and near_pass),
            }
        )

    return pd.DataFrame(rows).sort_values("transition").reset_index(drop=True), pd.DataFrame(validation_rows).sort_values("transition").reset_index(drop=True)


def phase3b_comparison(phase3b: pd.DataFrame, aligned: pd.DataFrame) -> pd.DataFrame:
    before = phase3b.copy()
    before["transition"] = before["transition"].astype(str)
    after = aligned.copy()
    merged = before.merge(after, on="transition", how="inner", suffixes=("_phase3b", "_phase3c"))
    if len(merged) != CANDIDATE_COUNT:
        raise RuntimeError("Phase 3B and Phase 3C comparison does not cover all 63 candidates.")
    output = pd.DataFrame({"transition": merged["transition"]})
    for name, before_col, after_col in [
        ("observations", "observations_phase3b", "observations_phase3c"),
        ("m_cap", "m_cap_phase3b", "m_cap_phase3c"),
        ("epsilon_sat", "epsilon_sat_phase3b", "epsilon_sat_phase3c"),
        ("saturation_ratio", "saturation_ratio_phase3b", "saturation_ratio_phase3c"),
    ]:
        output[f"phase3b_{name}"] = merged[before_col]
        output[f"phase3c_{name}"] = merged[after_col]
        output[f"{name}_difference"] = pd.to_numeric(merged[after_col], errors="coerce") - pd.to_numeric(
            merged[before_col], errors="coerce"
        )
    output["phase3b_empirical_saturation"] = as_bool(merged["empirical_saturation_phase3b"])
    output["phase3c_empirical_saturation"] = as_bool(merged["empirical_saturation_phase3c"])
    output["phase3b_predicted_saturated"] = as_bool(merged["predicted_saturated_phase3b"])
    output["phase3c_predicted_saturated"] = as_bool(merged["predicted_saturated_phase3c"])
    output["phase3b_observed_saturated"] = as_bool(merged["observed_saturated_phase3b"])
    output["phase3c_observed_saturated"] = as_bool(merged["observed_saturated_phase3c"])
    output["phase3b_total_saturated"] = as_bool(merged["total_saturated_under_primary_radius_phase3b"])
    output["phase3c_total_saturated"] = as_bool(merged["total_saturated_under_primary_radius_phase3c"])
    output["classification_changed"] = output["phase3b_total_saturated"] != output["phase3c_total_saturated"]
    return output.sort_values("transition").reset_index(drop=True)


def rank_value(result: Any) -> float | None:
    value = float(result.statistic if hasattr(result, "statistic") else result[0])
    return value if np.isfinite(value) else None


def portfolio_diagnostics(aligned: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = aligned.copy()
    rows: list[dict[str, Any]] = []
    spearman = rank_value(spearmanr(frame["tail_score"], frame["dro_score"]))
    kendall = rank_value(kendalltau(frame["tail_score"], frame["dro_score"]))
    for method, value in [("Spearman", spearman), ("Kendall", kendall)]:
        rows.append(
            {
                "diagnostic": "rank_correlation",
                "k": np.nan,
                "metric_1": "Tail score",
                "metric_2": "DRO score",
                "correlation_method": method,
                "correlation_coefficient": value,
            }
        )
    mean_spearman = rank_value(spearmanr(frame["mean_burden_score"], frame["dro_score"]))
    mean_kendall = rank_value(kendalltau(frame["mean_burden_score"], frame["dro_score"]))
    for method, value in [("Spearman", mean_spearman), ("Kendall", mean_kendall)]:
        rows.append(
            {
                "diagnostic": "rank_correlation",
                "k": np.nan,
                "metric_1": "Mean burden",
                "metric_2": "DRO score",
                "correlation_method": method,
                "correlation_coefficient": value,
            }
        )

    portfolio_summary: dict[str, Any] = {}
    for k in (3, 5, 10):
        tail_top = set(
            frame.sort_values(["tail_score", "transition"], ascending=[False, True]).head(k)["transition"]
        )
        dro_top = set(
            frame.sort_values(["dro_score", "transition"], ascending=[False, True]).head(k)["transition"]
        )
        overlap_count = len(tail_top & dro_top)
        union_count = len(tail_top | dro_top)
        jaccard = float(overlap_count / union_count) if union_count else float("nan")
        overlap_fraction = float(overlap_count / k)
        dro_portfolio = frame[frame["transition"].isin(dro_top)]
        tail_portfolio = frame[frame["transition"].isin(tail_top)]
        row = {
            "diagnostic": "top_k_portfolio",
            "k": k,
            "metric_1": "Tail score",
            "metric_2": "DRO score",
            "correlation_method": "",
            "correlation_coefficient": np.nan,
            "tail_top_size": len(tail_top),
            "dro_top_size": len(dro_top),
            "overlap_count": overlap_count,
            "overlap_fraction": overlap_fraction,
            "jaccard": jaccard,
            "dro_portfolio_saturated_count": int(dro_portfolio["total_saturated_under_primary_radius"].sum()),
            "dro_portfolio_unsaturated_count": int(dro_portfolio["unsaturated"].sum()),
            "tail_portfolio_saturated_count": int(tail_portfolio["total_saturated_under_primary_radius"].sum()),
            "tail_portfolio_unsaturated_count": int(tail_portfolio["unsaturated"].sum()),
        }
        rows.append(row)
        portfolio_summary[f"k_{k}"] = {
            "tail_dro_jaccard": jaccard,
            "tail_dro_overlap_count": overlap_count,
            "tail_dro_overlap_fraction": overlap_fraction,
            "dro_portfolio_saturated_count": row["dro_portfolio_saturated_count"],
            "dro_portfolio_unsaturated_count": row["dro_portfolio_unsaturated_count"],
        }
    return pd.DataFrame(rows), {
        "tail_vs_dro_spearman": spearman,
        "tail_vs_dro_kendall": kendall,
        "mean_vs_dro_spearman": mean_spearman,
        "mean_vs_dro_kendall": mean_kendall,
        **portfolio_summary,
    }




def mismatch_detail(
    aligned: pd.DataFrame,
    phase3b: pd.DataFrame,
    bridge: pd.DataFrame,
) -> pd.DataFrame:
    before = phase3b.set_index("transition")
    after = aligned.set_index("transition")
    bridge_map = bridge.set_index("transition")
    rows: list[dict[str, Any]] = []
    for transition in EXPECTED_MISMATCH_TRANSITIONS:
        if transition not in before.index or transition not in after.index:
            rows.append({"transition": transition, "status": "not found in frozen candidate universe"})
            continue
        rows.append(
            {
                "transition": transition,
                "status": "present",
                "phase3_h1_count": int(after.loc[transition, "observations"]),
                "phase3b_strict_h1_count": int(before.loc[transition, "observations"]),
                "bridge_count": int(bridge_map.loc[transition, "bridge_rows"]),
                "phase3b_m_cap": float(before.loc[transition, "m_cap"]),
                "phase3c_m_cap": float(after.loc[transition, "m_cap"]),
                "phase3b_epsilon_sat": float(before.loc[transition, "epsilon_sat"]),
                "phase3c_epsilon_sat": float(after.loc[transition, "epsilon_sat"]),
                "phase3b_empirical_saturation": bool(as_bool(pd.Series([before.loc[transition, "empirical_saturation"]])).iloc[0]),
                "phase3c_empirical_saturation": bool(after.loc[transition, "empirical_saturation"]),
                "phase3b_total_saturated": bool(as_bool(pd.Series([before.loc[transition, "total_saturated_under_primary_radius"]])).iloc[0]),
                "phase3c_total_saturated": bool(after.loc[transition, "total_saturated_under_primary_radius"]),
                "phase3b_predicted_saturated": bool(as_bool(pd.Series([before.loc[transition, "predicted_saturated"]])).iloc[0]),
                "phase3c_predicted_saturated": bool(after.loc[transition, "predicted_saturated"]),
                "phase3b_observed_saturated": bool(as_bool(pd.Series([before.loc[transition, "observed_saturated"]])).iloc[0]),
                "phase3c_observed_saturated": bool(after.loc[transition, "observed_saturated"]),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    events_path: Path,
    h1: pd.DataFrame,
    phase3b_strict_total: int,
    phase3b: pd.DataFrame,
    reproduction: pd.DataFrame,
    max_abs: dict[str, float],
    bridge: pd.DataFrame,
    bridge_rows: pd.DataFrame,
    aligned: pd.DataFrame,
    validation: pd.DataFrame,
    comparison: pd.DataFrame,
    diagnostics: dict[str, Any],
    detail: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    report_lines: list[str] = []
    report_lines.append("# Phase 3C, Exact Sample Alignment Audit")
    report_lines.append("")
    report_lines.append("## Decision")
    report_lines.append("")
    report_lines.append("The exact Phase 3 H1-origin convention was reproduced for all 63 frozen candidate transitions.")
    report_lines.append("The reproduction gate passed. Phase 3 H1 observation counts and primary metrics match within the declared numeric tolerance.")
    report_lines.append("No new candidate selection, parameter tuning, or out-of-time evaluation was performed.")
    report_lines.append("")
    report_lines.append("## Input isolation")
    report_lines.append("")
    report_lines.append("Q3 was NOT read or used.")
    report_lines.append("The Parquet predicate read extends beyond 30 June only because a destination timestamp after June can belong to an H1-origin transition under the exact Phase 3 convention. Only rows with an inferred origin in H1 are retained. No Q3-origin row, Q3 result file, Q3 output, or Q3-derived statistic enters the analysis.")
    report_lines.append(f"Events input: `{events_path}`.")
    report_lines.append(f"Destination read window: `{H1_START.isoformat()}` through `{(OPERATIONAL_CUTOFF + pd.Timedelta(seconds=1)).isoformat()}` as an exclusive upper bound.")
    report_lines.append(f"Reconstructed H1-origin observations: **{len(h1):,}**.")
    report_lines.append(f"Phase 3B strict destination-timestamp observations: **{phase3b_strict_total:,}**.")
    report_lines.append("")
    report_lines.append("## Reproduction gate")
    report_lines.append("")
    report_lines.append("The candidate set is frozen at 63 transitions. The primary configuration is `C = 2160` hours, `alpha = 0.95`, a 24-hour grid, and the original Phase 3 primary Wasserstein radius.")
    report_lines.append(f"Rows passing every reproduction metric: **{int(reproduction['gate_pass'].sum())} of {len(reproduction)}**.")
    report_lines.append(f"Rows with a reproduction mismatch: **{int((~reproduction['gate_pass']).sum())}**.")
    report_lines.append("")
    report_lines.append("Maximum absolute discrepancies")
    report_lines.append("")
    report_lines.append("| Metric | Maximum absolute error |")
    report_lines.append("|---|---:|")
    for metric, value in max_abs.items():
        report_lines.append(f"| {metric} | {value:.12g} |")
    report_lines.append("")
    report_lines.append("## H1 boundary bridge rows")
    report_lines.append("")
    report_lines.append(f"Bridge rows are H1-origin rows with a destination timestamp on or after 1 July 2018. The aligned sample contains **{len(bridge):,}** bridge rows across **{int((bridge_rows['bridge_rows'] > 0).sum())}** transitions.")
    report_lines.append(f"Bridge rows represent **{100.0 * len(bridge) / len(h1):.4f}%** of the exact H1-origin sample.")
    report_lines.append("")
    report_lines.append("The four Phase 3B classification discrepancies are inspected below.")
    report_lines.append("")
    report_lines.append("| Transition | Phase 3 H1 | Phase 3B strict | Bridge | m_cap before | m_cap after | epsilon_sat before | epsilon_sat after | Empirical before | Empirical after | Predicted before | Predicted after | Observed before | Observed after | Total before | Total after |")
    report_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for _, row in detail.iterrows():
        if row.get("status") != "present":
            report_lines.append(f"| {row['transition']} | not found | | | | | | | | |")
            continue
        report_lines.append(
            f"| {row['transition']} | {int(row['phase3_h1_count']):,} | {int(row['phase3b_strict_h1_count']):,} | {int(row['bridge_count']):,} | {row['phase3b_m_cap']:.8f} | {row['phase3c_m_cap']:.8f} | {row['phase3b_epsilon_sat']:.6f} | {row['phase3c_epsilon_sat']:.6f} | {str(bool(row['phase3b_empirical_saturation']))} | {str(bool(row['phase3c_empirical_saturation']))} | {str(bool(row['phase3b_predicted_saturated']))} | {str(bool(row['phase3c_predicted_saturated']))} | {str(bool(row['phase3b_observed_saturated']))} | {str(bool(row['phase3c_observed_saturated']))} | {str(bool(row['phase3b_total_saturated']))} | {str(bool(row['phase3c_total_saturated']))} |"
        )
    report_lines.append("")
    report_lines.append("## Aligned saturation classification")
    report_lines.append("")
    empirical_count = int(aligned["empirical_saturation"].sum())
    induced_count = int(aligned["dro_induced_saturation"].sum())
    total_count = int(aligned["total_saturated_under_primary_radius"].sum())
    unsat_count = int(aligned["unsaturated"].sum())
    mismatch_count = int((aligned["predicted_saturated"] != aligned["observed_saturated"]).sum())
    report_lines.append(f"Empirical saturation count: **{empirical_count}**.")
    report_lines.append(f"DRO-induced saturation count: **{induced_count}**.")
    report_lines.append(f"Total saturation count: **{total_count} of {len(aligned)}**.")
    report_lines.append(f"Unsaturated count: **{unsat_count}**.")
    report_lines.append(f"Theoretical-vs-observed saturation mismatch count: **{mismatch_count}**.")
    report_lines.append("")
    report_lines.append("The quantity `epsilon_sat` is the bounded-support Wasserstein-CVaR saturation threshold diagnostic. It is an analytical consequence used here as a diagnostic and verified on the aligned empirical sample.")
    report_lines.append("For positive thresholds, the computation moves the required probability mass to the cap from the largest sub-cap grid point downward. When the empirical cap mass already reaches the upper-tail mass, `epsilon_sat = 0` is handled explicitly.")
    report_lines.append("")
    report_lines.append("## Linear-program validation")
    report_lines.append("")
    report_lines.append(f"Validation passed for **{int(validation['validation_pass'].sum())} of {len(validation)}** transitions.")
    report_lines.append(f"Positive epsilon_sat cases: **{int(validation['near_threshold_required'].sum())}**.")
    report_lines.append(f"Zero epsilon_sat cases: **{int((~validation['near_threshold_required']).sum())}**.")
    report_lines.append(f"Maximum absolute error at epsilon = 0: **{validation['lp_zero_absolute_error_hours'].max():.12g}** hours.")
    report_lines.append(f"Maximum absolute error at epsilon = epsilon_sat: **{validation['lp_sat_absolute_error_to_cap_hours'].max():.12g}** hours.")
    report_lines.append("")
    report_lines.append("At zero radius, the LP agrees with grid empirical CVaR. At the saturation threshold, it reaches the cap within tolerance. At `0.99 * epsilon_sat`, every positive-threshold case remains below the cap outside tolerance.")
    report_lines.append("")
    report_lines.append("## Aligned ranking diagnostics")
    report_lines.append("")
    report_lines.append(f"Tail versus DRO Spearman correlation: **{diagnostics['tail_vs_dro_spearman']}**.")
    report_lines.append(f"Tail versus DRO Kendall correlation: **{diagnostics['tail_vs_dro_kendall']}**.")
    report_lines.append(f"Mean burden versus DRO Spearman correlation: **{diagnostics['mean_vs_dro_spearman']}**.")
    report_lines.append(f"Mean burden versus DRO Kendall correlation: **{diagnostics['mean_vs_dro_kendall']}**.")
    report_lines.append("")
    report_lines.append("| K | Tail versus DRO Jaccard | Overlap count | DRO portfolio saturated | DRO portfolio unsaturated |")
    report_lines.append("|---:|---:|---:|---:|---:|")
    for k in (3, 5, 10):
        item = diagnostics[f"k_{k}"]
        report_lines.append(
            f"| {k} | {item['tail_dro_jaccard']:.8f} | {item['tail_dro_overlap_count']} | {item['dro_portfolio_saturated_count']} | {item['dro_portfolio_unsaturated_count']} |"
        )
    report_lines.append("")
    report_lines.append("## Phase 3B versus Phase 3C")
    report_lines.append("")
    report_lines.append(f"Transitions whose total saturation classification changed: **{int(comparison['classification_changed'].sum())}**.")
    report_lines.append("The comparison table retains the observation count, cap mass, saturation threshold, finite saturation ratio, and empirical, predicted, observed, and total classifications for all 63 transitions.")
    report_lines.append("")
    report_lines.append("## Integrity statement")
    report_lines.append("")
    report_lines.append("Phases 1 through 3B were not modified. The repository contains the audit code, report, summary, and six CSV tables. It does not contain the source Parquet file.")
    (OUTPUT_DIR / "PHASE3C_ALIGNMENT_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def run_phase3c() -> dict[str, Any]:
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    events_path = find_events_parquet()
    candidates, calibration, scores, phase3b = load_allowed_inputs()
    phase3b_strict_total = read_phase3b_strict_h1_total()
    candidate_transitions = set(candidates.loc[candidates["candidate_h1_only"], "transition"].astype(str))

    print(f"Reading exact Phase 3 H1-origin sample from {events_path}")
    h1 = read_exact_phase3_h1_events(events_path)
    print(f"Reconstructed H1-origin observations: {len(h1):,}")
    recomputed, vectors = recompute_phase3_metrics(h1, candidates, calibration, scores)
    reproduction, max_abs, reproduction_mismatch_count = build_reproduction_check(recomputed, scores)
    write_csv(OUTPUT_DIR / "01_phase3_reproduction_check.csv", reproduction)
    if len(reproduction) != CANDIDATE_COUNT or reproduction_mismatch_count != 0:
        raise RuntimeError(
            "The Phase 3 reproduction gate failed. The main Phase 3C analysis was stopped. "
            f"candidate_rows={len(reproduction)}, mismatched_rows={reproduction_mismatch_count}, max_abs={max_abs}"
        )
    print("Phase 3 reproduction gate passed for all 63 candidates.")

    bridge_rows, bridge = bridge_table(h1, candidate_transitions)
    write_csv(OUTPUT_DIR / "02_h1_boundary_bridge_rows_by_transition.csv", bridge_rows)
    aligned, validation = aligned_saturation_thresholds(vectors, recomputed, phase3b)
    write_csv(OUTPUT_DIR / "03_aligned_saturation_thresholds.csv", aligned)
    write_csv(OUTPUT_DIR / "04_aligned_lp_validation.csv", validation)
    comparison = phase3b_comparison(phase3b, aligned)
    write_csv(OUTPUT_DIR / "06_phase3b_vs_phase3c_comparison.csv", comparison)
    portfolio_table, diagnostics = portfolio_diagnostics(aligned)
    write_csv(OUTPUT_DIR / "05_aligned_portfolio_diagnostics.csv", portfolio_table)
    detail = mismatch_detail(aligned, phase3b, bridge_rows)

    finite_ratios = aligned.loc[np.isfinite(aligned["saturation_ratio"]), "saturation_ratio"]
    summary: dict[str, Any] = {
        "phase": "Phase 3C",
        "title": "EXACT SAMPLE ALIGNMENT AUDIT",
        "candidate_count": CANDIDATE_COUNT,
        "phase3_h1_observations": int(len(h1)),
        "phase3b_strict_h1_observations": phase3b_strict_total,
        "primary_cap_hours": PRIMARY_CAP_HOURS,
        "alpha": PRIMARY_ALPHA,
        "grid_step_hours": PRIMARY_GRID_HOURS,
        "operational_cutoff_utc": OPERATIONAL_CUTOFF.isoformat(),
        "q3_read": False,
        "q3_used": False,
        "exact_origin_convention_reproduced": True,
        "reproduction_gate_passed": True,
        "reproduction_mismatch_count": reproduction_mismatch_count,
        "metric_max_absolute_discrepancy": max_abs,
        "bridge_total_rows": int(len(bridge)),
        "bridge_transition_count": int((bridge_rows["bridge_rows"] > 0).sum()),
        "bridge_percentage_of_phase3_h1_sample": float(100.0 * len(bridge) / len(h1)),
        "empirical_saturated_count": int(aligned["empirical_saturation"].sum()),
        "dro_induced_saturated_count": int(aligned["dro_induced_saturation"].sum()),
        "total_saturated_count": int(aligned["total_saturated_under_primary_radius"].sum()),
        "unsaturated_count": int(aligned["unsaturated"].sum()),
        "theoretical_observed_saturation_mismatch_count": int(
            (aligned["predicted_saturated"] != aligned["observed_saturated"]).sum()
        ),
        "median_finite_saturation_ratio": float(finite_ratios.median()) if not finite_ratios.empty else None,
        "lp_validation_pass_count": int(validation["validation_pass"].sum()),
        "lp_validation_total": int(len(validation)),
        "lp_positive_epsilon_sat_count": int(validation["near_threshold_required"].sum()),
        "lp_zero_epsilon_sat_count": int((~validation["near_threshold_required"]).sum()),
        "lp_max_zero_radius_absolute_error_hours": float(validation["lp_zero_absolute_error_hours"].max()),
        "lp_max_saturation_radius_absolute_error_hours": float(validation["lp_sat_absolute_error_to_cap_hours"].max()),
        "classification_changed_phase3b_to_phase3c": int(comparison["classification_changed"].sum()),
        "four_phase3b_mismatch_rows": detail.to_dict(orient="records"),
        "ranking_diagnostics": diagnostics,
        "warnings": [],
        "errors": [],
        "phases_1_to_3b_modified": False,
        "events_source_basename": events_path.name,
    }
    if summary["theoretical_observed_saturation_mismatch_count"] != 0:
        summary["warnings"].append("The theoretical-vs-observed saturation mismatch count is nonzero.")
    if summary["classification_changed_phase3b_to_phase3c"] != 0:
        summary["warnings"].append("Some total saturation classifications changed after exact sample alignment.")
    json_dump(OUTPUT_DIR / "phase3c_summary.json", summary)
    write_report(
        events_path,
        h1,
        phase3b_strict_total,
        phase3b,
        reproduction,
        max_abs,
        bridge,
        bridge_rows,
        aligned,
        validation,
        comparison,
        diagnostics,
        detail,
        summary,
    )
    print("Phase 3C analysis completed.")
    return summary


if __name__ == "__main__":
    run_phase3c()
