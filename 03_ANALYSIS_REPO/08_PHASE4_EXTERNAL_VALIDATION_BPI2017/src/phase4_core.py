from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import shutil
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from scipy.stats import kendalltau, spearmanr, wasserstein_distance


SEED = 20260914
FIGSHARE_ARTICLE_ID = 12696884
FIGSHARE_API_URL = "https://api.figshare.com/v2/articles/12696884"
OFFICIAL_PAGE = "https://figshare.com/articles/dataset/BPI_Challenge_2017/12696884"
H1_START = pd.Timestamp("2016-01-01 00:00:00", tz="UTC")
H1_END = pd.Timestamp("2016-07-01 00:00:00", tz="UTC")
Q3_START = pd.Timestamp("2016-07-01 00:00:00", tz="UTC")
Q3_END = pd.Timestamp("2016-10-01 00:00:00", tz="UTC")
PRIMARY_CAP_HOURS = 2160
SENSITIVITY_CAP_HOURS = 2880
PRIMARY_ALPHA = 0.95
SENSITIVITY_ALPHA = 0.90
PRIMARY_GRID_HOURS = 24
SENSITIVITY_GRID_HOURS = 12
PRIMARY_KAPPA = 0.90
SENSITIVITY_KAPPA = 0.75
H1_BLOCK_DAYS = 14
MIN_H1_OBSERVATIONS = 500
MIN_VALID_BLOCKS = 8
MIN_BLOCK_OBSERVATIONS = 20
H1_BOOTSTRAP_REPS = 500
Q3_BOOTSTRAP_REPS = 1000
PORTFOLIO_KS = (3, 5, 10)
METHODS = (
    "Frequency",
    "Mean burden",
    "Empirical Tail",
    "Original DRO",
    "Saturation-Aware DRO",
)
H1_STABILITY_METHODS = (
    "Mean burden",
    "Empirical Tail",
    "Original DRO",
    "Saturation-Aware DRO",
)
TRANSITION_SCORE_COLUMNS = {
    "Frequency": "frequency_score",
    "Mean burden": "mean_burden_score",
    "Empirical Tail": "tail_score",
    "Original DRO": "original_dro_score",
    "Saturation-Aware DRO": "sa_dro_score",
}
SENSITIVITY_CONFIGS = (
    ("primary", PRIMARY_CAP_HOURS, PRIMARY_ALPHA, PRIMARY_GRID_HOURS, PRIMARY_KAPPA),
    ("cap120", SENSITIVITY_CAP_HOURS, PRIMARY_ALPHA, PRIMARY_GRID_HOURS, SENSITIVITY_KAPPA),
    ("alpha90", PRIMARY_CAP_HOURS, SENSITIVITY_ALPHA, PRIMARY_GRID_HOURS, SENSITIVITY_KAPPA),
    ("grid12", PRIMARY_CAP_HOURS, PRIMARY_ALPHA, SENSITIVITY_GRID_HOURS, SENSITIVITY_KAPPA),
    ("cap120_alpha90", SENSITIVITY_CAP_HOURS, SENSITIVITY_ALPHA, PRIMARY_GRID_HOURS, SENSITIVITY_KAPPA),
    ("cap120_grid12", SENSITIVITY_CAP_HOURS, PRIMARY_ALPHA, SENSITIVITY_GRID_HOURS, SENSITIVITY_KAPPA),
    ("alpha90_grid12", PRIMARY_CAP_HOURS, SENSITIVITY_ALPHA, SENSITIVITY_GRID_HOURS, SENSITIVITY_KAPPA),
    (
        "cap120_alpha90_grid12",
        SENSITIVITY_CAP_HOURS,
        SENSITIVITY_ALPHA,
        SENSITIVITY_GRID_HOURS,
        SENSITIVITY_KAPPA,
    ),
)
MASS_TOL = 1e-12
LP_TOL = 1e-5
TRANSITION_COLUMNS = [
    "case_id",
    "origin_activity",
    "destination_activity",
    "transition",
    "origin_timestamp",
    "destination_timestamp",
    "elapsed_hours",
    "block_id",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parents[1]
DATASET_DIR = PROJECT_ROOT / "01_DATASET" / "02_EXTERNAL_VALIDATION_BPI2017"
RESULTS_DIR = PROJECT_ROOT / "04_ANALYSIS_RESULTS" / "08_PHASE4_EXTERNAL_VALIDATION_BPI2017"
DATASET_FILENAME = "BPI Challenge 2017.xes.gz"
RAW_METADATA_FILENAME = "figshare_12696884_metadata_raw.json"
SHA_FILENAME = f"{DATASET_FILENAME}.sha256.txt"


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", na_rep="")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dataset() -> tuple[Path, dict[str, Any]]:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = DATASET_DIR / RAW_METADATA_FILENAME
    if metadata_path.exists():
        raw_metadata = metadata_path.read_bytes()
    else:
        try:
            with urllib.request.urlopen(FIGSHARE_API_URL, timeout=60) as response:
                raw_metadata = response.read()
            metadata_path.write_bytes(raw_metadata)
        except Exception as exc:
            raise RuntimeError(
                "The official Figshare metadata could not be downloaded. "
                "Please manually download BPI Challenge 2017.xes.gz from "
                "https://ndownloader.figshare.com/files/24044117 and place it in "
                f"{DATASET_DIR}."
            ) from exc

    try:
        metadata = json.loads(raw_metadata.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("The saved Figshare metadata is not valid UTF-8 JSON.") from exc

    files = metadata.get("files", [])
    file_record = next(
        (
            item
            for item in files
            if str(item.get("name", "")).lower() == DATASET_FILENAME.lower()
            and str(item.get("download_url", ""))
        ),
        None,
    )
    if file_record is None:
        file_record = next(
            (
                item
                for item in files
                if str(item.get("name", "")).lower().endswith((".xes.gz", ".xes"))
                and str(item.get("download_url", ""))
            ),
            None,
        )
    if file_record is None:
        raise RuntimeError(
            "The official Figshare metadata does not expose an XES/XES.GZ file. "
            "Download the official BPI Challenge 2017 XES file manually and place it in "
            f"{DATASET_DIR}."
        )

    target = DATASET_DIR / str(file_record["name"])
    if target.name != DATASET_FILENAME:
        target = DATASET_DIR / DATASET_FILENAME
    expected_size = int(file_record.get("size", 0))
    if not target.exists() or (expected_size and target.stat().st_size != expected_size):
        temporary = DATASET_DIR / f"{target.name}.download"
        try:
            with urllib.request.urlopen(str(file_record["download_url"]), timeout=120) as response:
                with temporary.open("wb") as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            if expected_size and temporary.stat().st_size != expected_size:
                raise RuntimeError(
                    f"Downloaded file size {temporary.stat().st_size} does not match Figshare size {expected_size}."
                )
            temporary.replace(target)
        except Exception as exc:
            if temporary.exists():
                temporary.unlink()
            raise RuntimeError(
                "The official Figshare dataset could not be downloaded. "
                "Please manually download BPI Challenge 2017.xes.gz from "
                f"{file_record['download_url']} and place it in {DATASET_DIR}."
            ) from exc

    actual_size = target.stat().st_size
    if expected_size and actual_size != expected_size:
        raise RuntimeError(
            f"The local dataset size {actual_size} does not match the official size {expected_size}."
        )
    digest = sha256_file(target)
    (DATASET_DIR / SHA_FILENAME).write_text(f"{digest}  {target.name}\n", encoding="utf-8")

    provenance = {
        "figshare_article_id": FIGSHARE_ARTICLE_ID,
        "figshare_api_url": FIGSHARE_API_URL,
        "official_page": OFFICIAL_PAGE,
        "title": metadata.get("title"),
        "doi": metadata.get("doi"),
        "citation": metadata.get("citation"),
        "time_coverage": next(
            (
                item.get("value")
                for item in metadata.get("custom_fields", [])
                if item.get("name") == "Time coverage"
            ),
            None,
        ),
        "license_name": (metadata.get("license") or {}).get("name"),
        "license_url": (metadata.get("license") or {}).get("url"),
        "metadata_json_filename": metadata_path.name,
        "file_name": target.name,
        "file_size_bytes": actual_size,
        "official_download_url": file_record.get("download_url"),
        "figshare_file_id": file_record.get("id"),
        "official_supplied_md5": file_record.get("supplied_md5"),
        "official_computed_md5": file_record.get("computed_md5"),
        "sha256": digest,
        "sha256_filename": SHA_FILENAME,
        "original_file_preserved": True,
    }
    return target.resolve(), provenance


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return None
    if parsed is None or pd.isna(parsed):
        return None
    return pd.Timestamp(parsed)


def event_attributes(event_element: ET.Element) -> dict[str, str]:
    attributes: dict[str, str] = {}
    for child in list(event_element):
        key = child.attrib.get("key")
        value = child.attrib.get("value")
        if key is not None:
            attributes[str(key)] = "" if value is None else str(value)
    return attributes


def empty_parse_stats() -> dict[str, int]:
    return {
        "trace_count": 0,
        "raw_event_count": 0,
        "complete_event_count": 0,
        "excluded_non_complete_event_count": 0,
        "complete_events_missing_required_fields": 0,
        "complete_events_missing_case_id": 0,
        "complete_events_missing_activity": 0,
        "complete_events_invalid_timestamp": 0,
        "valid_complete_event_count": 0,
        "pair_candidates_before_negative_filter": 0,
        "negative_elapsed_rows_removed": 0,
        "final_direct_follow_rows": 0,
        "timestamp_tie_pairs": 0,
        "timestamp_tie_cases": 0,
        "timestamp_tie_groups": 0,
        "extracted_rows": 0,
    }


def parse_xes_origin_window(
    xes_path: Path,
    origin_start: pd.Timestamp,
    origin_end: pd.Timestamp,
    stage: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Stream one origin window and retain only its direct-follow rows."""
    stats = empty_parse_stats()
    rows: list[dict[str, Any]] = []
    block_seconds = H1_BLOCK_DAYS * 24 * 3600
    opener = gzip.open if xes_path.name.lower().endswith(".gz") else open
    with opener(xes_path, "rb") as handle:
        try:
            iterator = ET.iterparse(handle, events=("end",))
            for _, element in iterator:
                if local_name(element.tag) != "trace":
                    continue
                stats["trace_count"] += 1
                case_id: str | None = None
                event_elements: list[ET.Element] = []
                for child in list(element):
                    child_name = local_name(child.tag)
                    if child_name == "event":
                        event_elements.append(child)
                    elif child_name == "string" and child.attrib.get("key") == "concept:name":
                        value = child.attrib.get("value")
                        if value is not None and str(value).strip() != "":
                            case_id = str(value)

                complete_events: list[dict[str, Any]] = []
                for original_order, event_element in enumerate(event_elements):
                    stats["raw_event_count"] += 1
                    attributes = event_attributes(event_element)
                    lifecycle = attributes.get("lifecycle:transition", "")
                    if lifecycle.strip().casefold() != "complete":
                        stats["excluded_non_complete_event_count"] += 1
                        continue
                    stats["complete_event_count"] += 1
                    activity = attributes.get("concept:name", "")
                    timestamp = parse_timestamp(attributes.get("time:timestamp"))
                    missing_required = False
                    if case_id is None:
                        stats["complete_events_missing_case_id"] += 1
                        missing_required = True
                    if activity is None or str(activity).strip() == "":
                        stats["complete_events_missing_activity"] += 1
                        missing_required = True
                    if missing_required:
                        stats["complete_events_missing_required_fields"] += 1
                        continue
                    if timestamp is None:
                        stats["complete_events_invalid_timestamp"] += 1
                        continue
                    complete_events.append(
                        {
                            "case_id": case_id,
                            "activity": str(activity),
                            "timestamp": timestamp,
                            "original_order": original_order,
                        }
                    )

                stats["valid_complete_event_count"] += len(complete_events)
                complete_events.sort(key=lambda item: (item["timestamp"].value, item["original_order"]))
                tie_pairs_in_trace = 0
                tie_groups_in_trace = 0
                index = 0
                while index < len(complete_events):
                    next_index = index + 1
                    while next_index < len(complete_events) and complete_events[next_index]["timestamp"] == complete_events[index]["timestamp"]:
                        next_index += 1
                    group_size = next_index - index
                    if group_size > 1:
                        tie_groups_in_trace += 1
                        tie_pairs_in_trace += group_size - 1
                    index = next_index
                stats["timestamp_tie_pairs"] += tie_pairs_in_trace
                stats["timestamp_tie_groups"] += tie_groups_in_trace
                if tie_pairs_in_trace:
                    stats["timestamp_tie_cases"] += 1

                for origin, destination in zip(complete_events, complete_events[1:]):
                    stats["pair_candidates_before_negative_filter"] += 1
                    elapsed_hours = (destination["timestamp"] - origin["timestamp"]).total_seconds() / 3600.0
                    if elapsed_hours < 0:
                        stats["negative_elapsed_rows_removed"] += 1
                        continue
                    if origin_start <= origin["timestamp"] < origin_end:
                        block_id = int(
                            (origin["timestamp"] - H1_START).total_seconds() // block_seconds
                        )
                        rows.append(
                            {
                                "case_id": origin["case_id"],
                                "origin_activity": origin["activity"],
                                "destination_activity": destination["activity"],
                                "transition": f"{origin['activity']} -> {destination['activity']}",
                                "origin_timestamp": origin["timestamp"],
                                "destination_timestamp": destination["timestamp"],
                                "elapsed_hours": float(elapsed_hours),
                                "block_id": block_id,
                            }
                        )
            stats["final_direct_follow_rows"] = len(rows)
            stats["extracted_rows"] = len(rows)
        except ET.ParseError as exc:
            raise RuntimeError(f"The official XES file could not be parsed during {stage} extraction.") from exc
    frame = pd.DataFrame(rows, columns=TRANSITION_COLUMNS)
    if not frame.empty:
        frame["origin_timestamp"] = pd.to_datetime(frame["origin_timestamp"], utc=True)
        frame["destination_timestamp"] = pd.to_datetime(frame["destination_timestamp"], utc=True)
    return frame, stats


def compare_parse_stats(first: dict[str, int], second: dict[str, int]) -> None:
    keys = [
        key
        for key in first
        if key not in {"extracted_rows", "final_direct_follow_rows"}
    ]
    differences = {key: (first[key], second[key]) for key in keys if first[key] != second[key]}
    if differences:
        raise RuntimeError(f"The two XES passes produced inconsistent preprocessing counts: {differences}")


def clipped(values: Iterable[float], cap_hours: int) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return np.clip(array, 0.0, float(cap_hours))


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
    mapped = np.clip(
        np.rint(np.asarray(values, dtype=float) / step_hours) * step_hours,
        0.0,
        float(cap_hours),
    )
    positions = np.searchsorted(grid, mapped).astype(int)
    counts = np.bincount(positions, minlength=len(grid)).astype(float)
    if counts.sum() <= 0:
        raise RuntimeError("A transition has no finite elapsed-time observations.")
    return grid, counts / counts.sum()


def cvar_from_grid_probabilities(grid: np.ndarray, probabilities: np.ndarray, alpha: float) -> float:
    best = float("inf")
    for threshold in grid:
        hinge = np.maximum(grid - float(threshold), 0.0)
        candidate = float(threshold + np.dot(probabilities, hinge) / (1.0 - alpha))
        best = min(best, candidate)
    return best


def build_w1_constraints(probabilities: np.ndarray, step_hours: float):
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
        template = build_w1_constraints(probabilities, float(grid[1] - grid[0]))
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
            raise RuntimeError(f"Worst-case CVaR LP failed at threshold {threshold}: {result.message}")
        candidates.append(float(threshold - result.fun / (1.0 - alpha)))
    return float(min(candidates))


def fast_worst_case_cvar(
    grid: np.ndarray,
    probabilities: np.ndarray,
    alpha: float,
    epsilon: float,
) -> float:
    order = np.argsort(-grid)
    sorted_probabilities = probabilities[order]
    costs = np.maximum(float(grid[-1]) - grid[order], 0.0)
    best = float("inf")
    for threshold in grid:
        payoff = np.maximum(grid - float(threshold), 0.0)
        base = float(np.dot(probabilities, payoff))
        gains = np.maximum(float(grid[-1]) - float(threshold), 0.0) - payoff
        sorted_gains = gains[order]
        eligible = (sorted_probabilities > 0) & (costs > 0) & (sorted_gains > 0)
        if epsilon <= 0 or not np.any(eligible):
            addition = 0.0
        else:
            eligible_costs = sorted_probabilities[eligible] * costs[eligible]
            eligible_gains = sorted_probabilities[eligible] * sorted_gains[eligible]
            cumulative_costs = np.cumsum(eligible_costs)
            budget = float(epsilon)
            if budget >= float(cumulative_costs[-1]):
                addition = float(eligible_gains.sum())
            else:
                position = int(np.searchsorted(cumulative_costs, budget, side="right"))
                previous_cost = float(cumulative_costs[position - 1]) if position else 0.0
                previous_gain = float(eligible_gains[:position].sum()) if position else 0.0
                residual = budget - previous_cost
                addition = previous_gain + residual * float(eligible_gains[position] / eligible_costs[position])
        candidate = float(threshold + (base + addition) / (1.0 - alpha))
        best = min(best, candidate)
    return best


def saturation_threshold_from_grid(
    grid: np.ndarray,
    probabilities: np.ndarray,
    alpha: float,
    cap_hours: int,
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
        moved = min(float(probabilities[index]), remaining)
        cost += moved * (float(cap_hours) - float(grid[index]))
        remaining -= moved
        if remaining <= MASS_TOL:
            break
    if remaining > 1e-10:
        raise RuntimeError("The discrete saturation mass could not be transported to the cap.")
    return m_cap, delta, float(cost)


def calibration_for_frame(frame: pd.DataFrame, cap_hours: int, step_hours: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for transition, group in frame.groupby("transition", sort=True):
        values = clipped(group["elapsed_hours"].to_numpy(float), cap_hours)
        block_rows: list[dict[str, Any]] = []
        for block_id, block in group.groupby("block_id", sort=True):
            block_values = clipped(block["elapsed_hours"].to_numpy(float), cap_hours)
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
                "epsilon_temporal_hours": float(np.quantile(distances, 0.90)) if len(distances) else float("nan"),
                "epsilon_max_hours": float(np.max(distances)) if len(distances) else float("nan"),
                "cap_hours": int(cap_hours),
                "grid_step_hours": int(step_hours),
                "candidate_h1_only": bool(len(values) >= MIN_H1_OBSERVATIONS and len(block_rows) >= MIN_VALID_BLOCKS),
            }
        )
    return pd.DataFrame(rows)


def build_all_calibrations(h1: pd.DataFrame, candidate_transitions: set[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for config, cap_hours, alpha, step_hours, kappa in SENSITIVITY_CONFIGS:
        frame = calibration_for_frame(h1, cap_hours, step_hours)
        frame = frame[frame["transition"].astype(str).isin(candidate_transitions)].copy()
        frame["calibration_role"] = config
        frame["alpha"] = float(alpha)
        frame["kappa_sa"] = float(kappa)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(["calibration_role", "transition"]).reset_index(drop=True)


def transition_vectors(frame: pd.DataFrame, candidate_transitions: set[str]) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for transition in sorted(candidate_transitions):
        vector = pd.to_numeric(
            frame.loc[frame["transition"].astype(str) == transition, "elapsed_hours"],
            errors="coerce",
        ).to_numpy(float)
        vector = vector[np.isfinite(vector) & (vector >= 0)]
        if len(vector) == 0:
            raise RuntimeError(f"The H1 sample contains no observations for {transition!r}.")
        values[transition] = vector
    return values


def primary_scores(
    h1: pd.DataFrame,
    candidates: pd.DataFrame,
    calibration: pd.DataFrame,
    vectors: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibration_map = calibration.set_index("transition")
    total_h1 = len(h1)
    score_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for transition in sorted(vectors):
        vector = vectors[transition]
        capped = clipped(vector, PRIMARY_CAP_HOURS)
        grid, probabilities = grid_probabilities(capped, PRIMARY_CAP_HOURS, PRIMARY_GRID_HOURS)
        template = build_w1_constraints(probabilities, PRIMARY_GRID_HOURS)
        epsilon_temporal = float(calibration_map.loc[transition, "epsilon_temporal_hours"])
        epsilon_sat = saturation_threshold_from_grid(
            grid, probabilities, PRIMARY_ALPHA, PRIMARY_CAP_HOURS
        )[2]
        m_cap, delta_mass, _ = saturation_threshold_from_grid(
            grid, probabilities, PRIMARY_ALPHA, PRIMARY_CAP_HOURS
        )
        grid_empirical_cvar = cvar_from_grid_probabilities(grid, probabilities, PRIMARY_ALPHA)
        empirical_cvar = empirical_cvar_array(capped, PRIMARY_ALPHA)
        original_dro = worst_case_cvar(grid, probabilities, PRIMARY_ALPHA, epsilon_temporal, template)
        lp_zero = worst_case_cvar(grid, probabilities, PRIMARY_ALPHA, 0.0, template)
        lp_sat = worst_case_cvar(grid, probabilities, PRIMARY_ALPHA, epsilon_sat, template)
        near_epsilon = 0.99 * epsilon_sat
        lp_near = worst_case_cvar(grid, probabilities, PRIMARY_ALPHA, near_epsilon, template)
        frequency = float(len(vector) / total_h1) if total_h1 else 0.0
        mean_value = float(np.mean(capped))
        mean_score = frequency * mean_value
        tail_score = frequency * float(empirical_cvar)
        original_score = frequency * original_dro
        if epsilon_sat <= MASS_TOL:
            sa_epsilon = 0.0
        else:
            sa_epsilon = min(epsilon_temporal, PRIMARY_KAPPA * epsilon_sat)
        sa_dro = worst_case_cvar(grid, probabilities, PRIMARY_ALPHA, sa_epsilon, template)
        sa_score = frequency * sa_dro
        empirical_saturated = bool(abs(grid_empirical_cvar - PRIMARY_CAP_HOURS) <= LP_TOL)
        original_predicted_saturated = bool(
            True if epsilon_sat <= MASS_TOL else epsilon_temporal >= epsilon_sat - LP_TOL
        )
        original_observed_saturated = bool(abs(original_dro - PRIMARY_CAP_HOURS) <= LP_TOL)
        original_induced_saturated = bool(original_predicted_saturated and not empirical_saturated)
        original_total_saturated = bool(empirical_saturated or original_induced_saturated)
        sa_observed_saturated = bool(abs(sa_dro - PRIMARY_CAP_HOURS) <= LP_TOL)
        sa_induced_saturated = bool(sa_observed_saturated and not empirical_saturated)
        sa_total_saturated = bool(empirical_saturated or sa_induced_saturated)
        validation_rows.append(
            {
                "transition": transition,
                "epsilon_temporal": epsilon_temporal,
                "epsilon_zero": 0.0,
                "epsilon_sat": epsilon_sat,
                "epsilon_near_sat": near_epsilon,
                "grid_empirical_cvar95": grid_empirical_cvar,
                "lp_cvar_at_epsilon_zero": lp_zero,
                "lp_zero_absolute_error_hours": abs(lp_zero - grid_empirical_cvar),
                "lp_cvar_at_epsilon_sat": lp_sat,
                "lp_sat_absolute_error_to_cap_hours": abs(lp_sat - PRIMARY_CAP_HOURS),
                "lp_cvar_at_0_99_epsilon_sat": lp_near,
                "near_threshold_required": bool(epsilon_sat > MASS_TOL),
                "near_threshold_below_cap": bool(
                    epsilon_sat <= MASS_TOL or lp_near < PRIMARY_CAP_HOURS - LP_TOL
                ),
                "epsilon_zero_pass": bool(abs(lp_zero - grid_empirical_cvar) <= LP_TOL),
                "epsilon_sat_pass": bool(abs(lp_sat - PRIMARY_CAP_HOURS) <= LP_TOL),
                "near_threshold_pass": bool(
                    epsilon_sat <= MASS_TOL or lp_near < PRIMARY_CAP_HOURS - LP_TOL
                ),
                "original_saturation_classification_match": bool(
                    original_predicted_saturated == original_observed_saturated
                ),
                "validation_pass": bool(
                    abs(lp_zero - grid_empirical_cvar) <= LP_TOL
                    and abs(lp_sat - PRIMARY_CAP_HOURS) <= LP_TOL
                    and (epsilon_sat <= MASS_TOL or lp_near < PRIMARY_CAP_HOURS - LP_TOL)
                ),
            }
        )
        score_rows.append(
            {
                "transition": transition,
                "h1_observations": int(len(vector)),
                "h1_frequency_share": frequency,
                "h1_mean_capped_elapsed_hours": mean_value,
                "h1_empirical_cvar95_hours": float(empirical_cvar),
                "h1_grid_empirical_cvar95_hours": float(grid_empirical_cvar),
                "epsilon_temporal_hours": epsilon_temporal,
                "m_cap": m_cap,
                "delta_mass_to_cap": delta_mass,
                "epsilon_sat_hours": epsilon_sat,
                "saturation_ratio": float(epsilon_temporal / epsilon_sat) if epsilon_sat > MASS_TOL else float("nan"),
                "frequency_score": frequency,
                "mean_burden_score": mean_score,
                "tail_score": tail_score,
                "original_dro_epsilon_hours": epsilon_temporal,
                "original_dro_cvar95_hours": original_dro,
                "original_dro_score": original_score,
                "sa_dro_kappa": PRIMARY_KAPPA,
                "sa_dro_epsilon_hours": sa_epsilon,
                "sa_dro_cvar95_hours": sa_dro,
                "sa_dro_score": sa_score,
                "empirical_saturation": empirical_saturated,
                "original_dro_predicted_saturated": original_predicted_saturated,
                "original_dro_observed_saturated": original_observed_saturated,
                "original_dro_induced_saturation": original_induced_saturated,
                "original_dro_total_saturated": original_total_saturated,
                "sa_dro_observed_saturated": sa_observed_saturated,
                "sa_dro_induced_saturation": sa_induced_saturated,
                "sa_dro_total_saturated": sa_total_saturated,
                "original_dro_saturation_mismatch": bool(
                    original_predicted_saturated != original_observed_saturated
                ),
                "candidate_h1_only": True,
            }
        )
    scores = pd.DataFrame(score_rows).sort_values(["original_dro_score", "transition"], ascending=[False, True]).reset_index(drop=True)
    validation = pd.DataFrame(validation_rows).sort_values("transition").reset_index(drop=True)
    return scores, validation


def select_top(scores: pd.DataFrame, method: str, k: int) -> list[str]:
    column = TRANSITION_SCORE_COLUMNS[method]
    return (
        scores.sort_values([column, "transition"], ascending=[False, True])
        .head(k)["transition"]
        .astype(str)
        .tolist()
    )


def build_primary_selections(scores: pd.DataFrame) -> dict[tuple[str, int], list[str]]:
    return {
        (method, k): select_top(scores, method, k)
        for method in METHODS
        for k in PORTFOLIO_KS
    }


def build_selection_frame(
    scores: pd.DataFrame,
    selections: dict[tuple[str, int], list[str]],
    manifest_sha256: str,
) -> pd.DataFrame:
    indexed = scores.set_index("transition")
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        score_column = TRANSITION_SCORE_COLUMNS[method]
        for k in PORTFOLIO_KS:
            for rank, transition in enumerate(selections[(method, k)], start=1):
                row = indexed.loc[transition]
                rows.append(
                    {
                        "method": method,
                        "K": int(k),
                        "rank": int(rank),
                        "transition": transition,
                        "score_column": score_column,
                        "h1_score": float(row[score_column]),
                        "h1_frequency_share": float(row["h1_frequency_share"]),
                        "h1_mean_capped_elapsed_hours": float(row["h1_mean_capped_elapsed_hours"]),
                        "h1_empirical_cvar95_hours": float(row["h1_empirical_cvar95_hours"]),
                        "h1_original_dro_cvar95_hours": float(row["original_dro_cvar95_hours"]),
                        "h1_sa_dro_cvar95_hours": float(row["sa_dro_cvar95_hours"]),
                        "frozen_before_q3": True,
                        "selection_manifest_sha256": manifest_sha256,
                    }
                )
    return pd.DataFrame(rows)


def build_sensitivity_results(
    h1: pd.DataFrame,
    scores: pd.DataFrame,
    candidate_transitions: set[str],
    calibration_table: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, str, int], list[str]]]:
    vectors = transition_vectors(h1, candidate_transitions)
    rows: list[dict[str, Any]] = []
    selections: dict[tuple[str, str, int], list[str]] = {}
    for config, cap_hours, alpha, step_hours, kappa in SENSITIVITY_CONFIGS:
        calibration = calibration_table[calibration_table["calibration_role"] == config].set_index("transition")
        config_rows: list[dict[str, Any]] = []
        for transition in sorted(candidate_transitions):
            vector = vectors[transition]
            capped = clipped(vector, cap_hours)
            grid, probabilities = grid_probabilities(capped, cap_hours, step_hours)
            epsilon_temporal = float(calibration.loc[transition, "epsilon_temporal_hours"])
            m_cap, delta_mass, epsilon_sat = saturation_threshold_from_grid(
                grid, probabilities, alpha, cap_hours
            )
            empirical_cvar = cvar_from_grid_probabilities(grid, probabilities, alpha)
            original_cvar = fast_worst_case_cvar(
                grid, probabilities, alpha, epsilon_temporal
            )
            sa_epsilon = 0.0 if epsilon_sat <= MASS_TOL else min(epsilon_temporal, kappa * epsilon_sat)
            sa_cvar = fast_worst_case_cvar(grid, probabilities, alpha, sa_epsilon)
            frequency = float(len(vector) / len(h1)) if len(h1) else 0.0
            config_rows.append(
                {
                    "transition": transition,
                    "frequency_score": frequency,
                    "mean_burden_score": frequency * float(np.mean(capped)),
                    "tail_score": frequency * float(empirical_cvar),
                    "original_dro_score": frequency * float(original_cvar),
                    "sa_dro_score": frequency * float(sa_cvar),
                    "epsilon_temporal_hours": epsilon_temporal,
                    "m_cap": m_cap,
                    "delta_mass_to_cap": delta_mass,
                    "epsilon_sat_hours": epsilon_sat,
                    "original_dro_total_saturated": bool(abs(original_cvar - cap_hours) <= LP_TOL),
                    "sa_dro_total_saturated": bool(abs(sa_cvar - cap_hours) <= LP_TOL),
                    "empirical_saturation": bool(abs(empirical_cvar - cap_hours) <= LP_TOL),
                }
            )
        config_scores = pd.DataFrame(config_rows)
        for method in METHODS:
            for k in PORTFOLIO_KS:
                selected = select_top(config_scores, method, k)
                selections[(config, method, k)] = selected
                chosen_rows = config_scores[config_scores["transition"].isin(selected)]
                rows.append(
                    {
                        "configuration": config,
                        "cap_hours": int(cap_hours),
                        "alpha": float(alpha),
                        "grid_step_hours": int(step_hours),
                        "kappa_sa": float(kappa),
                        "method": method,
                        "K": int(k),
                        "candidate_count": int(len(config_scores)),
                        "empirical_saturated_count": int(config_scores["empirical_saturation"].sum()),
                        "original_dro_saturated_count": int(config_scores["original_dro_total_saturated"].sum()),
                        "sa_dro_saturated_count": int(config_scores["sa_dro_total_saturated"].sum()),
                        "selected_empirical_saturated_count": int(chosen_rows["empirical_saturation"].sum()),
                        "selected_transitions": " || ".join(selected),
                        "selection_h1_only": True,
                    }
                )
    return pd.DataFrame(rows), selections


def canonical_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def create_freeze_manifest(
    candidates: pd.DataFrame,
    scores: pd.DataFrame,
    primary_selections: dict[tuple[str, int], list[str]],
    sensitivity_selections: dict[tuple[str, str, int], list[str]],
) -> tuple[dict[str, Any], str]:
    primary_payload = {
        f"{method}|K={k}": primary_selections[(method, k)]
        for method in METHODS
        for k in PORTFOLIO_KS
    }
    sensitivity_payload = {
        f"{config}|{method}|K={k}": sensitivity_selections[(config, method, k)]
        for config, _, _, _, _ in SENSITIVITY_CONFIGS
        for method in METHODS
        for k in PORTFOLIO_KS
    }
    payload = {
        "stage": "portfolio freeze before Q3 external evaluation",
        "dataset": "BPI Challenge 2017",
        "candidate_count_h1_only": int(len(candidates)),
        "candidate_transitions": sorted(candidates["transition"].astype(str).tolist()),
        "primary_parameters": {
            "cap_hours": PRIMARY_CAP_HOURS,
            "alpha": PRIMARY_ALPHA,
            "grid_step_hours": PRIMARY_GRID_HOURS,
            "kappa_sa": PRIMARY_KAPPA,
            "portfolio_Ks": list(PORTFOLIO_KS),
        },
        "primary_selections": primary_payload,
        "sensitivity_selections": sensitivity_payload,
        "q3_accessed_before_freeze": False,
        "candidate_scores_sha256": canonical_sha256(
            {"scores": scores.sort_values("transition").to_dict(orient="records")}
        ),
    }
    digest = canonical_sha256(payload)
    manifest = {**payload, "checksum_algorithm": "SHA-256", "manifest_sha256": digest}
    return manifest, digest


def q3_universe_metrics(q3: pd.DataFrame, cap_hours: int, alpha: float) -> pd.DataFrame:
    work = q3.copy()
    work["capped_elapsed_hours"] = np.clip(work["elapsed_hours"].to_numpy(float), 0.0, float(cap_hours))
    rows: list[dict[str, Any]] = []
    total = len(work)
    for transition, group in work.groupby("transition", sort=True):
        vector = group["capped_elapsed_hours"].to_numpy(float)
        frequency = float(len(vector) / total) if total else 0.0
        mean_value = float(np.mean(vector)) if len(vector) else 0.0
        tail_value = float(empirical_cvar_array(vector, alpha)) if len(vector) else 0.0
        rows.append(
            {
                "transition": str(transition),
                "q3_observations": int(len(vector)),
                "q3_frequency_share": frequency,
                "q3_mean_capped_elapsed_hours": mean_value,
                "q3_empirical_cvar95_hours": tail_value,
                "q3_mean_burden": frequency * mean_value,
                "q3_tail_burden": frequency * tail_value,
                "q3_capped_elapsed_sum_hours": float(vector.sum()),
            }
        )
    return pd.DataFrame(rows)


def safe_correlation(x: pd.Series, y: pd.Series, method: str) -> float:
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    result = spearmanr(x, y) if method == "spearman" else kendalltau(x, y)
    value = float(result.statistic)
    return value if np.isfinite(value) else float("nan")


def evaluate_q3(
    scores: pd.DataFrame,
    q3: pd.DataFrame,
    selections: dict[tuple[str, int], list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe = q3_universe_metrics(q3, PRIMARY_CAP_HOURS, PRIMARY_ALPHA)
    lookup = universe.set_index("transition")
    all_mean = float(universe["q3_mean_burden"].sum())
    all_tail = float(universe["q3_tail_burden"].sum())
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        score_column = TRANSITION_SCORE_COLUMNS[method]
        rank_frame = scores[["transition", score_column]].merge(
            universe[["transition", "q3_mean_burden", "q3_tail_burden"]],
            on="transition",
            how="inner",
        )
        mean_spearman = safe_correlation(rank_frame[score_column], rank_frame["q3_mean_burden"], "spearman")
        mean_kendall = safe_correlation(rank_frame[score_column], rank_frame["q3_mean_burden"], "kendall")
        tail_spearman = safe_correlation(rank_frame[score_column], rank_frame["q3_tail_burden"], "spearman")
        tail_kendall = safe_correlation(rank_frame[score_column], rank_frame["q3_tail_burden"], "kendall")
        for k in PORTFOLIO_KS:
            chosen = selections[(method, k)]
            chosen_rows = universe[universe["transition"].isin(chosen)]
            selected_mean = float(chosen_rows["q3_mean_burden"].sum())
            selected_tail = float(chosen_rows["q3_tail_burden"].sum())
            mean_oracle = universe.sort_values(["q3_mean_burden", "transition"], ascending=[False, True]).head(k)
            tail_oracle = universe.sort_values(["q3_tail_burden", "transition"], ascending=[False, True]).head(k)
            mean_oracle_set = set(mean_oracle["transition"].astype(str))
            tail_oracle_set = set(tail_oracle["transition"].astype(str))
            chosen_set = set(chosen)
            selected_mean_share = 100.0 * selected_mean / all_mean if all_mean else float("nan")
            selected_tail_share = 100.0 * selected_tail / all_tail if all_tail else float("nan")
            mean_oracle_share = 100.0 * float(mean_oracle["q3_mean_burden"].sum()) / all_mean if all_mean else float("nan")
            tail_oracle_share = 100.0 * float(tail_oracle["q3_tail_burden"].sum()) / all_tail if all_tail else float("nan")
            rows.append(
                {
                    "method": method,
                    "K": int(k),
                    "q3_observations": int(len(q3)),
                    "q3_transition_universe": int(len(universe)),
                    "q3_candidate_coverage_count": int(sum(transition in lookup.index for transition in chosen)),
                    "q3_captured_mean_burden_share_pct": selected_mean_share,
                    "q3_captured_tail_burden_share_pct": selected_tail_share,
                    "q3_mean_oracle_share_pct": mean_oracle_share,
                    "q3_tail_oracle_share_pct": tail_oracle_share,
                    "q3_mean_burden_regret_pp": mean_oracle_share - selected_mean_share,
                    "q3_tail_burden_regret_pp": tail_oracle_share - selected_tail_share,
                    "jaccard_vs_q3_mean_oracle": len(chosen_set & mean_oracle_set) / len(chosen_set | mean_oracle_set),
                    "jaccard_vs_q3_tail_oracle": len(chosen_set & tail_oracle_set) / len(chosen_set | tail_oracle_set),
                    "spearman_h1_vs_q3_mean_burden": mean_spearman,
                    "kendall_h1_vs_q3_mean_burden": mean_kendall,
                    "spearman_h1_vs_q3_tail_burden": tail_spearman,
                    "kendall_h1_vs_q3_tail_burden": tail_kendall,
                    "ranking_n": int(len(rank_frame)),
                    "ranking_defined": bool(np.isfinite(mean_spearman) and np.isfinite(mean_kendall)),
                    "selected_transitions": " || ".join(chosen),
                    "q3_mean_oracle_transitions": " || ".join(mean_oracle["transition"].astype(str).tolist()),
                    "q3_tail_oracle_transitions": " || ".join(tail_oracle["transition"].astype(str).tolist()),
                }
            )
    return pd.DataFrame(rows), universe


def add_week_start(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    origin = pd.to_datetime(result["origin_timestamp"], utc=True)
    result["week_start"] = origin.dt.normalize() - pd.to_timedelta(origin.dt.dayofweek, unit="D")
    return result


def q3_bootstrap_differences(
    q3: pd.DataFrame,
    selections: dict[tuple[str, int], list[str]],
    reps: int = Q3_BOOTSTRAP_REPS,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    work = add_week_start(q3)
    work["capped_elapsed_hours"] = np.clip(work["elapsed_hours"].to_numpy(float), 0.0, float(PRIMARY_CAP_HOURS))
    weeks = sorted(work["week_start"].dropna().unique())
    if len(weeks) < 2:
        raise RuntimeError("The Q3 holdout does not contain at least two nonempty weekly blocks.")
    totals = np.asarray(
        [float(work.loc[work["week_start"] == week, "capped_elapsed_hours"].sum()) for week in weeks],
        dtype=float,
    )
    method_values: dict[str, np.ndarray] = {}
    for method in METHODS:
        chosen = set(selections[(method, 5)])
        method_values[method] = np.asarray(
            [
                float(work.loc[(work["week_start"] == week) & work["transition"].isin(chosen), "capped_elapsed_hours"].sum())
                for week in weeks
            ],
            dtype=float,
        )
    rng = np.random.default_rng(SEED)
    share_samples: dict[str, np.ndarray] = {method: np.zeros(reps, dtype=float) for method in METHODS}
    for rep in range(reps):
        sampled = rng.integers(0, len(weeks), size=len(weeks))
        denominator = float(totals[sampled].sum())
        for method in METHODS:
            numerator = float(method_values[method][sampled].sum())
            share_samples[method][rep] = numerator / denominator if denominator else float("nan")
    rows: list[dict[str, Any]] = []
    for baseline in ("Original DRO", "Mean burden", "Empirical Tail", "Frequency"):
        differences = 100.0 * (share_samples["Saturation-Aware DRO"] - share_samples[baseline])
        ci_low = float(np.nanquantile(differences, 0.025))
        ci_high = float(np.nanquantile(differences, 0.975))
        rows.append(
            {
                "comparison": "SA-DRO minus " + baseline,
                "metric": "Q3 captured mean-burden share",
                "sa_dro_mean_share_pct": 100.0 * float(np.nanmean(share_samples["Saturation-Aware DRO"])),
                "baseline_mean_share_pct": 100.0 * float(np.nanmean(share_samples[baseline])),
                "difference_mean_pp": float(np.nanmean(differences)),
                "difference_median_pp": float(np.nanmedian(differences)),
                "ci95_low_pp": ci_low,
                "ci95_high_pp": ci_high,
                "ci_includes_zero": bool(ci_low <= 0.0 <= ci_high),
                "replicates": int(reps),
                "weekly_blocks": int(len(weeks)),
                "seed": SEED,
            }
        )
    return pd.DataFrame(rows), {"weeks": weeks, "shares": share_samples}


def h1_bootstrap_stability(
    h1: pd.DataFrame,
    scores: pd.DataFrame,
    selections: dict[tuple[str, int], list[str]],
    vectors: dict[str, np.ndarray],
    reps: int = H1_BOOTSTRAP_REPS,
) -> pd.DataFrame:
    candidates = sorted(vectors)
    block_ids = sorted(h1["block_id"].unique().tolist())
    block_values: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
    block_totals: dict[int, int] = {}
    for block_id, block in h1.groupby("block_id", sort=True):
        block_id_int = int(block_id)
        block_totals[block_id_int] = int(len(block))
        for transition, group in block.groupby("transition", sort=True):
            if str(transition) in vectors:
                block_values[block_id_int][str(transition)] = clipped(
                    group["elapsed_hours"].to_numpy(float), PRIMARY_CAP_HOURS
                )
    score_index = scores.set_index("transition")
    fixed_original = score_index["original_dro_cvar95_hours"].to_dict()
    fixed_sa = score_index["sa_dro_cvar95_hours"].to_dict()
    inclusion = {(method, transition): 0 for method in H1_STABILITY_METHODS for transition in candidates}
    jaccards = {method: [] for method in H1_STABILITY_METHODS}
    rng = np.random.default_rng(SEED)
    for _ in range(reps):
        sampled_blocks = rng.choice(block_ids, size=len(block_ids), replace=True)
        total = sum(block_totals[int(block_id)] for block_id in sampled_blocks)
        sample_rows: list[dict[str, Any]] = []
        for transition in candidates:
            pieces = [
                block_values[int(block_id)].get(transition, np.empty(0, dtype=float))
                for block_id in sampled_blocks
            ]
            vector = np.concatenate([piece for piece in pieces if len(piece)]) if any(len(piece) for piece in pieces) else np.empty(0)
            frequency = len(vector) / total if total else 0.0
            mean_score = frequency * float(np.mean(vector)) if len(vector) else 0.0
            tail_score = frequency * float(empirical_cvar_array(vector, PRIMARY_ALPHA)) if len(vector) else 0.0
            sample_rows.append(
                {
                    "transition": transition,
                    "mean_burden_score": mean_score,
                    "tail_score": tail_score,
                    "original_dro_score": frequency * float(fixed_original[transition]),
                    "sa_dro_score": frequency * float(fixed_sa[transition]),
                }
            )
        sample_scores = pd.DataFrame(sample_rows)
        for method in H1_STABILITY_METHODS:
            selected = set(select_top(sample_scores, method, 5))
            full = set(selections[(method, 5)])
            jaccards[method].append(len(selected & full) / len(selected | full) if selected | full else 1.0)
            for transition in candidates:
                inclusion[(method, transition)] += int(transition in selected)

    rows: list[dict[str, Any]] = []
    for method in H1_STABILITY_METHODS:
        probabilities = {transition: inclusion[(method, transition)] / reps for transition in candidates}
        rows.append(
            {
                "row_type": "summary",
                "method": method,
                "K": 5,
                "transition": "__SUMMARY__",
                "full_h1_selected": None,
                "inclusion_probability": None,
                "mean_jaccard_vs_full_h1": float(np.mean(jaccards[method])),
                "jaccard_p05": float(np.quantile(jaccards[method], 0.05)),
                "jaccard_p95": float(np.quantile(jaccards[method], 0.95)),
                "transitions_inclusion_probability_ge_0_8": int(
                    sum(value >= 0.8 for value in probabilities.values())
                ),
                "replicates": int(reps),
                "seed": SEED,
            }
        )
        for transition in candidates:
            rows.append(
                {
                    "row_type": "transition",
                    "method": method,
                    "K": 5,
                    "transition": transition,
                    "full_h1_selected": transition in selections[(method, 5)],
                    "inclusion_probability": probabilities[transition],
                    "mean_jaccard_vs_full_h1": None,
                    "jaccard_p05": None,
                    "jaccard_p95": None,
                    "transitions_inclusion_probability_ge_0_8": None,
                    "replicates": int(reps),
                    "seed": SEED,
                }
            )
    return pd.DataFrame(rows)




def write_report(
    provenance: dict[str, Any],
    h1_stats: dict[str, int],
    q3_stats: dict[str, int],
    h1: pd.DataFrame,
    q3: pd.DataFrame,
    candidates: pd.DataFrame,
    scores: pd.DataFrame,
    validation: pd.DataFrame,
    q3_eval: pd.DataFrame,
    bootstrap_rows: pd.DataFrame,
    stability: pd.DataFrame,
    sensitivity: pd.DataFrame,
    manifest: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    lines: list[str] = []
    lines.append("# Phase 4, External Validation on BPI Challenge 2017")
    lines.append("")
    lines.append("## Dataset and provenance")
    lines.append("")
    lines.append("The external event log is the official BPI Challenge 2017 dataset from Figshare article 12696884.")
    lines.append(f"Official page: `{OFFICIAL_PAGE}`.")
    lines.append(f"DOI: `{provenance.get('doi')}`.")
    lines.append(f"Time coverage in public metadata: `{provenance.get('time_coverage')}`.")
    lines.append(f"License: `{provenance.get('license_name')}`.")
    lines.append(f"Original file: `{provenance.get('file_name')}`.")
    lines.append(f"Official download URL: `{provenance.get('official_download_url')}`.")
    lines.append(f"File size: **{int(provenance['file_size_bytes']):,} bytes**.")
    lines.append(f"SHA-256: `{provenance.get('sha256')}`.")
    lines.append("")
    lines.append("## Preprocessing")
    lines.append("")
    lines.append("The primary preprocessing retained only events with `lifecycle:transition == COMPLETE`, after case-insensitive normalization. START and SCHEDULE events were excluded. The analytical label was `concept:name`, and the case identifier was the trace-level `concept:name`.")
    lines.append("Complete events were ordered chronologically within each case. The original XES order was used as the stable tie-break for equal timestamps. The outcome is inter-event elapsed time between two consecutive COMPLETE events.")
    lines.append("")
    lines.append("| Preprocessing item | Count |")
    lines.append("|---|---:|")
    for label, key in [
        ("Traces", "trace_count"),
        ("Raw events", "raw_event_count"),
        ("COMPLETE events", "complete_event_count"),
        ("Excluded non-COMPLETE events", "excluded_non_complete_event_count"),
        ("COMPLETE events missing required fields", "complete_events_missing_required_fields"),
        ("COMPLETE events missing case id", "complete_events_missing_case_id"),
        ("COMPLETE events missing activity", "complete_events_missing_activity"),
        ("COMPLETE events with invalid timestamp", "complete_events_invalid_timestamp"),
        ("Valid COMPLETE events", "valid_complete_event_count"),
        ("Direct-follow pairs before negative elapsed filter", "pair_candidates_before_negative_filter"),
        ("Rows removed for negative elapsed time", "negative_elapsed_rows_removed"),
        ("Final direct-follow rows", "final_direct_follow_rows"),
        ("Consecutive timestamp tie pairs", "timestamp_tie_pairs"),
        ("Cases containing timestamp ties", "timestamp_tie_cases"),
        ("Timestamp tie groups", "timestamp_tie_groups"),
    ]:
        lines.append(f"| {label} | {int(h1_stats[key]):,} |")
    lines.append("")
    lines.append("No activity duration was created from lifecycle pairs. No negative elapsed row was retained.")
    lines.append("")
    lines.append("## Temporal design and leakage control")
    lines.append("")
    lines.append("H1 was fixed as origins from 1 January 2016 inclusive to 1 July 2016 exclusive. Q3 was fixed as origins from 1 July 2016 inclusive to 1 October 2016 exclusive.")
    lines.append("Q3 was not accessed before portfolio freeze. Before freeze, the extraction returned H1-origin rows only, all candidate selection used H1 only, all epsilon calibration used H1 only, and kappa = 0.90 was fixed. The Q3-origin holdout was opened only after the frozen selection manifest was written and checksummed.")
    lines.append(f"Frozen portfolio manifest SHA-256: `{manifest['manifest_sha256']}`.")
    lines.append("")
    lines.append("## H1 feasibility gate")
    lines.append("")
    lines.append(f"H1 direct-follow rows: **{len(h1):,}**.")
    lines.append(f"Distinct H1 activities: **{int(pd.unique(pd.concat([h1['origin_activity'], h1['destination_activity']])).size)}**.")
    lines.append(f"Distinct H1 transitions: **{int(h1['transition'].nunique())}**.")
    lines.append(f"Frozen H1-only candidate transitions: **{len(candidates)}**.")
    lines.append("The fixed candidate rule was 500 H1 observations, at least 8 valid 14-day blocks, and at least 20 observations in each valid block. The feasibility gate passed before Q3 evaluation.")
    lines.append("")
    lines.append("## Primary H1 model")
    lines.append("")
    lines.append("The primary cap was 2160 hours, alpha = 0.95, and the grid step was 24 hours. Temporal ambiguity radius was the q90 of H1 block Wasserstein-1 distances from each transition to its pooled H1 distribution.")
    lines.append("The saturation-specific quantity is used as a bounded-support diagnostic. Original DRO uses the full temporal radius. SA-DRO uses epsilon_SA = min(epsilon_temporal, 0.90 * epsilon_sat) when epsilon_sat is positive, and epsilon_SA = 0 when epsilon_sat is zero. The value kappa = 0.90 was fixed before Q3.")
    lines.append(f"Candidate transitions: **{len(scores)}**.")
    lines.append(f"Empirical saturation: **{int(scores['empirical_saturation'].sum())}**.")
    lines.append(f"Original-DRO induced saturation: **{int(scores['original_dro_induced_saturation'].sum())}**.")
    lines.append(f"Original-DRO total saturation: **{int(scores['original_dro_total_saturated'].sum())} of {len(scores)}**.")
    lines.append(f"SA-DRO total saturation: **{int(scores['sa_dro_total_saturated'].sum())} of {len(scores)}**.")
    finite_ratio = scores.loc[np.isfinite(scores["saturation_ratio"]), "saturation_ratio"]
    lines.append(f"Median finite saturation_ratio: **{float(finite_ratio.median()) if not finite_ratio.empty else float('nan'):.10f}**.")
    lines.append(f"Original-DRO theoretical-vs-observed saturation mismatches: **{int(scores['original_dro_saturation_mismatch'].sum())}**.")
    lines.append("")
    lines.append("## LP validation")
    lines.append("")
    lines.append(f"LP validation passed for **{int(validation['validation_pass'].sum())} of {len(validation)}** candidates.")
    lines.append(f"epsilon = 0 QA passed: **{bool(validation['epsilon_zero_pass'].all())}**.")
    lines.append(f"epsilon = epsilon_sat QA passed: **{bool(validation['epsilon_sat_pass'].all())}**.")
    lines.append(f"epsilon = 0.99 * epsilon_sat QA passed outside tolerance for positive thresholds: **{bool(validation['near_threshold_pass'].all())}**.")
    lines.append(f"Maximum zero-radius absolute error: **{float(validation['lp_zero_absolute_error_hours'].max()):.12g}** hours.")
    lines.append(f"Maximum saturation-threshold absolute error: **{float(validation['lp_sat_absolute_error_to_cap_hours'].max()):.12g}** hours.")
    lines.append("")
    lines.append("## External Q3 evaluation")
    lines.append("")
    lines.append(f"Q3-origin direct-follow rows: **{len(q3):,}**.")
    lines.append(f"Q3 transition universe: **{int(q3['transition'].nunique())}**.")
    lines.append("The Q3 oracle is retrospective and was not used for model construction, candidate selection, epsilon calibration, kappa selection, or portfolio freeze.")
    lines.append("")
    k5 = q3_eval[q3_eval["K"] == 5].copy()
    lines.append("### Primary K = 5 captured mean-burden share")
    lines.append("")
    lines.append("| Method | Q3 captured mean-burden share | Mean-burden regret |")
    lines.append("|---|---:|---:|")
    for _, row in k5.set_index("method").loc[list(METHODS)].reset_index().iterrows():
        lines.append(f"| {row['method']} | {row['q3_captured_mean_burden_share_pct']:.6f}% | {row['q3_mean_burden_regret_pp']:.6f} pp |")
    lines.append("")
    lines.append("## Q3 bootstrap uncertainty")
    lines.append("")
    lines.append(f"The block bootstrap used **{int(bootstrap_rows['replicates'].iloc[0])}** replicates over **{int(bootstrap_rows['weekly_blocks'].iloc[0])}** nonempty weekly blocks with seed **{SEED}**.")
    lines.append("")
    lines.append("| Comparison | Difference mean | 95% CI | Includes zero |")
    lines.append("|---|---:|---:|:---:|")
    for _, row in bootstrap_rows.iterrows():
        lines.append(f"| {row['comparison']} | {row['difference_mean_pp']:.6f} pp | [{row['ci95_low_pp']:.6f}, {row['ci95_high_pp']:.6f}] | {bool(row['ci_includes_zero'])} |")
    lines.append("")
    lines.append(f"Defensible external SA-DRO advantage under the pre-specified rule: **{summary['defensible_sa_dro_advantage_at_k5']}**.")
    lines.append("The rule requires the lower endpoint of every listed SA-DRO minus baseline interval to be strictly positive. When an interval includes zero, no superiority claim is made for that comparison.")
    lines.append("")
    lines.append("## H1 stability")
    lines.append("")
    summary_stability = stability[stability["row_type"] == "summary"]
    lines.append("| Method | Mean Jaccard versus full H1 | Transitions with inclusion probability at least 0.8 |")
    lines.append("|---|---:|---:|")
    for _, row in summary_stability.iterrows():
        lines.append(f"| {row['method']} | {row['mean_jaccard_vs_full_h1']:.6f} | {int(row['transitions_inclusion_probability_ge_0_8'])} |")
    lines.append("")
    lines.append("## Sensitivity")
    lines.append("")
    lines.append("Sensitivity configurations were run without selecting a preferred configuration from Q3. They vary the cap, alpha, grid step, and SA-DRO kappa according to the pre-specified primary and sensitivity values. K values 3, 5, and 10 were retained in every configuration.")
    lines.append(f"Sensitivity rows: **{len(sensitivity)}**.")
    lines.append("")
    lines.append("## Integrity")
    lines.append("")
    lines.append(f"Random seed: **{SEED}**.")
    lines.append("The outcome is inter-event elapsed time. The primary preprocessing is COMPLETE-only. H1 origins are January to June 2016. Q3 origins are July to September 2016.")
    lines.append("Phases 1 through 3C were not modified. The repository excludes the XES.GZ dataset and contains code, output tables, and provenance metadata.")
    lines.append("")
    lines.append("## Warnings")
    lines.append("")
    for warning in summary.get("warnings", []):
        lines.append(f"- {warning}")
    if not summary.get("warnings"):
        lines.append("- None.")
    (RESULTS_DIR / "PHASE4_EXTERNAL_VALIDATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase4() -> dict[str, Any]:
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    xes_path, provenance = ensure_dataset()
    json_dump(RESULTS_DIR / "00_dataset_provenance.json", provenance)
    print(f"Using official dataset: {xes_path}")
    print("Extracting H1-origin COMPLETE transitions before Q3 access.")
    h1, h1_stats = parse_xes_origin_window(xes_path, H1_START, H1_END, "H1")
    if h1.empty:
        raise RuntimeError("The H1 extraction returned no direct-follow rows.")
    distinct_activities = pd.unique(pd.concat([h1["origin_activity"], h1["destination_activity"]], ignore_index=True))
    all_h1_calibration = calibration_for_frame(h1, PRIMARY_CAP_HOURS, PRIMARY_GRID_HOURS)
    candidate_mask = (
        (all_h1_calibration["h1_observations"] >= MIN_H1_OBSERVATIONS)
        & (all_h1_calibration["valid_block_count"] >= MIN_VALID_BLOCKS)
    )
    candidates = all_h1_calibration.loc[candidate_mask].copy().sort_values("transition").reset_index(drop=True)
    if len(candidates) < 10:
        raise RuntimeError(
            "The H1 feasibility gate failed before Q3 access. "
            f"Only {len(candidates)} candidate transitions met the fixed thresholds. "
            "The design requires at least 10."
        )
    candidate_transitions = set(candidates["transition"].astype(str))
    print(f"H1 rows: {len(h1):,}. Candidate transitions: {len(candidates)}.")
    calibration_table = build_all_calibrations(h1, candidate_transitions)
    primary_calibration = calibration_table[calibration_table["calibration_role"] == "primary"].copy()
    vectors = transition_vectors(h1, candidate_transitions)
    scores, validation = primary_scores(h1, candidates, primary_calibration, vectors)
    if len(scores) < 10:
        raise RuntimeError("The primary H1 candidate score table contains fewer than 10 transitions.")
    if not bool(validation["validation_pass"].all()):
        raise RuntimeError("Primary LP validation failed before Q3 access.")
    if not bool(scores["original_dro_saturation_mismatch"].sum() == 0):
        raise RuntimeError("Original-DRO saturation classification QA failed before Q3 access.")
    write_csv(RESULTS_DIR / "01_external_h1_candidate_transitions.csv", candidates)
    write_csv(RESULTS_DIR / "02_external_wasserstein_calibration.csv", calibration_table)
    write_csv(RESULTS_DIR / "03_external_saturation_thresholds.csv", scores[
        [
            "transition",
            "h1_observations",
            "epsilon_temporal_hours",
            "m_cap",
            "delta_mass_to_cap",
            "epsilon_sat_hours",
            "saturation_ratio",
            "empirical_saturation",
            "original_dro_predicted_saturated",
            "original_dro_observed_saturated",
            "original_dro_induced_saturation",
            "original_dro_total_saturated",
            "sa_dro_induced_saturation",
            "sa_dro_total_saturated",
        ]
    ])
    write_csv(RESULTS_DIR / "04_external_transition_scores.csv", scores)
    write_csv(RESULTS_DIR / "10_external_lp_validation.csv", validation)
    sensitivity, sensitivity_selections = build_sensitivity_results(
        h1, scores, candidate_transitions, calibration_table
    )
    write_csv(RESULTS_DIR / "09_external_sensitivity_results.csv", sensitivity)
    primary_selections = build_primary_selections(scores)
    manifest, manifest_sha = create_freeze_manifest(
        candidates, scores, primary_selections, sensitivity_selections
    )
    write_csv(RESULTS_DIR / "05_external_frozen_portfolios.csv", build_selection_frame(scores, primary_selections, manifest_sha))
    json_dump(RESULTS_DIR / "phase4_frozen_portfolio_manifest.json", manifest)
    print(f"Portfolio freeze complete. Manifest SHA-256: {manifest_sha}")
    print("Opening Q3-origin holdout after the H1 freeze.")
    q3, q3_stats = parse_xes_origin_window(xes_path, Q3_START, Q3_END, "Q3")
    compare_parse_stats(h1_stats, q3_stats)
    if q3.empty:
        raise RuntimeError("The Q3 extraction returned no direct-follow rows.")
    q3_eval, q3_universe = evaluate_q3(scores, q3, primary_selections)
    write_csv(RESULTS_DIR / "06_external_q3_evaluation.csv", q3_eval)
    bootstrap_rows, bootstrap_samples = q3_bootstrap_differences(q3, primary_selections)
    bootstrap_output = bootstrap_rows.copy()
    write_csv(RESULTS_DIR / "07_external_q3_bootstrap_differences.csv", bootstrap_output)
    stability = h1_bootstrap_stability(h1, scores, primary_selections, vectors)
    write_csv(RESULTS_DIR / "08_external_h1_bootstrap_stability.csv", stability)

    ci_all_positive = bool((bootstrap_rows["ci95_low_pp"] > 0.0).all())
    finite_ratio = scores.loc[np.isfinite(scores["saturation_ratio"]), "saturation_ratio"]
    q3_k5 = q3_eval[q3_eval["K"] == 5].set_index("method")
    tail_original_spearman = float(q3_k5.loc["Original DRO", "spearman_h1_vs_q3_tail_burden"])
    tail_sa_spearman = float(q3_k5.loc["Saturation-Aware DRO", "spearman_h1_vs_q3_tail_burden"])
    tail_original_jaccard = float(q3_k5.loc["Original DRO", "jaccard_vs_q3_tail_oracle"])
    tail_sa_jaccard = float(q3_k5.loc["Saturation-Aware DRO", "jaccard_vs_q3_tail_oracle"])
    warnings: list[str] = []
    for _, row in bootstrap_rows.iterrows():
        if bool(row["ci_includes_zero"]):
            warnings.append(
                f"The 95% Q3 weekly block bootstrap interval for {row['comparison']} includes zero."
            )
    if not ci_all_positive:
        warnings.append("No superiority claim is made for SA-DRO because at least one pre-specified interval includes zero.")
    if int(scores["original_dro_saturation_mismatch"].sum()) != 0:
        warnings.append("Original-DRO theoretical-vs-observed saturation classification has nonzero mismatches.")
    summary: dict[str, Any] = {
        "phase": "Phase 4",
        "title": "EXTERNAL VALIDATION ON BPI CHALLENGE 2017",
        "dataset_provenance": provenance,
        "candidate_count_h1_only": int(len(scores)),
        "h1_direct_follow_rows": int(len(h1)),
        "h1_distinct_activities": int(len(distinct_activities)),
        "h1_distinct_transitions": int(h1["transition"].nunique()),
        "q3_direct_follow_rows": int(len(q3)),
        "q3_distinct_transitions": int(q3["transition"].nunique()),
        "h1_stats": h1_stats,
        "q3_stats": q3_stats,
        "feasibility_gate_passed": True,
        "candidate_thresholds": {
            "minimum_h1_observations": MIN_H1_OBSERVATIONS,
            "minimum_valid_h1_blocks": MIN_VALID_BLOCKS,
            "minimum_observations_per_valid_block": MIN_BLOCK_OBSERVATIONS,
        },
        "primary_parameters": {
            "cap_hours": PRIMARY_CAP_HOURS,
            "alpha": PRIMARY_ALPHA,
            "grid_step_hours": PRIMARY_GRID_HOURS,
            "kappa_sa": PRIMARY_KAPPA,
            "K_confirmatory": 5,
        },
        "sensitivity_parameters": {
            "cap_hours": [PRIMARY_CAP_HOURS, SENSITIVITY_CAP_HOURS],
            "alpha": [PRIMARY_ALPHA, SENSITIVITY_ALPHA],
            "grid_step_hours": [PRIMARY_GRID_HOURS, SENSITIVITY_GRID_HOURS],
            "kappa_sa": [PRIMARY_KAPPA, SENSITIVITY_KAPPA],
            "K": list(PORTFOLIO_KS),
        },
        "primary_lp_validation_passed": bool(validation["validation_pass"].all()),
        "epsilon_zero_lp_qa_passed": bool(validation["epsilon_zero_pass"].all()),
        "epsilon_sat_lp_qa_passed": bool(validation["epsilon_sat_pass"].all()),
        "epsilon_099_sat_lp_qa_passed": bool(validation["near_threshold_pass"].all()),
        "original_dro_theoretical_observed_mismatch_count": int(scores["original_dro_saturation_mismatch"].sum()),
        "empirical_saturated_count": int(scores["empirical_saturation"].sum()),
        "original_dro_induced_saturated_count": int(scores["original_dro_induced_saturation"].sum()),
        "original_dro_total_saturated_count": int(scores["original_dro_total_saturated"].sum()),
        "sa_dro_induced_saturated_count": int(scores["sa_dro_induced_saturation"].sum()),
        "sa_dro_total_saturated_count": int(scores["sa_dro_total_saturated"].sum()),
        "unsaturated_original_dro_count": int((~scores["original_dro_total_saturated"]).sum()),
        "unsaturated_sa_dro_count": int((~scores["sa_dro_total_saturated"]).sum()),
        "percentage_candidates_epsilon_temporal_ge_epsilon_sat": float(
            100.0 * scores["original_dro_predicted_saturated"].mean()
        ),
        "median_finite_saturation_ratio": float(finite_ratio.median()) if not finite_ratio.empty else None,
        "rank_correlation_tail_vs_original_dro_spearman": tail_original_spearman,
        "rank_correlation_tail_vs_sa_dro_spearman": tail_sa_spearman,
        "k5_tail_vs_original_dro_jaccard": tail_original_jaccard,
        "k5_tail_vs_sa_dro_jaccard": tail_sa_jaccard,
        "frozen_manifest_sha256": manifest_sha,
        "q3_accessed_before_portfolio_freeze": False,
        "primary_k5_selections": {
            method: primary_selections[(method, 5)] for method in METHODS
        },
        "q3_primary_k5": {
            method: {
                "captured_mean_burden_share_pct": float(q3_k5.loc[method, "q3_captured_mean_burden_share_pct"]),
                "captured_tail_burden_share_pct": float(q3_k5.loc[method, "q3_captured_tail_burden_share_pct"]),
            }
            for method in METHODS
        },
        "q3_bootstrap_ci_k5": bootstrap_rows.to_dict(orient="records"),
        "h1_stability_summary_k5": stability[stability["row_type"] == "summary"].to_dict(orient="records"),
        "sensitivity_row_count": int(len(sensitivity)),
        "defensible_sa_dro_advantage_at_k5": ci_all_positive,
        "seed": SEED,
        "outcome_label": "inter-event elapsed time",
        "complete_only_primary_preprocessing": True,
        "phases_1_to_3c_modified": False,
        "warnings": warnings,
        "errors": [],
    }
    json_dump(RESULTS_DIR / "phase4_summary.json", summary)
    write_report(
        provenance,
        h1_stats,
        q3_stats,
        h1,
        q3,
        candidates,
        scores,
        validation,
        q3_eval,
        bootstrap_rows,
        stability,
        sensitivity,
        manifest,
        summary,
    )
    print("Phase 4 external validation completed.")
    return summary


if __name__ == "__main__":
    run_phase4()
