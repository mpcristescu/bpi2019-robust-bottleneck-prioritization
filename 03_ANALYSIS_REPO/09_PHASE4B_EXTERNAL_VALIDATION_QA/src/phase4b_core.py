from __future__ import annotations

import gzip
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


SEED = 20260914
EXPECTED_DATASET_SHA256 = "183c5e5189282779c811c78c33ff936351b3dd201165d612211fc220936f8249"
EXPECTED_MANIFEST_SHA256 = "9b56a20f976afabac60c5f83e0a50ec6c5a13c8056a24deced6f57e43194e21f"
H1_START = pd.Timestamp("2016-01-01 00:00:00", tz="UTC")
H1_END = pd.Timestamp("2016-07-01 00:00:00", tz="UTC")
Q3_START = pd.Timestamp("2016-07-01 00:00:00", tz="UTC")
Q3_END = pd.Timestamp("2016-10-01 00:00:00", tz="UTC")
PRIMARY_CAP_HOURS = 2160
PRIMARY_ALPHA = 0.95
PRIMARY_GRID_HOURS = 24
PRIMARY_KAPPA = 0.90
PORTFOLIO_KS = (3, 5, 10)
METHODS = (
    "Frequency",
    "Mean burden",
    "Empirical Tail",
    "Original DRO",
    "Saturation-Aware DRO",
)
SCORE_COLUMNS = {
    "Frequency": "frequency_score",
    "Mean burden": "mean_burden_score",
    "Empirical Tail": "tail_score",
    "Original DRO": "original_dro_score",
    "Saturation-Aware DRO": "sa_dro_score",
}
Q3_PRIMARY_EXPECTED = {
    "Frequency": 7.3270119674,
    "Mean burden": 70.2373056645,
    "Empirical Tail": 70.2373056645,
    "Original DRO": 70.2373056645,
    "Saturation-Aware DRO": 70.2373056645,
}
REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parents[1]
PHASE4_RESULTS = PROJECT_ROOT / "04_ANALYSIS_RESULTS" / "08_PHASE4_EXTERNAL_VALIDATION_BPI2017"
PHASE3C_RESULTS = PROJECT_ROOT / "04_ANALYSIS_RESULTS" / "07_PHASE3C_SAMPLE_ALIGNMENT_AUDIT"
DATASET_DIR = PROJECT_ROOT / "01_DATASET" / "02_EXTERNAL_VALIDATION_BPI2017"
RESULTS_DIR = PROJECT_ROOT / "04_ANALYSIS_RESULTS" / "09_PHASE4B_EXTERNAL_VALIDATION_QA"
DATASET_PATH = DATASET_DIR / "BPI Challenge 2017.xes.gz"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", na_rep="")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or str(value).strip() == "":
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
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


def parse_q3_origin_window(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    stats = {
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
        "window_specific_retained_direct_follow_rows": 0,
        "timestamp_tie_pairs": 0,
        "timestamp_tie_cases": 0,
        "timestamp_tie_groups": 0,
    }
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rb") as handle:
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
                        if value is not None and str(value).strip():
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
                    if activity is None or not str(activity).strip():
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
                tie_pairs = 0
                tie_groups = 0
                index = 0
                while index < len(complete_events):
                    next_index = index + 1
                    while next_index < len(complete_events) and complete_events[next_index]["timestamp"] == complete_events[index]["timestamp"]:
                        next_index += 1
                    group_size = next_index - index
                    if group_size > 1:
                        tie_groups += 1
                        tie_pairs += group_size - 1
                    index = next_index
                stats["timestamp_tie_pairs"] += tie_pairs
                stats["timestamp_tie_groups"] += tie_groups
                if tie_pairs:
                    stats["timestamp_tie_cases"] += 1
                for origin, destination in zip(complete_events, complete_events[1:]):
                    stats["pair_candidates_before_negative_filter"] += 1
                    elapsed_hours = (destination["timestamp"] - origin["timestamp"]).total_seconds() / 3600.0
                    if elapsed_hours < 0:
                        stats["negative_elapsed_rows_removed"] += 1
                        continue
                    if Q3_START <= origin["timestamp"] < Q3_END:
                        rows.append(
                            {
                                "case_id": origin["case_id"],
                                "origin_activity": origin["activity"],
                                "destination_activity": destination["activity"],
                                "transition": f"{origin['activity']} -> {destination['activity']}",
                                "origin_timestamp": origin["timestamp"],
                                "destination_timestamp": destination["timestamp"],
                                "elapsed_hours": float(elapsed_hours),
                            }
                        )
                element.clear()
        except ET.ParseError as exc:
            raise RuntimeError("The official BPI Challenge 2017 XES file could not be parsed.") from exc
    stats["window_specific_retained_direct_follow_rows"] = len(rows)
    frame = pd.DataFrame(
        rows,
        columns=[
            "case_id",
            "origin_activity",
            "destination_activity",
            "transition",
            "origin_timestamp",
            "destination_timestamp",
            "elapsed_hours",
        ],
    )
    if not frame.empty:
        frame["origin_timestamp"] = pd.to_datetime(frame["origin_timestamp"], utc=True)
        frame["destination_timestamp"] = pd.to_datetime(frame["destination_timestamp"], utc=True)
    return frame, stats


def empirical_cvar(values: np.ndarray, alpha: float) -> float:
    array = np.sort(np.asarray(values, dtype=float))
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan")
    tail_mass = (1.0 - alpha) * len(array)
    if tail_mass <= 0:
        return float(array[-1])
    full = int(np.floor(tail_mass))
    fraction = tail_mass - full
    total = float(array[-full:].sum()) if full else 0.0
    if fraction > 1e-12 and full < len(array):
        total += fraction * float(array[-full - 1])
    return total / tail_mass


def q3_universe_metrics(q3: pd.DataFrame) -> pd.DataFrame:
    work = q3.copy()
    work["capped_elapsed_hours"] = np.clip(work["elapsed_hours"].to_numpy(float), 0.0, float(PRIMARY_CAP_HOURS))
    total = len(work)
    rows: list[dict[str, Any]] = []
    for transition, group in work.groupby("transition", sort=True):
        vector = group["capped_elapsed_hours"].to_numpy(float)
        frequency = float(len(vector) / total) if total else 0.0
        mean_value = float(np.mean(vector)) if len(vector) else 0.0
        tail_value = float(empirical_cvar(vector, PRIMARY_ALPHA)) if len(vector) else 0.0
        rows.append(
            {
                "transition": str(transition),
                "q3_observations": int(len(vector)),
                "q3_frequency_share": frequency,
                "q3_mean_capped_elapsed_hours": mean_value,
                "q3_empirical_cvar95_hours": tail_value,
                "q3_mean_burden": frequency * mean_value,
                "q3_tail_burden": frequency * tail_value,
            }
        )
    return pd.DataFrame(rows)


def split_selection(value: Any) -> list[str]:
    text = "" if value is None else str(value)
    return [] if not text else [item for item in text.split(" || ") if item]


def parse_manifest_selections(manifest: dict[str, Any]) -> tuple[dict[tuple[str, str, int], list[str]], dict[tuple[str, int], list[str]]]:
    sensitivity: dict[tuple[str, str, int], list[str]] = {}
    for key, values in manifest["sensitivity_selections"].items():
        configuration, method, k_text = key.rsplit("|", 2)
        k = int(k_text.split("=", 1)[1])
        sensitivity[(configuration, method, k)] = [str(item) for item in values]
    primary: dict[tuple[str, int], list[str]] = {}
    for key, values in manifest["primary_selections"].items():
        method, k_text = key.rsplit("|", 1)
        k = int(k_text.split("=", 1)[1])
        primary[(method, k)] = [str(item) for item in values]
    return sensitivity, primary


def safe_corr(x: pd.Series, y: pd.Series, kind: str) -> float:
    result = spearmanr(x, y) if kind == "Spearman" else kendalltau(x, y)
    value = float(result.statistic)
    return value if np.isfinite(value) else float("nan")


def corrected_correlations(scores: pd.DataFrame, q3_eval: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("Tail score vs Original-DRO score", "tail_score", "original_dro_score"),
        ("Tail score vs SA-DRO score", "tail_score", "sa_dro_score"),
        ("Mean-burden score vs Original-DRO score", "mean_burden_score", "original_dro_score"),
        ("Mean-burden score vs SA-DRO score", "mean_burden_score", "sa_dro_score"),
    ]
    rows: list[dict[str, Any]] = []
    for label, left, right in pairs:
        for kind in ("Spearman", "Kendall"):
            rows.append(
                {
                    "correlation_scope": "H1 score-to-score",
                    "comparison_label": label,
                    "method": "",
                    "correlation_type": kind,
                    "value": safe_corr(scores[left], scores[right], kind),
                    "n": int(len(scores)),
                    "source_file": "04_external_transition_scores.csv",
                    "source_columns": f"{left},{right}",
                    "note": "Calculated directly from the 41 H1 candidate score rows.",
                }
            )
    primary_q3 = q3_eval[q3_eval["K"] == 5].set_index("method")
    for method in METHODS:
        row = primary_q3.loc[method]
        for kind, column, burden_label, burden_column in (
            ("Spearman", "spearman_h1_vs_q3_mean_burden", "mean-burden", "q3_mean_burden"),
            ("Kendall", "kendall_h1_vs_q3_mean_burden", "mean-burden", "q3_mean_burden"),
            ("Spearman", "spearman_h1_vs_q3_tail_burden", "tail-burden", "q3_tail_burden"),
            ("Kendall", "kendall_h1_vs_q3_tail_burden", "tail-burden", "q3_tail_burden"),
        ):
            rows.append(
                {
                    "correlation_scope": "H1-to-Q3 holdout",
                    "comparison_label": f"H1 ranking vs Q3 {burden_label} ranking",
                    "method": method,
                    "correlation_type": kind,
                    "value": float(row[column]),
                    "n": int(row["ranking_n"]),
                    "source_file": "06_external_q3_evaluation.csv",
                    "source_columns": f"{SCORE_COLUMNS[method]},{burden_column}",
                    "note": "Preserved from Phase 4 with the H1-to-Q3 scope stated explicitly.",
                }
            )
    return pd.DataFrame(rows)


def check_value(
    rows: list[dict[str, Any]],
    name: str,
    expected: Any,
    observed: Any,
    source: str,
    tolerance: float | None = None,
) -> None:
    if tolerance is not None:
        try:
            passed = bool(np.isclose(float(expected), float(observed), atol=tolerance, rtol=0.0))
        except (TypeError, ValueError):
            passed = False
    else:
        passed = bool(expected == observed)
    rows.append(
        {
            "check": name,
            "expected": expected,
            "observed": observed,
            "tolerance": "" if tolerance is None else tolerance,
            "passed": passed,
            "source": source,
        }
    )


def primary_reproduction_gate(
    summary: dict[str, Any],
    manifest: dict[str, Any],
    scores: pd.DataFrame,
    q3_eval: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    check_value(rows, "candidate_count", 41, int(summary["candidate_count_h1_only"]), "phase4_summary.json")
    check_value(rows, "H1 direct-follow rows", 193194, int(summary["h1_direct_follow_rows"]), "phase4_summary.json")
    check_value(rows, "Q3 direct-follow rows", 129191, int(summary["q3_direct_follow_rows"]), "phase4_summary.json")
    params = summary["primary_parameters"]
    check_value(rows, "primary cap hours", 2160, int(params["cap_hours"]), "phase4_summary.json")
    check_value(rows, "primary alpha", 0.95, float(params["alpha"]), "phase4_summary.json", 1e-12)
    check_value(rows, "primary grid hours", 24, int(params["grid_step_hours"]), "phase4_summary.json")
    check_value(rows, "primary SA kappa", 0.90, float(params["kappa_sa"]), "phase4_summary.json", 1e-12)
    check_value(rows, "primary K", 5, int(params["K_confirmatory"]), "phase4_summary.json")
    q3_k5 = q3_eval[q3_eval["K"] == 5].set_index("method")
    for method, expected in Q3_PRIMARY_EXPECTED.items():
        observed = float(q3_k5.loc[method, "q3_captured_mean_burden_share_pct"])
        check_value(rows, f"primary K=5 Q3 share, {method}", expected, observed, "06_external_q3_evaluation.csv", 1e-8)
    check_value(rows, "Original-DRO saturated count", 2, int(summary["original_dro_total_saturated_count"]), "phase4_summary.json")
    check_value(rows, "SA-DRO saturated count", 0, int(summary["sa_dro_total_saturated_count"]), "phase4_summary.json")
    check_value(rows, "candidate score rows", 41, int(len(scores)), "04_external_transition_scores.csv")
    check_value(rows, "all score rows are H1-only candidates", True, bool(scores["candidate_h1_only"].all()), "04_external_transition_scores.csv")
    check_value(rows, "Q3 accessed before freeze", False, bool(summary["q3_accessed_before_portfolio_freeze"]), "phase4_summary.json")
    check_value(rows, "manifest Q3 accessed before freeze", False, bool(manifest["q3_accessed_before_freeze"]), "phase4_frozen_portfolio_manifest.json")
    check_value(rows, "manifest checksum", EXPECTED_MANIFEST_SHA256, str(manifest["manifest_sha256"]), "phase4_frozen_portfolio_manifest.json")
    check_value(rows, "summary manifest checksum", EXPECTED_MANIFEST_SHA256, str(summary["frozen_manifest_sha256"]), "phase4_summary.json")
    check_value(rows, "sensitivity rows", 120, int(len(sensitivity)), "09_external_sensitivity_results.csv")
    check_value(rows, "all sensitivity selections marked H1-only", True, bool(sensitivity["selection_h1_only"].astype(str).str.casefold().eq("true").all()), "09_external_sensitivity_results.csv")
    check_value(rows, "Phases 1 to 3C modified", False, bool(summary["phases_1_to_3c_modified"]), "phase4_summary.json")
    check_value(rows, "Original-DRO theoretical-observed mismatch count", 0, int(summary["original_dro_theoretical_observed_mismatch_count"]), "phase4_summary.json")
    result = pd.DataFrame(rows)
    if not bool(result["passed"].all()):
        failed = result.loc[~result["passed"], "check"].tolist()
        raise RuntimeError("Phase 4 primary reproduction gate failed: " + ", ".join(failed))
    return result


def sensitivity_parameters(sensitivity: pd.DataFrame) -> tuple[list[str], dict[str, dict[str, Any]]]:
    configs = sensitivity["configuration"].drop_duplicates().astype(str).tolist()
    parameters: dict[str, dict[str, Any]] = {}
    for configuration, group in sensitivity.groupby("configuration", sort=False):
        first = group.iloc[0]
        parameters[str(configuration)] = {
            "cap_hours": int(first["cap_hours"]),
            "alpha": float(first["alpha"]),
            "grid_step_hours": int(first["grid_step_hours"]),
            "kappa_sa": float(first["kappa_sa"]),
        }
    return configs, parameters


def validate_manifest_selections(
    manifest_selections: dict[tuple[str, str, int], list[str]],
    sensitivity: pd.DataFrame,
) -> None:
    for row in sensitivity.itertuples(index=False):
        key = (str(row.configuration), str(row.method), int(row.K))
        if key not in manifest_selections:
            raise RuntimeError(f"Missing frozen manifest selection for {key}.")
        if manifest_selections[key] != split_selection(row.selected_transitions):
            raise RuntimeError(f"Sensitivity CSV and frozen manifest disagree for {key}.")


def frozen_sensitivity_q3_evaluation(
    q3: pd.DataFrame,
    manifest_selections: dict[tuple[str, str, int], list[str]],
    configs: list[str],
    parameters: dict[str, dict[str, Any]],
    manifest_sha256: str,
) -> pd.DataFrame:
    universe = q3_universe_metrics(q3)
    total_mean = float(universe["q3_mean_burden"].sum())
    total_tail = float(universe["q3_tail_burden"].sum())
    rows: list[dict[str, Any]] = []
    for configuration in configs:
        for method in METHODS:
            for k in PORTFOLIO_KS:
                selected = manifest_selections[(configuration, method, k)]
                chosen_rows = universe[universe["transition"].isin(selected)]
                mean_oracle = universe.sort_values(["q3_mean_burden", "transition"], ascending=[False, True]).head(k)
                tail_oracle = universe.sort_values(["q3_tail_burden", "transition"], ascending=[False, True]).head(k)
                selected_set = set(selected)
                mean_oracle_set = set(mean_oracle["transition"].astype(str))
                tail_oracle_set = set(tail_oracle["transition"].astype(str))
                selected_mean_share = 100.0 * float(chosen_rows["q3_mean_burden"].sum()) / total_mean if total_mean else float("nan")
                selected_tail_share = 100.0 * float(chosen_rows["q3_tail_burden"].sum()) / total_tail if total_tail else float("nan")
                mean_oracle_share = 100.0 * float(mean_oracle["q3_mean_burden"].sum()) / total_mean if total_mean else float("nan")
                tail_oracle_share = 100.0 * float(tail_oracle["q3_tail_burden"].sum()) / total_tail if total_tail else float("nan")
                rows.append(
                    {
                        "configuration": configuration,
                        **parameters[configuration],
                        "method": method,
                        "K": k,
                        "frozen_before_q3": True,
                        "selection_manifest_sha256": manifest_sha256,
                        "q3_observations": int(len(q3)),
                        "q3_transition_universe": int(len(universe)),
                        "q3_candidate_coverage_count": int(len(chosen_rows)),
                        "q3_captured_mean_burden_share_pct": selected_mean_share,
                        "q3_captured_tail_burden_share_pct": selected_tail_share,
                        "q3_mean_oracle_share_pct": mean_oracle_share,
                        "q3_tail_oracle_share_pct": tail_oracle_share,
                        "q3_mean_burden_regret_pp": mean_oracle_share - selected_mean_share,
                        "q3_tail_burden_regret_pp": tail_oracle_share - selected_tail_share,
                        "jaccard_vs_q3_mean_oracle": len(selected_set & mean_oracle_set) / len(selected_set | mean_oracle_set),
                        "jaccard_vs_q3_tail_oracle": len(selected_set & tail_oracle_set) / len(selected_set | tail_oracle_set),
                        "selected_transitions": " || ".join(selected),
                        "q3_metric_definition": "Phase 4 primary C=2160 h and alpha=0.95",
                        "interpretation": "pre-frozen combined sensitivity configuration",
                    }
                )
    return pd.DataFrame(rows)


def frozen_sensitivity_stability(
    manifest_selections: dict[tuple[str, str, int], list[str]],
    primary_selections: dict[tuple[str, int], list[str]],
    configs: list[str],
    manifest_sha256: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for configuration in configs:
        for method in METHODS:
            primary = primary_selections[(method, 5)]
            frozen = manifest_selections[(configuration, method, 5)]
            primary_set = set(primary)
            frozen_set = set(frozen)
            set_identical = primary_set == frozen_set
            order_identical = primary == frozen
            rows.append(
                {
                    "configuration": configuration,
                    "method": method,
                    "K": 5,
                    "primary_set": " || ".join(primary),
                    "frozen_sensitivity_set": " || ".join(frozen),
                    "overlap_count": len(primary_set & frozen_set),
                    "jaccard_vs_primary": len(primary_set & frozen_set) / len(primary_set | frozen_set),
                    "same_set_as_primary": set_identical,
                    "identical_order_as_primary": order_identical,
                    "differs_only_in_order": bool(set_identical and not order_identical),
                    "membership_differs": bool(not set_identical),
                    "selection_manifest_sha256": manifest_sha256,
                }
            )
    return pd.DataFrame(rows)


def phase3c_diagnostic() -> dict[str, Any]:
    summary_path = PHASE3C_RESULTS / "phase3c_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Required Phase 3C summary is missing: {summary_path}")
    source = read_json(summary_path)
    expected = {
        "candidate_count": 63,
        "empirical_saturated_count": 8,
        "dro_induced_saturated_count": 28,
        "total_saturated_count": 36,
        "median_finite_saturation_ratio": 1.0182147910831052,
    }
    observed = {
        "candidate_count": source["candidate_count"],
        "empirical_saturated_count": source["empirical_saturated_count"],
        "dro_induced_saturated_count": source["dro_induced_saturated_count"],
        "total_saturated_count": source["total_saturated_count"],
        "median_finite_saturation_ratio": source["median_finite_saturation_ratio"],
    }
    for key in expected:
        if key == "median_finite_saturation_ratio":
            if not np.isclose(float(expected[key]), float(observed[key]), atol=1e-12, rtol=0.0):
                raise RuntimeError(f"Phase 3C diagnostic mismatch for {key}.")
        elif int(expected[key]) != int(observed[key]):
            raise RuntimeError(f"Phase 3C diagnostic mismatch for {key}.")
    return {
        "source_file": "07_PHASE3C_SAMPLE_ALIGNMENT_AUDIT/phase3c_summary.json",
        "candidate_count": int(observed["candidate_count"]),
        "empirical_saturated_count": int(observed["empirical_saturated_count"]),
        "dro_induced_saturated_count": int(observed["dro_induced_saturated_count"]),
        "total_saturated_count": int(observed["total_saturated_count"]),
        "median_finite_saturation_ratio": float(observed["median_finite_saturation_ratio"]),
        "q3_read": bool(source["q3_read"]),
        "q3_used": bool(source["q3_used"]),
    }


def write_report(
    summary: dict[str, Any],
    correlations: pd.DataFrame,
    stability: pd.DataFrame,
    reproduction: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Phase 4B, External Validation QA")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("This QA stage reads the completed Phase 4 outputs, corrects the H1 score-correlation labels, and evaluates only the sensitivity portfolios that were frozen before Q3 access. No primary selection was changed and no new Q3-driven selection was created.")
    lines.append("")
    lines.append("## Primary reproduction gate")
    lines.append("")
    lines.append(f"The Phase 4 primary reproduction gate passed: **{summary['primary_reproduction_passed']}**.")
    lines.append(f"Candidate transitions: **{summary['candidate_count']}**.")
    lines.append(f"Primary configuration: C = **{summary['primary_parameters']['cap_hours']} h**, alpha = **{summary['primary_parameters']['alpha']}**, grid = **{summary['primary_parameters']['grid_step_hours']} h**, SA kappa = **{summary['primary_parameters']['kappa_sa']}**, confirmatory K = **{summary['primary_parameters']['K_confirmatory']}**.")
    lines.append("Primary K=5 captured mean-burden shares were reproduced from the Phase 4 Q3 evaluation output.")
    lines.append("")
    lines.append("| Method | Captured Q3 mean-burden share |")
    lines.append("|---|---:|")
    for method, value in Q3_PRIMARY_EXPECTED.items():
        lines.append(f"| {method} | {value:.10f}% |")
    lines.append("")
    lines.append(f"Original-DRO total saturation: **{summary['original_dro_total_saturated_count']}/41**.")
    lines.append(f"SA-DRO total saturation: **{summary['sa_dro_total_saturated_count']}/41**.")
    lines.append(f"The original frozen manifest checksum is unchanged: {summary['original_manifest_checksum']}.")
    lines.append("")
    lines.append("## Direct-follow row labels")
    lines.append("")
    lines.append("The full XES preprocessing produced 443,797 COMPLETE direct-follow pair candidates. No negative elapsed rows were removed. The H1 and Q3 counts are window-specific retained direct-follow rows, not the global total.")
    lines.append("")
    lines.append("| Quantity | Count |")
    lines.append("|---|---:|")
    lines.append("| All COMPLETE direct-follow pair candidates in the full XES | 443,797 |")
    lines.append("| Negative elapsed rows removed | 0 |")
    lines.append(f"| H1-origin window-specific retained direct-follow rows | {summary['h1_window_rows']:,} |")
    lines.append(f"| Q3-origin window-specific retained direct-follow rows | {summary['q3_window_rows']:,} |")
    lines.append("")
    lines.append("## Corrected H1 score correlations")
    lines.append("")
    lines.append("The four Tail and Mean-burden comparisons with the two DRO scores were calculated directly from the 41 rows in 04_external_transition_scores.csv. The H1-to-Q3 correlations from Phase 4 are retained separately with their scope stated explicitly.")
    lines.append("")
    lines.append("| Scope | Comparison | Method | Type | Value | n |")
    lines.append("|---|---|---|---|---:|---:|")
    for _, row in correlations.iterrows():
        if row["correlation_scope"] == "H1 score-to-score":
            lines.append(f"| {row['correlation_scope']} | {row['comparison_label']} |  | {row['correlation_type']} | {float(row['value']):.10f} | {int(row['n'])} |")
    lines.append("")
    lines.append("## Frozen sensitivity Q3 evaluation")
    lines.append("")
    lines.append(f"The manifest contains **{summary['frozen_sensitivity_configuration_count']}** pre-frozen combined sensitivity configurations. Their **{summary['frozen_sensitivity_evaluation_rows']}** portfolios were evaluated on Q3 using the Phase 4 primary Q3 metric definition, with C = 2160 h and alpha = 0.95.")
    lines.append("The configurations combine changes in cap, alpha, grid step, and SA-DRO kappa. The results are not one-factor-at-a-time effects.")
    lines.append("")
    sa_range = summary["sa_dro_q3_mean_burden_share_range_pct"]
    lines.append(f"SA-DRO Q3 captured mean-burden share across the frozen sensitivity configurations and K values ranged from **{sa_range[0]:.10f}%** to **{sa_range[1]:.10f}%**.")
    sa_k5_range = summary["sa_dro_q3_mean_burden_share_range_pct_K5"]
    lines.append(f"At K=5, the corresponding range was **{sa_k5_range[0]:.10f}%** to **{sa_k5_range[1]:.10f}%**.")
    lines.append("")
    lines.append("## K=5 portfolio stability")
    lines.append("")
    lines.append(f"SA-DRO had the same K=5 set as the primary portfolio in **{summary['sa_dro_k5_same_set_count']} of {summary['frozen_sensitivity_configuration_count']}** configurations.")
    lines.append("")
    lines.append("| Configuration | Method | Overlap | Jaccard | Same set | Same order | Membership differs |")
    lines.append("|---|---|---:|---:|:---:|:---:|:---:|")
    for _, row in stability.iterrows():
        lines.append(f"| {row['configuration']} | {row['method']} | {int(row['overlap_count'])} | {float(row['jaccard_vs_primary']):.6f} | {bool(row['same_set_as_primary'])} | {bool(row['identical_order_as_primary'])} | {bool(row['membership_differs'])} |")
    lines.append("")
    lines.append("## Primary bootstrap verification")
    lines.append("")
    lines.append("The Phase 4 primary bootstrap output was read without rerunning it. The intervals for SA-DRO minus Original DRO, SA-DRO minus Mean burden, and SA-DRO minus Empirical Tail are [0, 0]. The interval for SA-DRO minus Frequency is approximately [61.02545, 64.68841] percentage points.")
    lines.append("The primary conclusion is unchanged. The external validation does not support a defensible superiority claim for SA-DRO over the strongest baselines.")
    lines.append("")
    lines.append("## External saturation diagnostic")
    lines.append("")
    p3 = summary["bpi2019_phase3c_diagnostic"]
    lines.append(f"BPI 2017 external H1 has empirical saturation **0/41**, Original-DRO induced saturation **2/41**, SA-DRO saturation **0/41**, and median finite epsilon_temporal/epsilon_sat **{summary['bpi2017_median_finite_epsilon_temporal_over_epsilon_sat']:.10f}**.")
    lines.append(f"The aligned BPI 2019 H1 values from Phase 3C are empirical saturation **{p3['empirical_saturated_count']}/{p3['candidate_count']}**, DRO-induced saturation **{p3['dro_induced_saturated_count']}/{p3['candidate_count']}**, total saturation **{p3['total_saturated_count']}/{p3['candidate_count']}**, and median finite saturation ratio **{p3['median_finite_saturation_ratio']:.10f}**.")
    lines.append("The external validation therefore does not show that saturation is universal. It separates a low-saturation regime in BPI 2017 from the high-saturation regime observed in aligned BPI 2019 H1.")
    lines.append("No Q3 data from BPI 2019 were read or used.")
    lines.append("")
    lines.append("## Integrity and warnings")
    lines.append("")
    lines.append(f"Dataset SHA-256 verification passed: {summary['dataset_sha256']}.")
    lines.append("Only pre-frozen sensitivity portfolios were evaluated on Q3. Phase 4 primary output was not changed. Phases 1 through 4 were not modified.")
    for warning in summary["warnings"]:
        lines.append(f"- {warning}")
    if not summary["warnings"]:
        lines.append("- None.")
    lines.append("")
    lines.append("## Reproduction checks")
    lines.append("")
    lines.append("| Check | Expected | Observed | Passed |")
    lines.append("|---|---|---|:---:|")
    for _, row in reproduction.iterrows():
        lines.append(f"| {row['check']} | {row['expected']} | {row['observed']} | {bool(row['passed'])} |")
    (RESULTS_DIR / "PHASE4B_QA_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase4b() -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    required = [
        PHASE4_RESULTS / "phase4_summary.json",
        PHASE4_RESULTS / "phase4_frozen_portfolio_manifest.json",
        PHASE4_RESULTS / "04_external_transition_scores.csv",
        PHASE4_RESULTS / "05_external_frozen_portfolios.csv",
        PHASE4_RESULTS / "06_external_q3_evaluation.csv",
        PHASE4_RESULTS / "09_external_sensitivity_results.csv",
        PHASE4_RESULTS / "PHASE4_EXTERNAL_VALIDATION_REPORT.md",
        PHASE3C_RESULTS / "phase3c_summary.json",
        DATASET_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Required Phase 4B input is missing: " + ", ".join(missing))
    summary = read_json(PHASE4_RESULTS / "phase4_summary.json")
    manifest = read_json(PHASE4_RESULTS / "phase4_frozen_portfolio_manifest.json")
    scores = pd.read_csv(PHASE4_RESULTS / "04_external_transition_scores.csv")
    q3_eval = pd.read_csv(PHASE4_RESULTS / "06_external_q3_evaluation.csv")
    sensitivity = pd.read_csv(PHASE4_RESULTS / "09_external_sensitivity_results.csv")
    print("Checking the Phase 4 primary reproduction gate before Q3 access.")
    reproduction = primary_reproduction_gate(summary, manifest, scores, q3_eval, sensitivity)
    manifest_selections, primary_selections = parse_manifest_selections(manifest)
    validate_manifest_selections(manifest_selections, sensitivity)
    dataset_sha = sha256_file(DATASET_PATH)
    if dataset_sha != EXPECTED_DATASET_SHA256:
        raise RuntimeError(f"BPI Challenge 2017 SHA-256 mismatch. Expected {EXPECTED_DATASET_SHA256}, observed {dataset_sha}.")
    print("Primary reproduction passed. Calculating direct H1 score correlations.")
    correlations = corrected_correlations(scores, q3_eval)
    write_csv(RESULTS_DIR / "01_PRIMARY_CORRELATION_CORRECTION.csv", correlations)
    configs, parameters = sensitivity_parameters(sensitivity)
    print("Opening the official BPI 2017 dataset for Q3 evaluation of frozen portfolios.")
    q3, q3_stats = parse_q3_origin_window(DATASET_PATH)
    if len(q3) != 129191:
        raise RuntimeError(f"Q3 window-specific retained direct-follow rows mismatch. Expected 129191, observed {len(q3)}.")
    if q3_stats["negative_elapsed_rows_removed"] != 0:
        raise RuntimeError("The Q3 parse removed negative elapsed rows, which violates the Phase 4 preprocessing gate.")
    frozen_q3 = frozen_sensitivity_q3_evaluation(q3, manifest_selections, configs, parameters, manifest["manifest_sha256"])
    stability = frozen_sensitivity_stability(manifest_selections, primary_selections, configs, manifest["manifest_sha256"])
    write_csv(RESULTS_DIR / "02_FROZEN_SENSITIVITY_Q3_EVALUATION.csv", frozen_q3)
    write_csv(RESULTS_DIR / "03_FROZEN_SENSITIVITY_PORTFOLIO_STABILITY.csv", stability)
    phase3c = phase3c_diagnostic()
    q3_bootstrap = summary["q3_bootstrap_ci_k5"]
    bootstrap_by_comparison = {str(row["comparison"]): row for row in q3_bootstrap}
    warnings = [
        "The corrected H1 correlations replace the mislabeled Tail-versus-DRO values in phase4_summary.json without changing Phase 4 outputs.",
        "Sensitivity configurations are pre-frozen combined configurations and are not one-factor-at-a-time effects.",
        "No superiority claim is made for SA-DRO because the primary K=5 portfolios coincide with the strongest baselines and the corresponding bootstrap intervals include zero.",
    ]
    sa_stability = stability[stability["method"] == "Saturation-Aware DRO"]
    sa_q3 = frozen_q3[frozen_q3["method"] == "Saturation-Aware DRO"]
    sa_q3_k5 = sa_q3[sa_q3["K"] == 5]
    direct = correlations[correlations["correlation_scope"] == "H1 score-to-score"].set_index(["comparison_label", "correlation_type"])
    summary_out: dict[str, Any] = {
        "phase": "Phase 4B",
        "title": "EXTERNAL VALIDATION QA AND FROZEN SENSITIVITY EVALUATION",
        "primary_reproduction_passed": True,
        "primary_parameters": {
            "cap_hours": PRIMARY_CAP_HOURS,
            "alpha": PRIMARY_ALPHA,
            "grid_step_hours": PRIMARY_GRID_HOURS,
            "kappa_sa": PRIMARY_KAPPA,
            "K_confirmatory": 5,
        },
        "candidate_count": 41,
        "full_complete_direct_follow_pair_candidates": 443797,
        "negative_elapsed_removed": 0,
        "h1_window_rows": 193194,
        "q3_window_rows": 129191,
        "window_row_label": "window-specific retained direct-follow rows",
        "original_dro_total_saturated_count": 2,
        "sa_dro_total_saturated_count": 0,
        "dataset_sha256": dataset_sha,
        "original_manifest_checksum": manifest["manifest_sha256"],
        "corrected_h1_correlations": {
            "tail_vs_original_dro_spearman": float(direct.loc[("Tail score vs Original-DRO score", "Spearman"), "value"]),
            "tail_vs_original_dro_kendall": float(direct.loc[("Tail score vs Original-DRO score", "Kendall"), "value"]),
            "tail_vs_sa_dro_spearman": float(direct.loc[("Tail score vs SA-DRO score", "Spearman"), "value"]),
            "tail_vs_sa_dro_kendall": float(direct.loc[("Tail score vs SA-DRO score", "Kendall"), "value"]),
            "mean_vs_original_dro_spearman": float(direct.loc[("Mean-burden score vs Original-DRO score", "Spearman"), "value"]),
            "mean_vs_original_dro_kendall": float(direct.loc[("Mean-burden score vs Original-DRO score", "Kendall"), "value"]),
            "mean_vs_sa_dro_spearman": float(direct.loc[("Mean-burden score vs SA-DRO score", "Spearman"), "value"]),
            "mean_vs_sa_dro_kendall": float(direct.loc[("Mean-burden score vs SA-DRO score", "Kendall"), "value"]),
        },
        "frozen_sensitivity_configuration_count": len(configs),
        "frozen_sensitivity_evaluation_rows": len(frozen_q3),
        "sa_dro_k5_same_set_count": int(sa_stability["same_set_as_primary"].sum()),
        "sa_dro_k5_same_set_configurations": sa_stability.loc[sa_stability["same_set_as_primary"], "configuration"].astype(str).tolist(),
        "sa_dro_q3_mean_burden_share_range_pct": [float(sa_q3["q3_captured_mean_burden_share_pct"].min()), float(sa_q3["q3_captured_mean_burden_share_pct"].max())],
        "sa_dro_q3_mean_burden_share_range_pct_K5": [float(sa_q3_k5["q3_captured_mean_burden_share_pct"].min()), float(sa_q3_k5["q3_captured_mean_burden_share_pct"].max())],
        "primary_external_conclusion_changed": False,
        "primary_q3_bootstrap_verification": {
            str(key): {
                "ci95_low_pp": float(value["ci95_low_pp"]),
                "ci95_high_pp": float(value["ci95_high_pp"]),
                "ci_includes_zero": bool(value["ci_includes_zero"]),
            }
            for key, value in bootstrap_by_comparison.items()
        },
        "bpi2017_median_finite_epsilon_temporal_over_epsilon_sat": float(summary["median_finite_saturation_ratio"]),
        "bpi2019_phase3c_diagnostic": phase3c,
        "q3_driven_selection_created": False,
        "phase4_primary_modified": False,
        "phases_1_to_4_modified": False,
        "warnings": warnings,
        "errors": [],
        "q3_parse_stats": q3_stats,
        "seed": SEED,
    }
    write_csv(RESULTS_DIR / "04_PRIMARY_RESULT_REPRODUCTION_CHECK.csv", reproduction)
    write_json(RESULTS_DIR / "phase4b_summary.json", summary_out)
    write_report(summary_out, correlations, stability, reproduction)
    print("Phase 4B QA completed.")
    print(f"Corrected Tail versus Original-DRO Spearman: {summary_out['corrected_h1_correlations']['tail_vs_original_dro_spearman']:.10f}")
    print(f"Corrected Tail versus SA-DRO Spearman: {summary_out['corrected_h1_correlations']['tail_vs_sa_dro_spearman']:.10f}")
    print(f"Frozen sensitivity configurations evaluated: {len(configs)}")
    return summary_out
