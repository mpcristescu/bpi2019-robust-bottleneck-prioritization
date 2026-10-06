"""Reproduce the availability-aware analysis and all numerical manuscript tables.

Run from the repository root with ``python -m analysis.run_analysis --help``.
No event-level observations, figures or manuscript-editing code are exported.
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from .data import availability_audit, checksum, historical, holdout, parse_log, times
from .evaluation import evaluate, paired_bootstrap
from .model import (METHODS, SCORE, SEED, VERSION, diagnostic, dual_lp, empirical_cvar,
                    fit, jaccard, selections, summarize, transport_threshold)


def save(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8")


def portfolio_frame(portfolios, label):
    return pd.DataFrame([{"log": label, "method": method, "K": k, "rank": rank, "transition": t}
                         for (method, k), chosen in portfolios.items()
                         for rank, t in enumerate(chosen, 1)])


def selection_bootstrap(prepared, totals, blocks, primary, portfolios, methods, replicates=500):
    rng = np.random.default_rng(SEED+11)
    overlaps = {key: [] for key in portfolios}
    inclusion = {key: Counter() for key in portfolios}
    eligibility = Counter()
    candidate_counts = []
    for rep in range(replicates):
        weights = np.bincount(rng.integers(0, blocks, size=blocks), minlength=blocks)
        fitted = summarize(prepared, totals, blocks, weights=weights)
        eligibility.update(fitted.index)
        candidate_counts.append(len(fitted))
        chosen = selections(fitted, methods)
        for key in portfolios:
            overlaps[key].append(jaccard(chosen[key], portfolios[key]))
            inclusion[key].update(chosen[key])
        if (rep+1) % 100 == 0:
            print("selection bootstrap", rep+1, flush=True)
    rows, probs = [], []
    for key, values in overlaps.items():
        method, k = key
        rows.append({"method": method, "K": k, "replicates": replicates,
                     "mean_jaccard": float(np.mean(values)), "p05_jaccard": float(np.quantile(values, .05)),
                     "p95_jaccard": float(np.quantile(values, .95)),
                     "median_eligible_candidates": float(np.median(candidate_counts)),
                     "min_eligible_candidates": min(candidate_counts), "max_eligible_candidates": max(candidate_counts)})
        for name, count in inclusion[key].items():
            probs.append({"method": method, "K": k, "transition": name,
                          "selection_probability": count/replicates,
                          "eligibility_probability": eligibility[name]/replicates,
                          "in_primary": name in portfolios[key]})
    return pd.DataFrame(rows), pd.DataFrame(probs)


def sampling_benchmark(prepared, primary, replicates=100):
    """Exchangeable-observation null with each transition's actual block counts.

    Permuting delays across the fixed origin-block slots preserves the pooled
    empirical distribution and all block sizes. It does not preserve case-level
    dependence and is a finite-sample benchmark, not a coverage guarantee.
    """
    rng = np.random.default_rng(SEED+23)
    rows = []
    for item in prepared:
        if item.transition not in primary.index:
            continue
        counts = item.counts.astype(int)
        ids = np.repeat(np.arange(len(item.x)), item.hist.sum(axis=0).astype(int))
        pooled = np.cumsum(item.hist.sum(axis=0))[:-1]/len(ids)
        valid = counts >= 20
        samples = []
        for _ in range(replicates):
            permuted = rng.permutation(ids)
            distances, offset = [], 0
            for n in counts:
                if n >= 20:
                    freq = np.bincount(permuted[offset:offset+n], minlength=len(item.x))
                    cdf = np.cumsum(freq)[:-1]/n
                    distances.append(float(np.sum(np.abs(cdf-pooled)*item.dx)))
                offset += n
            samples.append(float(np.quantile(distances, .90)))
        rows.append({"transition": item.transition, "observed_epsilon": primary.loc[item.transition, "epsilon"],
                     "null_median_epsilon": float(np.median(samples)), "null_p95_epsilon": float(np.quantile(samples, .95)),
                     "observed_above_null_p95": primary.loc[item.transition, "epsilon"] > np.quantile(samples, .95),
                     "replicates": replicates})
    return pd.DataFrame(rows)


def lp_checks(prepared, metrics, cap=2160, alpha=.95):
    rows = []
    for item in prepared:
        if item.transition not in metrics.index:
            continue
        p = item.hist.sum(axis=0)
        p /= p.sum()
        tail = empirical_cvar(np.repeat(item.x, item.hist.sum(axis=0).astype(int)), alpha)
        direct = transport_threshold(item.x, p, cap, alpha)
        closed = (1-alpha)*(cap-tail)
        if abs(direct-closed) > 1e-6:
            raise AssertionError("Transport threshold disagrees with closed form")
        rows.append({"transition": item.transition, "representation": "original_support",
                     "epsilon": metrics.loc[item.transition, "epsilon"], "threshold_transport": direct,
                     "threshold_closed_form": closed, "threshold_error": abs(direct-closed),
                     "lp_risk": np.nan, "closed_form_risk": metrics.loc[item.transition, "risk"], "risk_error": np.nan})
        grid = np.clip(np.rint(item.x/24)*24, 0, cap)
        x = np.unique(grid)
        mass = np.zeros(len(x))
        np.add.at(mass, np.searchsorted(x, grid), p)
        from .model import weighted_cvar, robust_cvar
        cv = weighted_cvar(x, mass, alpha)
        sat = (1-alpha)*(cap-cv)
        for epsilon in sorted(set((0., .5*sat, sat, 2*sat, float(metrics.loc[item.transition, "epsilon"])))):
            lp = dual_lp(x, mass, epsilon, cap, alpha)
            exact = robust_cvar(cv, epsilon, cap, alpha)
            if abs(lp-exact) > 1e-5:
                raise AssertionError(f"LP mismatch: {item.transition}, {epsilon}, {lp}, {exact}")
            rows.append({"transition": item.transition, "representation": "24h_grid",
                         "epsilon": epsilon, "threshold_transport": transport_threshold(x, mass, cap, alpha),
                         "threshold_closed_form": sat, "threshold_error": abs(transport_threshold(x, mass, cap, alpha)-sat),
                         "lp_risk": lp, "closed_form_risk": exact, "risk_error": abs(lp-exact)})
    for x, p in (([0., 1., 2.], [.90, .08, .02]), ([0., 2.], [.94, .06]),
                 ([0., 1.], [.5, .5]), ([2.], [1.]), ([0., 1., 2.], [.98, .015, .005])):
        from .model import weighted_cvar, robust_cvar
        cv = weighted_cvar(x, p, alpha)
        sat = (1-alpha)*(2-cv)
        for epsilon in (0., sat/2, sat, sat*2, .1):
            lp = dual_lp(x, p, epsilon, 2, alpha)
            exact = robust_cvar(cv, epsilon, 2, alpha)
            if abs(lp-exact) > 1e-7:
                raise AssertionError("Synthetic LP mismatch")
            rows.append({"transition": str(x), "representation": "synthetic_original_support", "epsilon": epsilon,
                         "threshold_transport": transport_threshold(np.array(x), np.array(p), 2, alpha),
                         "threshold_closed_form": sat, "threshold_error": abs(transport_threshold(np.array(x), np.array(p), 2, alpha)-sat),
                         "lp_risk": lp, "closed_form_risk": exact, "risk_error": abs(lp-exact)})
    return pd.DataFrame(rows)


def run_log(log, source, cache, output, bootstrap_replicates=1000, selection_replicates=500, noise_replicates=100):
    output.mkdir(parents=True, exist_ok=True)
    frame, meta = parse_log(source, log, cache)
    t = times(log)
    if log == "bpi2017":
        t["observation_end"] = pd.Timestamp(meta["observed_end"])
    methods = METHODS[:4] if log == "bpi2019" else METHODS
    train = historical(frame, t["start"], t["decision"], t["decision"])
    test = holdout(frame, t["decision"], t["q3_end"], t["observation_end"])
    audit = [availability_audit(frame, t["start"], t["decision"], t["decision"], "H1_snapshot")]
    for lag in (90, 120):
        audit.append(availability_audit(frame, t["start"], t["decision"]-pd.Timedelta(days=lag), t["decision"], f"H1_maturity_lag_{lag}d"))
    audit.append(availability_audit(frame, t["decision"], t["q3_end"], t["observation_end"]+pd.Timedelta(nanoseconds=1), "Q3_observation"))
    for lag in (90, 120):
        end = min(t["q3_end"], t["observation_end"]-pd.Timedelta(days=lag))
        audit.append(availability_audit(frame, t["decision"], end, t["observation_end"]+pd.Timedelta(nanoseconds=1), f"Q3_followup_{lag}d"))
    save(pd.DataFrame(audit), output/"availability_audit.csv")
    monthly = []
    for month in range(1, 7):
        start = pd.Timestamp(year=t["start"].year, month=month, day=1, tz="UTC")
        end = start+pd.offsets.MonthBegin(1)
        monthly.append(availability_audit(frame, start, end, t["decision"], f"H1_month_{month}"))
    save(pd.DataFrame(monthly), output/"availability_by_month.csv")
    metrics, prepared, totals, blocks = fit(train, t["start"], t["decision"])
    ports = selections(metrics, methods)
    # Freeze the complete historical output before any holdout calculation.
    save(metrics.reset_index(), output/"historical_scores.csv")
    save(portfolio_frame(ports, log), output/"portfolios.csv")
    frozen_hash = checksum(output/"portfolios.csv")
    evaluation, outcomes = evaluate(metrics, ports, test)
    save(evaluation, output/"holdout_evaluation.csv")
    save(outcomes.reset_index(), output/"holdout_transition_burdens.csv")
    save(lp_checks(prepared, metrics), output/"mathematical_checks.csv")
    overlap = []
    for k in (3,5,10):
        for left, right in combinations(methods, 2):
            overlap.append({"K": k, "left": left, "right": right, "jaccard": jaccard(ports[(left,k)], ports[(right,k)]),
                            "same_membership": set(ports[(left,k)]) == set(ports[(right,k)])})
    save(pd.DataFrame(overlap), output/"portfolio_overlap.csv")

    sensitivity = []
    configs = [("primary", {}), ("block_7d", {"block_days": 7}), ("block_28d", {"block_days": 28}),
               ("block_min_50", {"min_count": 50}), ("block_min_100", {"min_count": 100}),
               ("leave_one_block_out", {"reference": "leave_one_block_out"}),
               ("grid_12h", {"grid": 12}), ("grid_24h", {"grid": 24}),
               ("alpha_090", {"alpha": .90})]
    configs += [(f"radius_{value:g}", {"multiplier": value}) for value in (0., .5, 1.5, 2.)]
    configs += [(f"kappa_{value:.2f}", {"kappa": value}) for value in (.50, .75, .95, .99)]
    all_sensitivity_portfolios, risk_sensitivity = [], []
    for label, kwargs in configs:
        fitted, _, _, _ = fit(train, t["start"], t["decision"], fixed_candidates=metrics.index, **kwargs)
        chosen = selections(fitted, methods)
        result, _ = evaluate(fitted, chosen, test, alpha=kwargs.get("alpha", .95))
        diag = diagnostic(fitted)
        for method in methods:
            for k in (3,5,10):
                row = result.loc[(result.method == method) & (result.K == k)].iloc[0].to_dict()
                row.update(diag, configuration=label, strict_eligible_candidates=int(fitted.strict_eligibility.sum()),
                           jaccard_primary=jaccard(chosen[(method,k)], ports[(method,k)]))
                sensitivity.append(row)
        pf = portfolio_frame(chosen, log)
        pf["configuration"] = label
        all_sensitivity_portfolios.append(pf)
        risk = fitted.reset_index()[["transition", "cvar", "epsilon", "epsilon_sat", "rho", "risk", "sa_risk", "saturated"]].copy()
        risk["configuration"] = label
        risk_sensitivity.append(risk)
    # Both caps are evaluated on exactly the same adequately aged origin window.
    common_end = min(t["q3_end"], t["observation_end"]-pd.Timedelta(days=120))
    common_test = holdout(frame, t["decision"], common_end, t["observation_end"])
    for cap in (2160, 2880):
        fitted, _, _, _ = fit(train, t["start"], t["decision"], cap=cap, fixed_candidates=metrics.index)
        chosen = selections(fitted, methods)
        result, _ = evaluate(fitted, chosen, common_test, cap=cap)
        for method in methods:
            for k in (3,5,10):
                row = result.loc[(result.method == method) & (result.K == k)].iloc[0].to_dict()
                row.update(diagnostic(fitted), configuration=f"cap_{cap//24}d_common_followup",
                           jaccard_primary=jaccard(chosen[(method,k)], ports[(method,k)]),
                           strict_eligible_candidates=int(fitted.strict_eligibility.sum()))
                sensitivity.append(row)
    # Maturation changes the observable training population, not only the cap.
    maturity_metrics = []
    for lag in (90,120):
        end = t["decision"]-pd.Timedelta(days=lag)
        mature_train = historical(frame, t["start"], end, t["decision"])
        fitted, _, _, _ = fit(mature_train, t["start"], end)
        chosen = selections(fitted, methods)
        result, _ = evaluate(fitted, chosen, test)
        save(fitted.reset_index(), output/f"historical_scores_maturity_{lag}d.csv")
        pf = portfolio_frame(chosen, log)
        pf["configuration"] = f"maturity_{lag}d"
        all_sensitivity_portfolios.append(pf)
        for method in methods:
            for k in (3,5,10):
                row = result.loc[(result.method == method) & (result.K == k)].iloc[0].to_dict()
                row.update(diagnostic(fitted), configuration=f"maturity_{lag}d",
                           jaccard_primary=jaccard(chosen[(method,k)], ports[(method,k)]),
                           strict_eligible_candidates=len(fitted))
                sensitivity.append(row)
                maturity_metrics.append(row)
    save(pd.DataFrame(sensitivity), output/"sensitivity_summary.csv")
    save(pd.concat(risk_sensitivity, ignore_index=True), output/"sensitivity_transition_risks.csv")
    save(pd.concat(all_sensitivity_portfolios, ignore_index=True), output/"sensitivity_portfolios.csv")
    benchmark = sampling_benchmark(prepared, metrics, noise_replicates)
    save(benchmark, output/"sampling_variability_benchmark.csv")
    print(log, "primary", diagnostic(metrics), flush=True)
    print(evaluation.loc[evaluation.K == 5].to_string(index=False), flush=True)
    bs_results = []
    for scheme in ("calendar_7d", "calendar_14d", "calendar_28d", "case_cluster"):
        print(log, "holdout bootstrap", scheme, flush=True)
        boot = paired_bootstrap(test, ports, metrics.index, t["decision"], replicates=bootstrap_replicates, scheme=scheme)
        bs_results.append(boot)
    save(pd.concat(bs_results, ignore_index=True), output/"paired_bootstrap.csv")
    stability, probabilities = selection_bootstrap(prepared, totals, blocks, metrics, ports, methods, selection_replicates)
    save(stability, output/"selection_stability.csv")
    save(probabilities, output/"selection_probabilities.csv")
    source_meta = dict(meta, **{key: value.isoformat() for key, value in t.items()})
    summary = dict(log=log, version=VERSION, seed=SEED, source=source_meta, historical_observations=len(train),
                   q3_observations=len(test), historical_calendar_blocks=blocks, **diagnostic(metrics),
                   portfolio_sha256=frozen_hash, common_cap_origin_end=common_end.isoformat(),
                   noise_above_null95_count=int(benchmark.observed_above_null_p95.sum()),
                   noise_median_observed_to_null_ratio=float((benchmark.observed_epsilon/benchmark.null_median_epsilon.replace(0, np.nan)).median()),
                   methods=methods, bootstrap_replicates=bootstrap_replicates, selection_replicates=selection_replicates,
                   noise_replicates=noise_replicates, pending_does_not_imply_censoring=True,
                   outcome_population="Pairs with both events observed by the specified information cutoff")
    (output/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bpi2019", type=Path, required=True)
    parser.add_argument("--bpi2017", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True, help="Private cache outside the repository")
    parser.add_argument("--output", type=Path, default=Path("results/availability_analysis"))
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--selection-replicates", type=int, default=500)
    parser.add_argument("--noise-replicates", type=int, default=100)
    parser.add_argument("--log", choices=("bpi2019", "bpi2017", "both"), default="both")
    args = parser.parse_args()
    summaries = {}
    for log in ("bpi2019", "bpi2017"):
        if args.log not in (log, "both"):
            continue
        summaries[log] = run_log(log, getattr(args, log), args.cache, args.output/log,
                                 args.bootstrap_replicates, args.selection_replicates, args.noise_replicates)
    code_hashes = {path.name: checksum(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    manifest = {"version": VERSION, "seed": SEED, "code_sha256": code_hashes,
                "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "parameters": {"cap_hours": 2160, "alpha": .95, "block_days": 14, "minimum_observations": 500,
                               "minimum_block_observations": 20, "valid_block_fraction": .60, "radius_quantile": .90,
                               "kappa": .90, "portfolio_sizes": [3,5,10], "tie_break": "transition name ascending"},
                "sources": {log: summary["source"] for log, summary in summaries.items()},
                "outputs": {str(path.relative_to(args.output)): checksum(path) for path in sorted(args.output.rglob("*.csv"))}}
    (args.output/"analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
