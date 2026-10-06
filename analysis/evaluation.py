"""Holdout metrics and paired resampling with exact empirical tail calculations."""
from __future__ import annotations

from itertools import combinations
import math

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from .model import SCORE, SEED, empirical_cvar, jaccard, weighted_cvar


def outcomes(frame, cap=2160, alpha=0.95):
    work = frame.assign(y=np.minimum(frame.delay.to_numpy(), cap))
    rows = []
    for transition, group in work.groupby("transition", sort=True, observed=True):
        values = group.y.to_numpy()
        rows.append({"transition": transition, "n": len(values), "mean": float(values.mean()),
                     "cvar": empirical_cvar(values, alpha), "mean_burden": float(values.sum()),
                     "tail_burden": len(values)*empirical_cvar(values, alpha)})
    return pd.DataFrame(rows).set_index("transition")


def oracle(data, candidates, k, target):
    scoped = data.loc[data.index.intersection(candidates)] if candidates is not None else data
    return tuple(scoped.reset_index().sort_values([target, "transition"], ascending=[False, True]).transition.head(k))


def evaluate(metrics, portfolios, frame, cap=2160, alpha=0.95):
    data = outcomes(frame, cap, alpha)
    mean_total, tail_total = data.mean_burden.sum(), data.tail_burden.sum()
    common = metrics.index.intersection(data.index)
    rows = []
    for (method, k), chosen in portfolios.items():
        selected = data.reindex(sorted(chosen)).fillna(0)
        mo = oracle(data, metrics.index, k, "mean_burden")
        to = oracle(data, metrics.index, k, "tail_burden")
        mg = oracle(data, None, k, "mean_burden")
        tg = oracle(data, None, k, "tail_burden")
        mean_share = 100*selected.mean_burden.sum()/mean_total
        tail_share = 100*selected.tail_burden.sum()/tail_total
        mean_oracle = 100*data.reindex(mo).mean_burden.sum()/mean_total
        tail_oracle = 100*data.reindex(to).tail_burden.sum()/tail_total
        rows.append({"method": method, "K": k, "mean_share_pct": mean_share, "tail_share_pct": tail_share,
                     "mean_oracle_eligible_pct": mean_oracle, "tail_oracle_eligible_pct": tail_oracle,
                     "mean_oracle_global_pct": 100*data.reindex(mg).mean_burden.sum()/mean_total,
                     "tail_oracle_global_pct": 100*data.reindex(tg).tail_burden.sum()/tail_total,
                     "mean_regret_pp": max(0.0, mean_oracle-mean_share), "tail_regret_pp": max(0.0, tail_oracle-tail_share),
                     "jaccard_mean_oracle": jaccard(chosen, mo), "jaccard_tail_oracle": jaccard(chosen, to),
                     "kendall_mean": float(kendalltau(metrics.loc[common, SCORE[method]], data.loc[common, "mean_burden"]).statistic),
                     "kendall_tail": float(kendalltau(metrics.loc[common, SCORE[method]], data.loc[common, "tail_burden"]).statistic),
                     "ranking_common_transitions": len(common), "q3_transitions": len(data),
                     "q3_observations": len(frame), "candidate_mean_coverage_pct": 100*data.reindex(metrics.index).mean_burden.sum()/mean_total,
                     "candidate_tail_coverage_pct": 100*data.reindex(metrics.index).tail_burden.sum()/tail_total})
    return pd.DataFrame(rows), data


def paired_bootstrap(frame, portfolios, candidates, q3_start, cap=2160, alpha=0.95,
                     replicates=1000, scheme="calendar_7d", seed=SEED):
    """Recompute each transition CVaR on every common weighted resample.

    Calendar schemes attach a row to the block of its origin. A case crossing
    calendar blocks is split in these schemes. The case-cluster scheme assigns
    one weight to all included Q3 rows of each case and preserves that dependence.
    Oracle portfolios are recomputed inside every replicate on the eligible set.
    """
    work = frame.copy()
    work["y"] = np.minimum(work.delay.to_numpy(), cap)
    if scheme.startswith("calendar_"):
        days = int(scheme.split("_")[1].removesuffix("d"))
        work["cluster"] = ((work.origin-q3_start).dt.total_seconds()//(days*86400)).astype(int)
    elif scheme == "case_cluster":
        work["cluster"] = pd.factorize(work.case, sort=True)[0]
    else:
        raise ValueError(scheme)
    cluster_count = int(work.cluster.max())+1
    names = sorted(work.transition.unique())
    indices = {name: i for i, name in enumerate(names)}
    arrays = []
    for name in names:
        group = work.loc[work.transition == name].sort_values("y", kind="stable")
        arrays.append((group.y.to_numpy(), group.cluster.to_numpy()))
    chosen_indices = {key: np.array(sorted(indices[t] for t in chosen if t in indices), dtype=int)
                      for key, chosen in portfolios.items()}
    candidate_indices = np.array([indices[t] for t in candidates if t in indices], dtype=int)
    rng = np.random.default_rng(seed)
    samples = {key: np.zeros((replicates, 2)) for key in portfolios}
    oracle_samples = {k: np.zeros((replicates, 2)) for k in sorted({k for _, k in portfolios})}
    for rep in range(replicates):
        weights = np.bincount(rng.integers(0, cluster_count, size=cluster_count), minlength=cluster_count)
        mean_burdens, tail_burdens = np.zeros(len(names)), np.zeros(len(names))
        for i, (values, clusters) in enumerate(arrays):
            row_weights = weights[clusters]
            count = float(row_weights.sum())
            mean_burdens[i] = np.dot(values, row_weights)
            if count:
                tail_burdens[i] = count*weighted_cvar(values, row_weights, alpha, assume_sorted=True)
        mean_total, tail_total = mean_burdens.sum(), tail_burdens.sum()
        if not mean_total or not tail_total:
            raise RuntimeError("Empty bootstrap burden denominator")
        for key, selected in chosen_indices.items():
            samples[key][rep] = (100*mean_burdens[selected].sum()/mean_total,
                                  100*tail_burdens[selected].sum()/tail_total)
        for k in oracle_samples:
            oracle_samples[k][rep] = (100*np.sort(mean_burdens[candidate_indices])[-k:].sum()/mean_total,
                                      100*np.sort(tail_burdens[candidate_indices])[-k:].sum()/tail_total)
    rows = []
    for k in sorted({key[1] for key in portfolios}):
        methods = [method for method, size in portfolios if size == k]
        for left, right in combinations(methods, 2):
            differences = samples[(left, k)]-samples[(right, k)]
            same = set(portfolios[(left, k)]) == set(portfolios[(right, k)])
            if same and not np.all(differences == 0):
                raise AssertionError("Identical portfolios have nonzero paired differences")
            for target, col in (("mean", 0), ("tail", 1)):
                values = differences[:, col]
                rows.append({"scheme": scheme, "clusters": cluster_count, "replicates": replicates,
                             "K": k, "left": left, "right": right, "target": target,
                             "difference_bootstrap_mean_pp": float(values.mean()),
                             "ci_lower_pp": float(np.quantile(values, .025)),
                             "ci_upper_pp": float(np.quantile(values, .975)), "identical_membership": same})
        for method in methods:
            for target, col in (("mean", 0), ("tail", 1)):
                values = oracle_samples[k][:, col]-samples[(method, k)][:, col]
                rows.append({"scheme": scheme, "clusters": cluster_count, "replicates": replicates,
                             "K": k, "left": "Eligible oracle", "right": method, "target": target,
                             "difference_bootstrap_mean_pp": float(values.mean()),
                             "ci_lower_pp": float(np.quantile(values, .025)),
                             "ci_upper_pp": float(np.quantile(values, .975)), "identical_membership": False})
    return pd.DataFrame(rows)
