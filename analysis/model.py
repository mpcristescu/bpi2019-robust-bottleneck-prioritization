"""Exact bounded-support Wasserstein-CVaR, historical calibration and ranking."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

VERSION = "1.1.0"
SEED = 20260914
METHODS = ("Frequency", "Mean burden", "Empirical Tail", "Original DRO", "Saturation-Aware DRO")
SCORE = dict(zip(METHODS, ("frequency", "mean_score", "tail_score", "dro_score", "sa_score")))


def weighted_cvar(x, weights, alpha=0.95, assume_sorted=False):
    x = np.asarray(x, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if not assume_sorted:
        order = np.argsort(x, kind="stable")
        x, weights = x[order], weights[order]
    x, weights = x[::-1], weights[::-1]
    mass = float(weights.sum()) * (1 - alpha)
    if mass <= 0:
        return float("nan")
    cumulative = np.cumsum(weights)
    index = min(int(np.searchsorted(cumulative, mass, side="left")), len(x) - 1)
    before = cumulative[index - 1] if index else 0.0
    return float((np.dot(x[:index], weights[:index]) + x[index] * (mass - before)) / mass)


def empirical_cvar(values, alpha=0.95):
    values = np.asarray(values, dtype=float)
    return weighted_cvar(values, np.ones(len(values)), alpha)


def robust_cvar(cvar, epsilon, cap, alpha=0.95):
    return float(min(cap, cvar + max(0.0, epsilon) / (1 - alpha)))


def transport_threshold(x, weights, cap, alpha=0.95):
    """Greedy minimum transport cost, independent of the closed-form calculation."""
    order = np.argsort(x)[::-1]
    x, p = np.asarray(x)[order], np.asarray(weights, dtype=float)[order]
    p = p / p.sum()
    missing = max(0.0, 1 - alpha - float(p[x == cap].sum()))
    cost = 0.0
    for point, probability in zip(x, p):
        if point == cap:
            continue
        moved = min(probability, missing)
        cost += moved * (cap - point)
        missing -= moved
        if missing <= 1e-14:
            break
    return float(cost)


def dual_lp(x, probabilities, epsilon, cap, alpha=0.95):
    """Finite exact dual for the hinge loss on a bounded interval.

    For each reference point x_i, s_i >= 0, s_i >= x_i-t, and
    s_i >= C-t-lambda*(C-x_i). The hinge minus transport penalty attains its
    maximum at x_i or C (the zero branch is covered by s_i >= 0).
    """
    x = np.asarray(x, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    n = len(x)
    objective = np.r_[1.0, epsilon / (1 - alpha), probabilities / (1 - alpha)]
    a = lil_matrix((2*n, n+2))
    rhs = np.zeros(2*n)
    for i, point in enumerate(x):
        a[2*i, 0] = -1
        a[2*i, i+2] = -1
        rhs[2*i] = -point
        a[2*i+1, 0] = -1
        a[2*i+1, 1] = -(cap-point)
        a[2*i+1, i+2] = -1
        rhs[2*i+1] = -cap
    solution = linprog(objective, A_ub=a.tocsr(), b_ub=rhs,
                       bounds=[(0, cap), (0, None)] + [(0, None)]*n, method="highs")
    if not solution.success:
        raise RuntimeError(solution.message)
    return float(solution.fun)


@dataclass
class Distribution:
    transition: str
    x: np.ndarray
    hist: np.ndarray
    counts: np.ndarray
    cdfs: np.ndarray
    dx: np.ndarray


def prepare(frame, origin_start, origin_end, cap=2160, block_days=14, grid=0):
    blocks = max(1, math.ceil((origin_end-origin_start).total_seconds()/(block_days*86400)))
    work = frame.copy()
    work["block"] = ((work.origin-origin_start).dt.total_seconds()//(block_days*86400)).astype(int)
    work["y"] = np.minimum(work.delay.to_numpy(dtype=float), cap)
    if grid:
        work["y"] = np.clip(np.rint(work.y/grid)*grid, 0, cap)
    prepared = []
    totals = np.bincount(work.block, minlength=blocks).astype(float)
    for name, group in work.groupby("transition", sort=True, observed=True):
        x, inverse = np.unique(group.y.to_numpy(), return_inverse=True)
        hist = np.zeros((blocks, len(x)), dtype=float)
        np.add.at(hist, (group.block.to_numpy(), inverse), 1.0)
        counts = hist.sum(axis=1)
        cdfs = np.cumsum(hist, axis=1)/np.maximum(counts[:, None], 1)
        prepared.append(Distribution(str(name), x, hist, counts, cdfs[:, :-1], np.diff(x)))
    return prepared, totals, blocks


def summarize(prepared, totals, blocks, cap=2160, alpha=0.95, min_count=20,
              reference="pooled", kappa=0.90, multiplier=1.0, weights=None,
              fixed_candidates=None):
    weights = np.ones(blocks) if weights is None else np.asarray(weights, dtype=float)
    required = math.ceil(0.60*float(weights.sum()))
    denominator = float(np.dot(totals, weights))
    rows = []
    fixed = None if fixed_candidates is None else set(fixed_candidates)
    for item in prepared:
        if fixed is not None and item.transition not in fixed:
            continue
        count = float(np.dot(item.counts, weights))
        valid = (item.counts >= min_count) & (weights > 0)
        valid_count = int(round(float(weights[valid].sum())))
        eligible = count >= 500 and valid_count >= required
        if (fixed is None and not eligible) or count <= 0 or not valid.any():
            continue
        pmf = weights @ item.hist
        pooled = np.cumsum(pmf)[:-1]/count
        if reference == "pooled":
            refs = pooled[None, :]
        elif reference == "leave_one_block_out":
            other = count-item.counts
            refs = (count*pooled[None, :]-item.counts[:, None]*item.cdfs)/np.maximum(other[:, None], 1)
            valid &= other > 0
        else:
            raise ValueError(reference)
        if not valid.any():
            continue
        distances = np.sum(np.abs(item.cdfs-refs)*item.dx[None, :], axis=1)
        repeated = np.repeat(distances[valid], weights[valid].astype(int))
        epsilon = float(np.quantile(repeated, 0.90)) * multiplier
        tail = weighted_cvar(item.x, pmf, alpha, assume_sorted=True)
        mean = float(np.dot(item.x, pmf)/count)
        threshold = max(0.0, (1-alpha)*(cap-tail))
        empirically_saturated = threshold <= 1e-8
        risk = robust_cvar(tail, epsilon, cap, alpha)
        sa_epsilon = min(epsilon, kappa*threshold) if threshold > 1e-8 else 0.0
        sa_risk = robust_cvar(tail, sa_epsilon, cap, alpha)
        frequency = count/denominator
        rows.append({"transition": item.transition, "n": int(round(count)), "valid_blocks": valid_count,
                     "required_valid_blocks": required, "strict_eligibility": eligible, "frequency": frequency,
                     "mean": mean, "cvar": tail, "epsilon": epsilon, "epsilon_sat": threshold,
                     "rho": epsilon/threshold if threshold > 1e-8 else np.nan,
                     "m_cap": float(pmf[item.x == cap].sum()/count),
                     "empirically_saturated": empirically_saturated,
                     "saturated": risk >= cap-1e-7, "sa_saturated": sa_risk >= cap-1e-7,
                     "risk": risk, "sa_epsilon": sa_epsilon, "sa_risk": sa_risk,
                     "mean_score": frequency*mean, "tail_score": frequency*tail,
                     "dro_score": frequency*risk, "sa_score": frequency*sa_risk})
    return pd.DataFrame(rows).set_index("transition")


def fit(frame, start, end, **kwargs):
    prep_keys = {key: kwargs[key] for key in ("cap", "block_days", "grid") if key in kwargs}
    prepared, totals, blocks = prepare(frame, start, end, **prep_keys)
    summary_keys = {key: value for key, value in kwargs.items() if key not in ("block_days", "grid")}
    return summarize(prepared, totals, blocks, **summary_keys), prepared, totals, blocks


def selections(metrics, methods=METHODS, sizes=(3, 5, 10)):
    result = {}
    for method in methods:
        ordered = metrics.reset_index().sort_values([SCORE[method], "transition"], ascending=[False, True], kind="stable")
        for k in sizes:
            result[(method, k)] = tuple(ordered.transition.head(k))
    return result


def jaccard(left, right):
    left, right = set(left), set(right)
    return len(left & right)/len(left | right) if left or right else 1.0


def diagnostic(metrics):
    return {"candidates": len(metrics), "empirical_saturated": int(metrics.empirically_saturated.sum()),
            "induced_saturated": int((metrics.saturated & ~metrics.empirically_saturated).sum()),
            "total_saturated": int(metrics.saturated.sum()), "sa_saturated": int(metrics.sa_saturated.sum()),
            "median_rho": float(metrics.rho.median()), "median_epsilon": float(metrics.epsilon.median())}
