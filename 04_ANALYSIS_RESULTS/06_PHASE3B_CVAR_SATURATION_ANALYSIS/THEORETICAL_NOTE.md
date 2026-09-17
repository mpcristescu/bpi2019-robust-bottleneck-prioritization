# Analytical Saturation Result for Bounded-Support Wasserstein-CVaR

## Setting

Let `P` be a probability distribution supported on `[0, C]`, let `alpha` belong to `(0, 1)`, and let

`m_C = P(X = C)`.

The upper tail used by `CVaR_alpha` has mass `1 - alpha`. The amount that must be placed at the cap in order for the whole upper tail to be located at `C` is

`delta = max(0, 1 - alpha - m_C)`.

The quantity `epsilon_sat(P, alpha, C)` is the minimum Wasserstein-1 transport cost needed to move this amount of probability to `C`. When `m_C >= 1 - alpha`, the threshold is zero because the empirical upper tail is already concentrated at the cap.

For `m_C < 1 - alpha`, the minimum cost is obtained by transporting probability from the part of the support closest to `C`. In quantile form,

`epsilon_sat(P, alpha, C) = integral from alpha to 1 - m_C of [C - F^{-1}(u)] du`.

For a grid distribution, the integral is evaluated exactly as a finite sum. Starting with the largest grid point strictly below `C`, probability is moved to `C` until `delta` is covered. The cost of moving mass `q` from `x` is `q(C - x)`.

## Saturation equivalence

The bounded-support Wasserstein-CVaR satisfies

`sup { CVaR_alpha(Q) : W1(Q, P) <= epsilon } = C`

if and only if

`epsilon >= epsilon_sat(P, alpha, C)`.

The upper bound follows immediately from the support restriction. No distribution in the ambiguity set can produce a CVaR larger than `C`.

Equality with `C` requires the entire upper tail of probability `1 - alpha` to be located at `C`. Any probability below the cap that remains in that tail lowers its average value. The empirical distribution already contributes `m_C` to the required mass. If `m_C < 1 - alpha`, an additional amount `delta = 1 - alpha - m_C` must therefore be transported to the cap.

The least expensive transport moves probability from the largest values below `C`. For any two source points `x_1 < x_2 < C`, moving equal mass from `x_2` costs less than moving it from `x_1`. Repeating this exchange argument yields the descending greedy construction used in the discrete implementation. Its cost is the quantile integral above. Consequently, the cap is attainable exactly when the Wasserstein radius reaches this minimum cost.

This statement is an analytical result derived and empirically verified in this study. It is not presented as a novelty claim pending the literature review.

## Scaling property

Let `s > 0`, define `Y = sX`, and let the cap be `sC`. Distances in the transport problem scale by `s`, while probability masses and the CVaR tail level remain unchanged. Therefore,

`epsilon_sat(sP, alpha, sC) = s epsilon_sat(P, alpha, C)`.

The same relation follows directly from the quantile representation because `F_Y^{-1}(u) = sF_X^{-1}(u)`.

## Empirical verification protocol

The implementation discretizes each H1 transition distribution on the 24-hour grid ending at `C = 2160` hours. It computes the finite transport sum, then evaluates the worst-case CVaR with an explicit linear program at zero radius, at `epsilon_sat`, and at `0.99 epsilon_sat`. The zero-radius value is compared with grid empirical CVaR. The threshold value is compared with the cap. The near-threshold value is checked to remain below the cap outside the declared numerical tolerance.

The synthetic verification uses deterministic grids on `[0, 1]` for four Beta distributions and two mixtures with a point mass at one. The threshold transition is checked for alpha equal to `0.90`, `0.95`, and `0.99`.
