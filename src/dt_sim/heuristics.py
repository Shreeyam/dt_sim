"""Chain-length estimator and decision-rule scoring.

The two-boundary chain estimator is the paper's heuristic. It maps (lam, L, g)
-> E[K] where lam is the thinned (clear) Poisson rate, L is the raw observation
window, and g is the worst-case slew that bounds chain spacing. Both bracketing
scheduled accesses act as anchors, each contributing g of deterministic
clearance.

`score_lookahead` returns the per-pitch decision score: visible count for the
"always" variant, advantage minus opportunity cost for "renewal".
"""
import numpy as np
import scipy.stats as st


def chain_two_boundary(lam: float, L: float, g: float) -> float:
    if lam <= 0 or L <= 0 or g <= 0:
        return 0.0
    total = 0.0
    max_k = int(L / g) + 3
    for k in range(1, max_k + 1):
        remaining = L - (k + 1) * g
        if remaining <= 0:
            break
        p = float(st.gamma.cdf(remaining, a=k, scale=1.0 / lam))
        if p < 1e-15:
            break
        total += p
    return total


def score_lookahead(variant, *, n_visible, t_window, p_clear, n_sched, avg_u,
                    missed_utility, g_eff):
    if variant == "always":
        return float(n_visible)
    if variant != "renewal":
        raise ValueError(f"score_lookahead: variant {variant!r} not scoring-capable")
    if n_visible <= 0 or t_window <= 0 or p_clear <= 0:
        return -np.inf
    lam = p_clear * n_visible / t_window
    chain = chain_two_boundary(lam, t_window, g_eff)
    advantage = (chain - p_clear * n_sched) * avg_u
    return advantage - missed_utility
