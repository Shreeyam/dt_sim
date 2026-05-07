"""Smoke tests for dt_sim.

Run from project root:
    python3 -m pytest tests/ -v

Component tests run in <1s; the end-to-end test takes ~5-10s on the small
n_cities=500 / never config and is the only one that touches MILP.
"""
import datetime

import numpy as np
import pytest
import scipy.stats as st

from dt_sim import SimConfig, run_pipeline
from dt_sim.access import Access, Request
from dt_sim.config import VARIANTS
from dt_sim.heuristics import chain_two_boundary, score_lookahead


# ---------- SimConfig ----------

def test_default_agility_at_for():
    """agility(t_slew_at_for) should hit slew_at_for_s exactly at the FoR angle."""
    cfg = SimConfig()
    assert cfg.agility()(cfg.field_of_regard_deg) == pytest.approx(cfg.slew_at_for_s)


def test_agility_at_zero_is_t_s():
    cfg = SimConfig()
    assert cfg.agility()(0) == pytest.approx(cfg.t_s)


def test_t0_parses_iso():
    cfg = SimConfig(t0_iso="2024-06-15T12:00:00")
    assert cfg.t0 == datetime.datetime(2024, 6, 15, 12, 0, 0)


def test_invalid_variant_rejected():
    with pytest.raises(ValueError, match="variant"):
        SimConfig(variant="weitzman")


def test_all_listed_variants_accepted():
    for v in VARIANTS:
        SimConfig(variant=v)  # no exception


# ---------- Access.aid ----------

def test_aid_is_unique_and_monotone():
    r = Request(0, 0.0, 0.0, "test")
    a1 = Access(r, datetime.datetime(2025, 1, 1), 0)
    a2 = Access(r, datetime.datetime(2025, 1, 1), 0)
    assert a1.aid != a2.aid
    assert a2.aid > a1.aid


def test_aid_survives_dict_roundtrip():
    """The whole point of aid: keying is stable even after copying."""
    import copy
    r = Request(0, 0.0, 0.0, "test")
    a = Access(r, datetime.datetime(2025, 1, 1), 0)
    a_copy = copy.copy(a)
    assert a.aid == a_copy.aid     # aid travels
    assert id(a) != id(a_copy)     # but the object id does not


# ---------- Chain estimator ----------

def test_chain_zero_rate():
    assert chain_two_boundary(0.0, 100.0, 5.0) == 0.0


def test_chain_nonneg():
    for lam in [0.01, 0.1, 1.0]:
        for L in [10, 100, 1000]:
            for g in [1, 5, 25]:
                assert chain_two_boundary(lam, L, g) >= 0


def test_chain_window_too_small():
    """If L < 2g there's no room for any chain point between two anchors."""
    assert chain_two_boundary(1.0, 5.0, 5.0) == 0.0


def test_chain_matches_mc_one_case():
    """Spot-check the analytic two-boundary form against a 5k-trial MC."""
    lam, L, g = 0.05, 200.0, 10.0
    analytic = chain_two_boundary(lam, L, g)

    rng = np.random.default_rng(0)
    n_trials = 5000
    chain_lengths = []
    for _ in range(n_trials):
        # Generate Poisson arrivals in (0, L) then enforce min-gap g from anchors at 0 and L.
        n = rng.poisson(lam * L)
        if n == 0:
            chain_lengths.append(0)
            continue
        arrivals = np.sort(rng.uniform(0, L, n))
        # Greedy chain build: skip any arrival < g from previous accepted (or from 0)
        prev = 0.0
        accepted = 0
        for x in arrivals:
            if x - prev >= g and L - x >= g:
                accepted += 1
                prev = x
        chain_lengths.append(accepted)
    mc_mean = float(np.mean(chain_lengths))
    # Tolerance accounts for both MC noise (~0.05) and the chain-vs-renewal abstraction gap.
    assert abs(analytic - mc_mean) < 0.5, f"analytic={analytic:.3f}, mc={mc_mean:.3f}"


# ---------- score_lookahead ----------

def test_always_returns_visible_count():
    score = score_lookahead("always", n_visible=7, t_window=100, p_clear=0.5,
                            n_sched=2, avg_u=1.0, missed_utility=0.0, g_eff=10)
    assert score == 7.0


def test_renewal_negative_when_dominated_by_baseline():
    """If the existing schedule already saturates, renewal advantage is non-positive."""
    score = score_lookahead("renewal", n_visible=2, t_window=100, p_clear=0.5,
                            n_sched=10, avg_u=1.0, missed_utility=0.0, g_eff=10)
    assert score < 0


# ---------- End-to-end (slow but worth it) ----------

@pytest.mark.slow
def test_never_e2e_no_lookaheads():
    """Tiny `never` run completes and never invokes the lookahead loop."""
    cfg = SimConfig(
        variant="never",
        n_trials=1,
        n_cities=500,
        horizon_h=6.0,
        tag="pytest_never",
    )
    result = run_pipeline(cfg)
    assert len(result.trials) == 1
    t = result.trials[0]
    assert t.n_lookaheads == 0
    assert t.dt_clear == t.conv_clear   # never policy: dt and conv are identical
    assert t.omni_clear >= t.conv_clear  # omni is upper bound


@pytest.mark.slow
def test_omniscient_dominates_never():
    """Omniscient should be >= never on the same scenario by construction."""
    cfg_never = SimConfig(variant="never", n_trials=1, n_cities=500,
                          horizon_h=6.0, tag="pytest_never2")
    cfg_omni = SimConfig(variant="omniscient", n_trials=1, n_cities=500,
                         horizon_h=6.0, tag="pytest_omni")
    r_n = run_pipeline(cfg_never).trials[0]
    r_o = run_pipeline(cfg_omni).trials[0]
    assert r_o.dt_clear >= r_n.dt_clear
