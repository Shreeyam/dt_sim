"""Regenerate ``optimal_timing.pdf`` using a real MILP schedule.

Loads 10,000 world cities, propagates a 400 km ISS-inclination orbit for 24 h,
produces accesses with random occlusion states, and runs the conventional MILP
scheduler. Picks one representative gap from the output, sweeps the lookahead
start time ``t`` within the gap, and computes ``J+``, ``J-``, and the net
``DJ`` at each sample using the renewal-chain formula (for advantage) and the
actual next-scheduled-tasks count (for opportunity cost).
"""
from __future__ import annotations

import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  # registers the "science" style
from scipy.special import gammainc

from dt_sim.access import get_accesses
from dt_sim.constants import Constants
from dt_sim.orbits import circular_orbit
from dt_sim.scheduling import load_worldcities, milp_schedule

REPO_ROOT = Path(__file__).resolve().parents[2]
MPL_STYLE = REPO_ROOT / "shreeyam.mplstyle"
OUT_DIR = Path(__file__).resolve().parents[1] / "figures"

plt.style.use(["science", "grid", str(MPL_STYLE)])


# ---------------------------------------------------------------------------
# Geometry helpers (same as regenerate_lookahead_figures).
# ---------------------------------------------------------------------------


def subtending_angle_from_fov(fov: float, h: float) -> float:
    R_E = Constants.R_E
    sign = np.sign(fov)
    angle = np.abs(fov)
    beta = np.pi - np.arcsin((R_E + h) / R_E * np.sin(angle))
    theta = np.pi - (angle + beta)
    return theta * sign


def calc_lookahead_time(angle_rad: float, h: float) -> float:
    R_E = Constants.R_E
    thetadot = np.sqrt(Constants.mu / (R_E + h) ** 3)
    max_angle = np.arcsin(R_E / (R_E + h))
    if angle_rad > max_angle:
        angle_rad = max_angle - 1e-9
    return subtending_angle_from_fov(angle_rad, h) / thetadot


def expected_chain_two_boundary(lam: float, L: float, g: float) -> float:
    """Expected internal chain points between two anchors, under the
    two-boundary form: sum_{k>=1} gamma_reg(k, lam (L - (k+1) g))."""
    if lam <= 0 or L <= 0 or g <= 0:
        return 0.0
    total = 0.0
    max_k = int(L / g) + 3
    for k in range(1, max_k + 1):
        remaining = L - (k + 1) * g
        if remaining <= 0:
            break
        val = float(gammainc(k, lam * remaining))
        if val < 1e-15:
            break
        total += val
    return total


def advantage(N_obs: float, L_act: float, g: float, p: float,
              N_sched: int) -> float:
    """Two-boundary advantage with schedule-baseline subtraction:
        hat_J^+ / cbar = E[K; lambda_c, L_act, g] - p * N_sched.
    The scheduled-in-window count N_sched is read directly from the existing
    schedule (the operational pipeline's baseline)."""
    if L_act <= 0 or N_obs < 1:
        return -p * N_sched
    lam_c = p * N_obs / L_act
    chain_term = expected_chain_two_boundary(lam_c, L_act, g)
    return chain_term - p * N_sched


# ---------------------------------------------------------------------------
# Simulation.
# ---------------------------------------------------------------------------


# Agile-class bang-bang agility from method.tex (25 s full-FoR, 10 s settle).
# Solves 25 = 10 + BETA * sqrt(pi/2)  =>  BETA = 15 / sqrt(pi/2).
T_S = 10.0
BETA = 15.0 / np.sqrt(np.pi / 2)


def agility_bangbang(theta_rad: float) -> float:
    return T_S + BETA * np.sqrt(np.abs(theta_rad))


def build_schedule(h_km: float, t0: datetime.datetime, t_end: datetime.datetime):
    requests = load_worldcities(10_000)
    orbit = circular_orbit(
        a=Constants.R_E + h_km,
        i=np.deg2rad(51.6),
        Omega=np.pi,
        M=np.pi / 2,
        t=t0,
    )
    accesses = get_accesses(requests, orbit, 500, 30, t0, t_end)
    rng = np.random.default_rng(0)
    for a in accesses:
        a.state = {"occluded": int(rng.integers(0, 2)), "observed": False}

    schedule = milp_schedule(accesses, requests, agility_bangbang)
    schedule = sorted(schedule, key=lambda a: a.time)
    return accesses, schedule, t0, t_end


def gap_duration_s(a_prev, a_next) -> float:
    """Duration of the idle interval [t_prev + t_s, t_next] per method.tex §4.4."""
    return (a_next.time - a_prev.time).total_seconds() - T_S


def collect_valid_gaps(
    schedule,
    accesses,
    t0: datetime.datetime,
    *,
    min_gap: float,
    max_gap: float,
    t_horizon: float,
    min_ahead: int,
):
    """Return all gaps with duration in [min_gap, max_gap] and at least
    ``min_ahead`` accesses in the next ``t_horizon`` seconds."""
    access_times = np.array([(a.time - t0).total_seconds() for a in accesses])
    kept = []
    for a_prev, a_next in zip(schedule[:-1], schedule[1:]):
        g = gap_duration_s(a_prev, a_next)
        if not (min_gap <= g <= max_gap):
            continue
        t_next_s = (a_next.time - t0).total_seconds()
        n_ahead = int(
            np.sum(
                (access_times >= t_next_s)
                & (access_times <= t_next_s + t_horizon)
            )
        )
        if n_ahead < min_ahead:
            continue
        kept.append((g, a_prev, a_next))
    return kept


# ---------------------------------------------------------------------------
# Plot.
# ---------------------------------------------------------------------------


def curves_for_gap(
    a_prev, a_next, *,
    access_times: np.ndarray,
    schedule_times: np.ndarray,
    t0: datetime.datetime,
    offsets: np.ndarray,
    t_horizon: float,
    t_man: float,
    g: float,
    p: float,
    cbar: float,
):
    """Evaluate J+, J-, net at offsets relative to t_star for one gap.

    ``offsets`` is an array of (t - t_star) values. Points outside the
    physically valid range (``t < gap_start`` or ``t > t_next``) return NaN.
    """
    t_prev_s = (a_prev.time - t0).total_seconds()
    t_next_s = (a_next.time - t0).total_seconds()
    gap_start = t_prev_s + T_S
    t_star = t_next_s - t_man

    ts = t_star + offsets
    valid = (ts >= gap_start) & (ts <= t_next_s)

    j_plus = np.full_like(offsets, np.nan, dtype=float)
    j_minus = np.full_like(offsets, np.nan, dtype=float)

    for i, t in enumerate(ts):
        if not valid[i]:
            continue
        L_act = max(t + t_horizon - t_next_s, 0.0)
        N_obs = float(
            np.sum((access_times >= t_next_s) & (access_times <= t + t_horizon))
        )
        # Scheduled accesses inside the actionable observation window:
        # the existing schedule's in-window count is the renewal baseline.
        N_sched = int(
            np.sum(
                (schedule_times >= t_next_s)
                & (schedule_times <= t + t_horizon)
            )
        )
        j_plus[i] = cbar * advantage(N_obs, L_act, g, p, N_sched)

        if t + t_man > t_next_s:
            n_missed = int(
                np.sum(
                    (schedule_times >= t_next_s)
                    & (schedule_times <= t + t_man)
                )
            )
            j_minus[i] = p * cbar * n_missed
        else:
            j_minus[i] = 0.0

    return j_plus, j_minus, j_plus - j_minus


def plot_from_schedule(
    out_path: Path,
    *,
    h_km: float = 400.0,
    alpha_deg: float = 65.0,
    p: float = 0.5,
    cbar: float = 1.0,
    min_gap: float = 120.0,
    max_gap: float = 250.0,
    min_ahead: int = 6,
    offset_min: float = -120.0,
    offset_max: float = 30.0,
    n_offset: int = 200,
) -> None:
    t0 = datetime.datetime(2024, 1, 1, 0, 0, 0)
    t_end = datetime.datetime(2024, 1, 2, 0, 0, 0)
    accesses, schedule, t0, t_end = build_schedule(h_km, t0, t_end)

    t_horizon = calc_lookahead_time(np.deg2rad(90), h_km)
    t_slew_alpha = agility_bangbang(np.deg2rad(alpha_deg))
    t_man = 2 * (t_slew_alpha - T_S)
    g = agility_bangbang(np.deg2rad(90))  # renewal dead time

    gaps = collect_valid_gaps(
        schedule, accesses, t0,
        min_gap=min_gap, max_gap=max_gap,
        t_horizon=t_horizon, min_ahead=min_ahead,
    )
    if not gaps:
        raise RuntimeError("No valid gaps for ensemble")

    access_times = np.array([(a.time - t0).total_seconds() for a in accesses])
    schedule_times = np.array([(s.time - t0).total_seconds() for s in schedule])
    offsets = np.linspace(offset_min, offset_max, n_offset)

    all_jp, all_jm, all_net = [], [], []
    for _, a_prev, a_next in gaps:
        jp, jm, net = curves_for_gap(
            a_prev, a_next,
            access_times=access_times, schedule_times=schedule_times,
            t0=t0, offsets=offsets, t_horizon=t_horizon,
            t_man=t_man, g=g, p=p, cbar=cbar,
        )
        all_jp.append(jp); all_jm.append(jm); all_net.append(net)
    JP = np.vstack(all_jp)
    JM = np.vstack(all_jm)
    NET = np.vstack(all_net)

    mean_jp, std_jp = np.nanmean(JP, axis=0), np.nanstd(JP, axis=0)
    mean_jm, std_jm = np.nanmean(JM, axis=0), np.nanstd(JM, axis=0)
    mean_net, std_net = np.nanmean(NET, axis=0), np.nanstd(NET, axis=0)

    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    # Only the advantage carries meaningful between-gap variance;
    # opp. cost and net inherit it, so shade only J+.
    ax.fill_between(
        offsets, mean_jp - std_jp, mean_jp + std_jp,
        color="C0", alpha=0.15, linewidth=0,
    )
    ax.plot(offsets, mean_jp, color="C0", linewidth=1.3,
            label=r"Advantage $\hat{J}^{+}$")
    ax.plot(offsets, mean_jm, color="C1", linewidth=1.3,
            label=r"Opp.\ cost $\hat{J}^{-}$")
    ax.plot(offsets, mean_net, color="k", linewidth=1.6,
            label=r"Net $\widehat{\Delta J}$")

    # t_star is offset=0.
    ax.axvline(0, linestyle="--", color="0.35", linewidth=0.8, zorder=0)
    peak_idx = int(np.nanargmax(mean_net))
    ax.plot(offsets[peak_idx], mean_net[peak_idx],
            marker="*", markersize=11, color="#ffbf00",
            markeredgecolor="k", markeredgewidth=0.5, linestyle="", zorder=5)
    ax.text(-3, ax.get_ylim()[1] * 0.04 if ax.get_ylim()[1] > 0 else 0.1,
            r"$t^\star$", fontsize=10, color="0.2", ha="right")

    ax.axhline(0, color="0.6", linewidth=0.5, zorder=0)
    ax.set_xlim(offset_min, offset_max)
    ax.set_xlabel(r"Lookahead offset $t - t^\star$ [s]")
    ax.set_ylabel(r"Expected clear captures")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path)
    plt.close(fig)

    print(
        f"N gaps: {len(gaps)} | t_man: {t_man:.1f} s | "
        f"offset of peak: {offsets[peak_idx]:.1f} s"
    )


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_from_schedule(OUT_DIR / "optimal_timing.pdf")
    print(f"Wrote {OUT_DIR/'optimal_timing.pdf'}")
