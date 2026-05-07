"""Receding-horizon dynamic tasking simulator.

Steps through the conventional schedule one task at a time. At each schedule
gap, scores candidate pitch directions via the heuristic, picks the best (or
skips), executes the lookahead, runs a local MILP repair, and splices the
repaired slice back into the schedule.
"""
import datetime
import json
import time as clock
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, List

import numpy as np

from .cameras import (filter_accesses_horizon, get_intrinsics_from_fov,
                      project_in_box)
from .config import SimConfig
from .heuristics import score_lookahead
from .orbits import eci2ecef, horizon_angle, kepler2eci, propagate_orbit
from .scenario import Scenario, build_scenario, reset_states
from .scheduling import load_worldcities, milp_schedule


@dataclass
class TrialResult:
    raan: float
    n_lookaheads: int
    dt_clear: float
    conv_clear: float
    omni_clear: float
    elapsed_s: float


@dataclass
class RunResult:
    cfg: dict
    trials: List[TrialResult] = field(default_factory=list)

    def summary(self) -> dict:
        if not self.trials:
            return {}
        dc = float(np.mean([t.dt_clear for t in self.trials]))
        cc = float(np.mean([t.conv_clear for t in self.trials]))
        oc = float(np.mean([t.omni_clear for t in self.trials]))
        nl = float(np.mean([t.n_lookaheads for t in self.trials]))
        gap = oc - cc
        gap_closed = (dc - cc) / gap * 100 if gap > 0.5 else 0.0
        return {
            "dt_clear_mean": dc,
            "conv_clear_mean": cc,
            "omni_clear_mean": oc,
            "n_lookaheads_mean": nl,
            "gap_closed_pct": float(gap_closed),
            "n_trials": len(self.trials),
        }


def _acc_tuples(accesses):
    """Build the (ecef, angle, time, access, idx) tuple list filter_accesses_horizon expects.

    `a.ecef` is cached at Access construction, so this is just attribute lookups.
    """
    return [(a.ecef, a.angle, a.time, a, i) for i, a in enumerate(accesses)]


def simulate_dt(scenario: Scenario, cfg: SimConfig):
    """Run one trial of the given decision variant on the prepared scenario.

    Returns (final_schedule, n_lookaheads).
    """
    width, height = 800, 800
    K = get_intrinsics_from_fov(cfg.fov_deg, width, height)
    h_angle = horizon_angle(scenario.orbit)
    boresight_pitch = cfg.boresight_pitch
    p_clear = 1.0 - cfg.cloud_prob
    agility = cfg.agility()
    g_eff = agility(cfg.field_of_regard_deg)
    agility_at_zero = agility(0)
    pitch_grid = np.linspace(10, np.rad2deg(h_angle) * 0.9, cfg.pitch_candidates)

    schedule = list(scenario.schedule_conv)
    accesses = scenario.accesses
    cloud_states = scenario.cloud_states
    requests = scenario.requests
    # Tuple form is invariant across the simulation (lat/long/angle/time of an
    # Access don't change). Build once and let filter_accesses_horizon slice it.
    all_tuples = _acc_tuples(accesses)
    n_lookaheads = 0
    current_idx = 0

    while current_idx < len(schedule) - 1:
        time = schedule[current_idx].time
        orbit_now = propagate_orbit(scenario.orbit, time)
        r, _ = kepler2eci(orbit_now)
        pos_ecef = eci2ecef(r, time)

        acc_filtered = filter_accesses_horizon(
            orbit_now, time, all_tuples, pos_ecef, cfg.field_of_regard_deg)
        if not acc_filtered:
            current_idx += 1
            continue

        best_pitch = None
        best_value = -np.inf
        best_in_box = []
        best_acc_filtered = []

        for pitch_deg in pitch_grid:
            t_maneuver = (agility(pitch_deg - boresight_pitch) - agility_at_zero) * 2
            obs_time = time + datetime.timedelta(seconds=t_maneuver / 2)
            orbit_obs = propagate_orbit(scenario.orbit, obs_time)
            r_obs, _ = kepler2eci(orbit_obs)
            pos_ecef_obs = eci2ecef(r_obs, obs_time)

            acc_obs = filter_accesses_horizon(
                orbit_obs, obs_time, all_tuples,
                pos_ecef_obs, cfg.field_of_regard_deg)
            if not acc_obs:
                continue
            points_obs = np.array([rr for rr, _, _, _, _ in acc_obs])

            _, in_box, _ = project_in_box(
                pitch_deg, 0.0, orbit_obs, obs_time,
                acc_obs, points_obs, width, height, K)
            if len(in_box) == 0:
                continue

            n_visible = len(in_box)
            vis_times = [(acc_obs[i][-2].time - obs_time).total_seconds()
                         for i in in_box]
            t_window = max(vis_times) - min(vis_times) if len(vis_times) > 1 else 1.0
            avg_u = float(np.mean([acc_obs[i][-2].utility for i in in_box]))

            min_obs = obs_time + datetime.timedelta(seconds=min(vis_times))
            max_obs = obs_time + datetime.timedelta(seconds=max(vis_times))
            n_sched = sum(1 for s in schedule if min_obs <= s.time <= max_obs)

            missed_utility = sum(
                s.utility for s in schedule[current_idx + 1:]
                if (s.time - time).total_seconds() < t_maneuver
                and cloud_states.get(s.aid, False)
            )

            value = score_lookahead(
                cfg.variant,
                n_visible=n_visible,
                t_window=t_window,
                p_clear=p_clear,
                n_sched=n_sched,
                avg_u=avg_u,
                missed_utility=missed_utility,
                g_eff=g_eff,
            )

            if value > best_value:
                best_value = value
                best_pitch = pitch_deg
                best_in_box = in_box
                best_acc_filtered = acc_obs

        gate_open = cfg.variant == "always" or best_value > 0
        if best_pitch is None or not gate_open:
            current_idx += 1
            continue
        if len(best_in_box) == 0:
            current_idx += 1
            continue

        n_lookaheads += 1
        t_maneuver_chosen = (agility(best_pitch - boresight_pitch) - agility_at_zero) * 2

        # Drop schedule items consumed by the maneuver
        consumed = []
        for j in range(current_idx + 1, len(schedule)):
            if (schedule[j].time - time).total_seconds() < t_maneuver_chosen:
                consumed.append(j)
            else:
                break
        for j in reversed(consumed):
            schedule.pop(j)

        for i in best_in_box:
            best_acc_filtered[i][-2].state["observed"] = True

        obs_times = [best_acc_filtered[i][-2].time for i in best_in_box]
        if not obs_times:
            current_idx += 1
            continue

        min_time, max_time = min(obs_times), max(obs_times)
        slice_idx = [i for i, a in enumerate(schedule)
                     if min_time <= a.time <= max_time]
        if not slice_idx:
            current_idx += 1
            continue

        idx_start, idx_end = min(slice_idx), max(slice_idx)
        scheduled_outside = {a.requestid for a in
                             schedule[:idx_start] + schedule[idx_end + 1:]}
        slice_accesses = [
            a for a in accesses
            if min_time <= a.time <= max_time
            and not (a.state.get("observed", False)
                     and not cloud_states.get(a.aid, True))
            and a.requestid not in scheduled_outside
        ]
        if not slice_accesses:
            current_idx += 1
            continue

        force_in = []
        if idx_start > 0:
            slice_accesses.insert(0, schedule[idx_start - 1])
            force_in.append(schedule[idx_start - 1])
        if idx_end < len(schedule) - 1:
            slice_accesses.append(schedule[idx_end + 1])
            force_in.append(schedule[idx_end + 1])

        new_slice = milp_schedule(slice_accesses, requests, agility,
                                  force_in if force_in else None)
        if force_in:
            new_slice = [a for a in new_slice if a not in force_in]

        schedule = schedule[:idx_start] + new_slice + schedule[idx_end + 1:]
        current_idx += 1

    return schedule, n_lookaheads


def _evaluate_variant(scen, cfg: SimConfig):
    """Run one variant on a prepared scenario; return (schedule_final, n_looks)."""
    if cfg.variant == "never":
        return scen.schedule_conv, 0
    if cfg.variant == "omniscient":
        return scen.schedule_omni, 0
    reset_states(scen)
    return simulate_dt(scen, cfg)


def run_paired(cfg_base: SimConfig, variants: List[str], requests=None
               ) -> Dict[str, RunResult]:
    """Run multiple variants paired on the same scenarios.

    Builds each scenario once per RAAN and evaluates all variants on it.
    Saves one MILP baseline pair per RAAN instead of per (RAAN, variant) cell.
    Returns a dict mapping variant name -> RunResult.
    """
    if requests is None:
        requests = load_worldcities(n=cfg_base.n_cities)
    rng = np.random.default_rng(cfg_base.seed)
    raans = np.linspace(0, 2 * np.pi, cfg_base.n_trials, endpoint=False)

    cfgs = {v: replace(cfg_base, variant=v, tag=f"{cfg_base.tag}_{v}") for v in variants}
    results = {v: RunResult(cfg=cfgs[v].to_dict()) for v in variants}

    print(f"[{cfg_base.tag}] paired sweep: variants={variants}, "
          f"t_slew={cfg_base.slew_at_for_s}s, fov={cfg_base.fov_deg}°, "
          f"n_trials={cfg_base.n_trials}, horizon={cfg_base.horizon_h}h")

    for trial_idx, raan in enumerate(raans):
        t_build = clock.time()
        scen = build_scenario(requests=requests, raan=raan, cfg=cfg_base, rng=rng)
        if scen is None:
            print(f"  trial {trial_idx+1}/{cfg_base.n_trials}: skipped (insufficient accesses)")
            continue
        build_s = clock.time() - t_build
        conv_clear = sum(a.utility for a in scen.schedule_conv
                         if scen.cloud_states[a.aid])
        omni_clear = sum(a.utility for a in scen.schedule_omni
                         if scen.cloud_states[a.aid])
        gap = omni_clear - conv_clear

        print(f"  trial {trial_idx+1}/{cfg_base.n_trials} (RAAN={np.rad2deg(raan):.0f}°), "
              f"conv={conv_clear:.0f}, omni={omni_clear:.0f}, build={build_s:.1f}s")
        for v in variants:
            t_var = clock.time()
            sched, n_looks = _evaluate_variant(scen, cfgs[v])
            dt_clear = sum(a.utility for a in sched
                           if scen.cloud_states.get(a.aid, False))
            elapsed = clock.time() - t_var
            results[v].trials.append(TrialResult(
                raan=float(raan),
                n_lookaheads=int(n_looks),
                dt_clear=float(dt_clear),
                conv_clear=float(conv_clear),
                omni_clear=float(omni_clear),
                elapsed_s=float(elapsed),
            ))
            gc = (dt_clear - conv_clear) / gap * 100 if gap > 0.5 else 0
            print(f"    {v:<11} dt={dt_clear:.0f}, looks={n_looks}, "
                  f"gap={gc:.0f}% ({elapsed:.1f}s)")

    print(f"[{cfg_base.tag}] summary:")
    for v in variants:
        s = results[v].summary()
        if s:
            print(f"    {v:<11} dt={s['dt_clear_mean']:.1f}, "
                  f"looks={s['n_lookaheads_mean']:.1f}, "
                  f"gap={s['gap_closed_pct']:.1f}%")
    return results


def run_pipeline(cfg: SimConfig, requests=None) -> RunResult:
    """Run all trials for one config cell. Returns a RunResult."""
    if requests is None:
        requests = load_worldcities(n=cfg.n_cities)
    rng = np.random.default_rng(cfg.seed)
    raans = np.linspace(0, 2 * np.pi, cfg.n_trials, endpoint=False)
    result = RunResult(cfg=cfg.to_dict())

    print(f"[{cfg.tag}] variant={cfg.variant}, t_slew={cfg.slew_at_for_s}s, "
          f"fov={cfg.fov_deg}°, n_trials={cfg.n_trials}, horizon={cfg.horizon_h}h")

    for trial_idx, raan in enumerate(raans):
        t_trial = clock.time()
        scen = build_scenario(requests=requests, raan=raan, cfg=cfg, rng=rng)
        if scen is None:
            print(f"  trial {trial_idx+1}/{cfg.n_trials}: skipped (insufficient accesses)")
            continue

        conv_clear = sum(a.utility for a in scen.schedule_conv
                         if scen.cloud_states[a.aid])
        omni_clear = sum(a.utility for a in scen.schedule_omni
                         if scen.cloud_states[a.aid])

        if cfg.variant == "never":
            schedule_final, n_looks = scen.schedule_conv, 0
        elif cfg.variant == "omniscient":
            schedule_final, n_looks = scen.schedule_omni, 0
        else:
            reset_states(scen)
            schedule_final, n_looks = simulate_dt(scen, cfg)

        dt_clear = sum(a.utility for a in schedule_final
                       if scen.cloud_states.get(a.aid, False))
        elapsed = clock.time() - t_trial

        result.trials.append(TrialResult(
            raan=float(raan),
            n_lookaheads=int(n_looks),
            dt_clear=float(dt_clear),
            conv_clear=float(conv_clear),
            omni_clear=float(omni_clear),
            elapsed_s=float(elapsed),
        ))
        gap = omni_clear - conv_clear
        gc = (dt_clear - conv_clear) / gap * 100 if gap > 0.5 else 0
        print(f"  trial {trial_idx+1}/{cfg.n_trials} (RAAN={np.rad2deg(raan):.0f}°): "
              f"dt={dt_clear:.0f}, conv={conv_clear:.0f}, omni={omni_clear:.0f}, "
              f"looks={n_looks}, gap={gc:.0f}% ({elapsed:.1f}s)")

    s = result.summary()
    if s:
        print(f"[{cfg.tag}] mean: dt={s['dt_clear_mean']:.1f}, "
              f"looks={s['n_lookaheads_mean']:.1f}, gap={s['gap_closed_pct']:.1f}%")
    return result


def save_result(result: RunResult, cfg: SimConfig) -> Path:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg.tag}.json"
    payload = {
        "cfg": result.cfg,
        "summary": result.summary(),
        "trials": [asdict(t) for t in result.trials],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path
