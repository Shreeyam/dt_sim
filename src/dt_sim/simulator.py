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
from .heuristics import geometric_lookahead_window, score_lookahead
from .orbits import _era_rad, eci2ecef, horizon_angle, kepler2eci, propagate_orbit
from .scenario import Scenario, build_scenario, reset_states, sample_trial_datetimes
from .scheduling import load_worldcities, milp_schedule


@dataclass
class TrialResult:
    raan: float
    mean_anom: float
    t0_iso: str
    n_lookaheads: int
    dt_clear: float
    conv_clear: float
    omni_clear: float
    elapsed_s: float
    n_missed: int = 0
    n_missed_clear: int = 0
    n_scheduled: int = 0
    n_scheduled_conv: int = 0
    n_scheduled_omni: int = 0
    n_lookahead_obs: int = 0
    n_lookahead_clear: int = 0
    lookahead_events: List[dict] = field(default_factory=list)


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


def simulate_dt(scenario: Scenario, cfg: SimConfig) -> dict:
    """Run one trial of the given decision variant on the prepared scenario."""
    width, height = 800, 800
    K = get_intrinsics_from_fov(cfg.fov_deg, width, height)
    h_angle = horizon_angle(scenario.orbit)
    boresight_pitch = cfg.boresight_pitch
    p_clear = 1.0 - cfg.cloud_prob
    agility = cfg.agility()
    g_eff = agility(cfg.field_of_regard_deg)
    agility_at_zero = agility(0)
    pitch_max = np.rad2deg(h_angle) * 0.9
    if cfg.cap_far_edge_at_limb:
        pitch_max = min(pitch_max, np.rad2deg(h_angle) - cfg.fov_deg / 2)
    if pitch_max < boresight_pitch:
        pitch_grid = np.array([boresight_pitch])
    else:
        pitch_grid = np.linspace(boresight_pitch, pitch_max,
                                 cfg.pitch_candidates)
    greedy_rules = ("greedy",)
    value_rules = ("two_anchor", "break_even")
    optimal_timing_rules = value_rules

    schedule = list(scenario.schedule_conv)
    accesses = scenario.accesses
    cloud_states = scenario.cloud_states
    requests = scenario.requests
    all_tuples = _acc_tuples(accesses)
    n_lookaheads = 0
    n_missed_total = 0
    n_missed_clear_total = 0
    n_lookahead_obs_total = 0
    n_lookahead_clear_total = 0
    lookahead_events = []
    current_idx = -1

    while current_idx < len(schedule) - 1:
        if current_idx < 0:
            t_prev = scenario.orbit.t
            t_next = schedule[0].time
        else:
            t_prev = schedule[current_idx].time
            t_next = schedule[current_idx + 1].time
        gap = (t_next - t_prev).total_seconds() - agility_at_zero
        time = t_prev
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
        best_event = None
        baseline_start = t_prev
        baseline_end = t_next
        n_sched_baseline = 0
        if cfg.variant in value_rules:
            max_support_end = t_next
            for pitch_baseline in pitch_grid:
                t_maneuver_baseline = (
                    agility(pitch_baseline - boresight_pitch) - agility_at_zero
                ) * 2
                if t_maneuver_baseline > gap:
                    continue
                obs_baseline = t_next - datetime.timedelta(
                    seconds=t_maneuver_baseline / 2)
                orbit_baseline = propagate_orbit(scenario.orbit, obs_baseline)
                _, _, baseline_end_s = geometric_lookahead_window(
                    orbit_baseline, pitch_baseline, cfg.fov_deg,
                    earliest_offset_s=(t_next - obs_baseline).total_seconds())
                max_support_end = max(
                    max_support_end,
                    obs_baseline + datetime.timedelta(seconds=baseline_end_s))
            baseline_end = max_support_end
            for scheduled in schedule:
                if scheduled.time > max_support_end:
                    baseline_end = scheduled.time
                    break
            n_sched_baseline = sum(
                1 for s in schedule if baseline_start < s.time < baseline_end)

        for pitch_deg in pitch_grid:
            t_maneuver = (agility(pitch_deg - boresight_pitch) - agility_at_zero) * 2
            if cfg.variant in optimal_timing_rules:
                if t_maneuver > gap:
                    continue
                obs_time = t_next - datetime.timedelta(seconds=t_maneuver / 2)
            else:
                obs_time = t_prev + datetime.timedelta(seconds=t_maneuver / 2)
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

            in_box_accesses = [acc_obs[j][-2] for j in in_box]
            in_box_unobs = [
                j for j in in_box
                if not acc_obs[j][-2].state.get("observed", False)
            ]
            if not in_box_unobs:
                continue

            n_clear_in_box = sum(1 for a in in_box_accesses
                                 if cloud_states.get(a.aid, False))
            unobs_accesses = [acc_obs[j][-2] for j in in_box_unobs]
            n_clear_unobs = sum(1 for a in unobs_accesses
                                if cloud_states.get(a.aid, False))

            if cfg.variant == "greedy":
                score_accesses = unobs_accesses
                vis_times = [(a.time - obs_time).total_seconds()
                             for a in score_accesses]
                min_obs = obs_time + datetime.timedelta(seconds=min(vis_times))
                max_obs = obs_time + datetime.timedelta(seconds=max(vis_times))
                t_window = max(max(vis_times) - min(vis_times), 1.0)
                support_window = t_window
                anchor_window = t_window
                left_anchor_gap = 0.0
                right_anchor_gap = 0.0
                avg_u = float(np.mean([a.utility for a in score_accesses]))
                n_sched = 0
                n_sched_window = 0
                value = score_lookahead(
                    "greedy",
                    n_visible=len(score_accesses),
                    t_window=1.0,
                    p_clear=p_clear,
                    n_sched=0,
                    avg_u=1.0,
                    missed_utility=0.0,
                    g_eff=g_eff,
                )
            elif cfg.variant in value_rules:
                score_accesses = [
                    a for a in unobs_accesses if a.time >= t_next
                ]
                if not score_accesses:
                    continue
                lead_s = (t_next - obs_time).total_seconds()
                t_window, geom_start_s, geom_end_s = geometric_lookahead_window(
                    orbit_obs, pitch_deg, cfg.fov_deg,
                    earliest_offset_s=lead_s)
                if t_window <= 0:
                    continue
                min_obs = obs_time + datetime.timedelta(seconds=geom_start_s)
                max_obs = obs_time + datetime.timedelta(seconds=geom_end_s)
                support_window = t_window
                anchor_window = max(
                    (baseline_end - baseline_start).total_seconds(), 1.0)
                left_anchor_gap = (min_obs - baseline_start).total_seconds()
                right_anchor_gap = (baseline_end - max_obs).total_seconds()
                avg_u = float(np.mean([a.utility for a in score_accesses]))
                n_sched_window = sum(
                    1 for s in schedule if min_obs <= s.time <= max_obs)
                n_sched = n_sched_baseline
                value = score_lookahead(
                    cfg.variant,
                    n_visible=len(score_accesses),
                    t_window=anchor_window,
                    p_clear=p_clear,
                    n_sched=n_sched,
                    avg_u=avg_u,
                    missed_utility=0.0,
                    g_eff=g_eff,
                    support_window=support_window,
                    left_anchor_gap=left_anchor_gap,
                    right_anchor_gap=right_anchor_gap,
                )
            else:
                value = -np.inf

            better_value = value > best_value
            lower_pitch_tie = (
                best_pitch is not None
                and np.isclose(value, best_value, rtol=1e-12, atol=1e-9)
                and pitch_deg < best_pitch
            )
            if better_value or lower_pitch_tie:
                best_value = value
                best_pitch = pitch_deg
                best_in_box = in_box
                best_acc_filtered = acc_obs
                best_event = {
                    "variant": cfg.variant,
                    "t_prev_iso": t_prev.isoformat(),
                    "t_next_iso": t_next.isoformat(),
                    "obs_time_iso": obs_time.isoformat(),
                    "gap_s": float(gap),
                    "best_pitch_deg": float(pitch_deg),
                    "boresight_pitch_deg": float(boresight_pitch),
                    "pitch_offset_deg": float(pitch_deg - boresight_pitch),
                    "score": float(value),
                    "t_maneuver_s": float(t_maneuver),
                    "n_in_box": int(len(in_box)),
                    "n_unobserved_in_box": int(len(in_box_unobs)),
                    "n_scored_in_box": int(len(score_accesses)),
                    "n_clear_in_box": int(n_clear_in_box),
                    "n_clear_unobserved_in_box": int(n_clear_unobs),
                    "n_scheduled_window": int(n_sched),
                    "n_scheduled_detector_window": int(n_sched_window),
                    "n_scheduled_baseline": int(n_sched_baseline),
                    "t_window_s": float(anchor_window),
                    "anchor_window_s": float(anchor_window),
                    "support_window_s": float(support_window),
                    "geometric_window_s": float(support_window),
                    "left_anchor_gap_s": float(left_anchor_gap),
                    "right_anchor_gap_s": float(right_anchor_gap),
                    "baseline_window_start_iso": baseline_start.isoformat(),
                    "baseline_window_end_iso": baseline_end.isoformat(),
                    "baseline_window_s": float(
                        (baseline_end - baseline_start).total_seconds()),
                    "avg_utility": float(avg_u),
                    "min_visible_offset_s": float(min(
                        (a.time - obs_time).total_seconds()
                        for a in unobs_accesses)),
                    "max_visible_offset_s": float(max(
                        (a.time - obs_time).total_seconds()
                        for a in unobs_accesses)),
                    "score_window_start_offset_s": float(
                        (min_obs - obs_time).total_seconds()),
                    "score_window_end_offset_s": float(
                        (max_obs - obs_time).total_seconds()),
                }

        gate_open = (cfg.variant in greedy_rules and best_pitch is not None) or (
            best_value > 0 and best_pitch is not None)
        if not gate_open:
            current_idx += 1
            continue
        if len(best_in_box) == 0:
            current_idx += 1
            continue

        n_lookaheads += 1
        t_maneuver_chosen = (agility(best_pitch - boresight_pitch) - agility_at_zero) * 2
        event_missed = 0
        event_missed_clear = 0

        if cfg.variant not in optimal_timing_rules:
            missed_indices = []
            for j in range(current_idx + 1, len(schedule)):
                if (schedule[j].time - t_prev).total_seconds() < t_maneuver_chosen:
                    missed_indices.append(j)
                else:
                    break
            n_missed_total += len(missed_indices)
            event_missed = len(missed_indices)
            event_missed_clear = sum(
                1 for j in missed_indices if cloud_states.get(schedule[j].aid, False))
            n_missed_clear_total += event_missed_clear
            for j in reversed(missed_indices):
                schedule.pop(j)

        n_lookahead_obs_total += len(best_in_box)
        for i in best_in_box:
            acc_obj = best_acc_filtered[i][-2]
            acc_obj.state["observed"] = True
            if cloud_states.get(acc_obj.aid, False):
                n_lookahead_clear_total += 1

        obs_times = [best_acc_filtered[i][-2].time for i in best_in_box]
        if not obs_times:
            current_idx += 1
            continue
        min_time, max_time = min(obs_times), max(obs_times)
        repair_min_time = (max(min_time, t_next)
                           if cfg.variant in optimal_timing_rules
                           else min_time)
        if best_event is not None:
            best_event.update({
                "lookahead_index": int(n_lookaheads),
                "t_maneuver_chosen_s": float(t_maneuver_chosen),
                "n_missed": int(event_missed),
                "n_missed_clear": int(event_missed_clear),
                "repair_min_time_iso": repair_min_time.isoformat(),
                "repair_max_time_iso": max_time.isoformat(),
            })
            lookahead_events.append(best_event)

        slice_idx = [i for i, a in enumerate(schedule)
                     if repair_min_time <= a.time <= max_time]
        if not slice_idx:
            current_idx += 1
            continue

        idx_start, idx_end = min(slice_idx), max(slice_idx)
        scheduled_outside = {a.requestid for a in
                             schedule[:idx_start] + schedule[idx_end + 1:]}
        slice_accesses = [
            a for a in accesses
            if repair_min_time <= a.time <= max_time
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

    return {
        "schedule": schedule,
        "n_lookaheads": n_lookaheads,
        "n_missed": n_missed_total,
        "n_missed_clear": n_missed_clear_total,
        "n_lookahead_obs": n_lookahead_obs_total,
        "n_lookahead_clear": n_lookahead_clear_total,
        "lookahead_events": lookahead_events,
    }


def _evaluate_variant(scen, cfg: SimConfig):
    """Run one variant on a prepared scenario; return the simulator result dict."""
    if cfg.variant == "never":
        return {
            "schedule": scen.schedule_conv,
            "n_lookaheads": 0,
            "n_missed": 0,
            "n_missed_clear": 0,
            "n_lookahead_obs": 0,
            "n_lookahead_clear": 0,
            "lookahead_events": [],
        }
    if cfg.variant == "omniscient":
        return {
            "schedule": scen.schedule_omni,
            "n_lookaheads": 0,
            "n_missed": 0,
            "n_missed_clear": 0,
            "n_lookahead_obs": 0,
            "n_lookahead_clear": 0,
            "lookahead_events": [],
        }
    reset_states(scen)
    return simulate_dt(scen, cfg)


def _trial_specs(cfg: SimConfig, rng: np.random.Generator):
    """Return (t0, raan, mean_anom) tuples for a run."""
    if cfg.trial_sampling in (
        "dates", "date_grid", "month_starts", "explicit_dates",
    ):
        dates = sample_trial_datetimes(cfg)
        raan_ref = np.deg2rad(cfg.raan_deg)
        mean_anom = np.deg2rad(cfg.mean_anom_deg)
        if cfg.match_ground_track and dates:
            era_ref = _era_rad(dates[0])
            return [
                (t0, float((raan_ref + _era_rad(t0) - era_ref) % (2 * np.pi)),
                 mean_anom)
                for t0 in dates
            ]
        return [(t0, raan_ref, mean_anom) for t0 in dates]

    raans = np.linspace(0, 2 * np.pi, cfg.n_trials, endpoint=False)
    return [(cfg.t0, float(raan), float(rng.uniform(0, 2 * np.pi)))
            for raan in raans]


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
    specs = _trial_specs(cfg_base, rng)

    cfgs = {v: replace(cfg_base, variant=v, tag=f"{cfg_base.tag}_{v}") for v in variants}
    results = {v: RunResult(cfg=cfgs[v].to_dict()) for v in variants}

    print(f"[{cfg_base.tag}] paired sweep: variants={variants}, "
          f"t_slew={cfg_base.slew_at_for_s}s, fov={cfg_base.fov_deg}°, "
          f"n_trials={cfg_base.n_trials}, horizon={cfg_base.horizon_h}h")

    for trial_idx, (t0, raan, mean_anom) in enumerate(specs):
        t_build = clock.time()
        scen = build_scenario(
            requests=requests, raan=raan, cfg=cfg_base, rng=rng,
            t0=t0, mean_anom=mean_anom)
        if scen is None:
            print(f"  trial {trial_idx+1}/{cfg_base.n_trials}: skipped (insufficient accesses)")
            continue
        build_s = clock.time() - t_build
        conv_clear = sum(a.utility for a in scen.schedule_conv
                         if scen.cloud_states[a.aid])
        omni_clear = sum(a.utility for a in scen.schedule_omni
                         if scen.cloud_states[a.aid])
        gap = omni_clear - conv_clear

        print(f"  trial {trial_idx+1}/{cfg_base.n_trials} "
              f"(t0={t0.isoformat()}, RAAN={np.rad2deg(raan):.0f}°), "
              f"conv={conv_clear:.0f}, omni={omni_clear:.0f}, build={build_s:.1f}s")
        for v in variants:
            t_var = clock.time()
            sim = _evaluate_variant(scen, cfgs[v])
            sched = sim["schedule"]
            dt_clear = sum(a.utility for a in sched
                           if scen.cloud_states.get(a.aid, False))
            elapsed = clock.time() - t_var
            results[v].trials.append(TrialResult(
                raan=float(raan),
                mean_anom=float(mean_anom),
                t0_iso=t0.isoformat(),
                n_lookaheads=int(sim["n_lookaheads"]),
                dt_clear=float(dt_clear),
                conv_clear=float(conv_clear),
                omni_clear=float(omni_clear),
                elapsed_s=float(elapsed),
                n_missed=int(sim["n_missed"]),
                n_missed_clear=int(sim["n_missed_clear"]),
                n_scheduled=len(sched),
                n_scheduled_conv=len(scen.schedule_conv),
                n_scheduled_omni=len(scen.schedule_omni),
                n_lookahead_obs=int(sim["n_lookahead_obs"]),
                n_lookahead_clear=int(sim["n_lookahead_clear"]),
                lookahead_events=sim.get("lookahead_events", []),
            ))
            gc = (dt_clear - conv_clear) / gap * 100 if gap > 0.5 else 0
            print(f"    {v:<11} dt={dt_clear:.0f}, looks={sim['n_lookaheads']}, "
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
    specs = _trial_specs(cfg, rng)
    result = RunResult(cfg=cfg.to_dict())

    print(f"[{cfg.tag}] variant={cfg.variant}, t_slew={cfg.slew_at_for_s}s, "
          f"fov={cfg.fov_deg}°, n_trials={cfg.n_trials}, horizon={cfg.horizon_h}h")

    for trial_idx, (t0, raan, mean_anom) in enumerate(specs):
        t_trial = clock.time()
        scen = build_scenario(
            requests=requests, raan=raan, cfg=cfg, rng=rng,
            t0=t0, mean_anom=mean_anom)
        if scen is None:
            print(f"  trial {trial_idx+1}/{cfg.n_trials}: skipped (insufficient accesses)")
            continue

        conv_clear = sum(a.utility for a in scen.schedule_conv
                         if scen.cloud_states[a.aid])
        omni_clear = sum(a.utility for a in scen.schedule_omni
                         if scen.cloud_states[a.aid])

        sim = _evaluate_variant(scen, cfg)
        schedule_final = sim["schedule"]

        dt_clear = sum(a.utility for a in schedule_final
                       if scen.cloud_states.get(a.aid, False))
        elapsed = clock.time() - t_trial

        result.trials.append(TrialResult(
            raan=float(raan),
            mean_anom=float(mean_anom),
            t0_iso=t0.isoformat(),
            n_lookaheads=int(sim["n_lookaheads"]),
            dt_clear=float(dt_clear),
            conv_clear=float(conv_clear),
            omni_clear=float(omni_clear),
            elapsed_s=float(elapsed),
            n_missed=int(sim["n_missed"]),
            n_missed_clear=int(sim["n_missed_clear"]),
            n_scheduled=len(schedule_final),
            n_scheduled_conv=len(scen.schedule_conv),
            n_scheduled_omni=len(scen.schedule_omni),
            n_lookahead_obs=int(sim["n_lookahead_obs"]),
            n_lookahead_clear=int(sim["n_lookahead_clear"]),
            lookahead_events=sim.get("lookahead_events", []),
        ))
        gap = omni_clear - conv_clear
        gc = (dt_clear - conv_clear) / gap * 100 if gap > 0.5 else 0
        print(f"  trial {trial_idx+1}/{cfg.n_trials} "
              f"(t0={t0.isoformat()}, RAAN={np.rad2deg(raan):.0f}°): "
              f"dt={dt_clear:.0f}, conv={conv_clear:.0f}, omni={omni_clear:.0f}, "
              f"looks={sim['n_lookaheads']}, gap={gc:.0f}% ({elapsed:.1f}s)")

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
