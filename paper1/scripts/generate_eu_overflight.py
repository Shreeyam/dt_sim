"""Regenerate the Europe overflight figure under the current two-boundary
renewal heuristic. Saves to paper1/figures/lookahead_chain_eu.png.

Run from anywhere; uses the same orbit/setup as
experiments/full_pipeline_comparison.py and the renewal scoring from there
(two-boundary chain, g = t_slew(theta_FoR), N_sched anchor).
"""

import datetime
from pathlib import Path

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import scienceplots  # noqa: F401

from dt_sim.access import get_accesses
from dt_sim.constants import Constants
from dt_sim.orbits import (
    circular_orbit, propagate_orbit, kepler2eci, kepler2latlong,
    horizon_time, horizon_angle, latlong2ecef, eci2ecef,
    earth_line_intersection, ecef2latlong, split_orbit_track,
)
from dt_sim.cameras import (
    get_intrinsics_from_fov, project_in_box, filter_accesses_horizon,
    create_box, unproject_from_orbit,
)
from dt_sim.scheduling import milp_schedule, load_worldcities

STYLE_PATH = Path(__file__).resolve().parent.parent.parent / "shreeyam.mplstyle"
plt.style.use(["science", str(STYLE_PATH)])

# ---- Orbit / scenario (matches full_pipeline_comparison.py defaults) ----
H_KM = 400
INC_DEG = 51.6
# Exactly the params from `lookahead_discounting_claude_test copy.ipynb`,
# the actual generator of the original Europe overflight figure. Spacecraft
# starts at (51.6 N, -13.2 E) — Atlantic west of Ireland — at noon UTC and
# descends across Europe → Middle East over 17 minutes.
RAAN = np.pi - 0.05
MEAN_ANOM = np.pi / 2
T0 = datetime.datetime(2024, 1, 1, 12, 0, 0)
T_END = T0 + datetime.timedelta(minutes=17)
P_CLEAR = 0.34
FIELD_OF_REGARD = 30  # matches the notebook (paper experiments use 45)
FOV_DEG = 45
WIDTH = 800
HEIGHT = 600  # matches the notebook
BORESIGHT_PITCH = FOV_DEG / 2  # 22.5

# ---- Agility (same bang-bang as pipeline) ----
T_SETTLE = 10
BETA = 25 / np.sqrt(90)


def agility(theta):
    return T_SETTLE + BETA * np.sqrt(np.abs(theta))


# ---- Renewal heuristic (two-boundary form, same as simulate_dt) ----

def E_internal_two_boundary(lam, L, g):
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


def main():
    rng = np.random.default_rng(0)

    print("Loading cities + accesses...")
    requests = load_worldcities(n=10000)
    orbit = circular_orbit(
        a=Constants.R_E + H_KM,
        i=np.deg2rad(INC_DEG),
        Omega=RAAN,
        M=MEAN_ANOM,
        t=T0,
    )
    accesses = get_accesses(requests, orbit, t_coarse=500,
                             field_of_regard=FIELD_OF_REGARD,
                             t0=T0, t_end=T_END)
    print(f"  {len(accesses)} accesses over {(T_END - T0).total_seconds() / 60:.0f} min")

    cloud_states = {id(a): bool(rng.random() < P_CLEAR) for a in accesses}
    for a in accesses:
        a.state = {"observed": False, "cloudy": not cloud_states[id(a)]}

    K = get_intrinsics_from_fov(FOV_DEG, WIDTH, HEIGHT)
    box_hires = create_box(WIDTH, HEIGHT, points_per_edge=12)
    g_eff = agility(FIELD_OF_REGARD)

    schedule = milp_schedule(accesses, requests, agility)
    print(f"  initial schedule: {len(schedule)} tasks, "
          f"clear: {sum(1 for s in schedule if cloud_states[id(s)])}")

    # ---- Walk schedule, take renewal lookaheads, log boxes ----
    lookahead_boxes = []  # list of (pitch, time, [lat/lon corners])
    n_lookaheads = 0
    # current_idx = -1 acts as a virtual anchor at T0 so the heuristic can
    # propose a lookahead in the [T0, schedule[0]] dead zone.
    current_idx = -1

    while current_idx < len(schedule) - 1:
        # The schedule gap is [t_prev, t_next] = [schedule[i].time,
        # schedule[i+1].time]. The paper's optimal-timing proposition says the
        # lookahead maneuver should END at t_next, so we start it at
        # t* = t_next - t_man(alpha) (per-pitch). Each pitch picks its own t*
        # below; outside the loop we just need a reference time for the
        # filter_accesses_horizon call.
        if current_idx < 0:
            t_prev = T0
            t_next = schedule[0].time
        else:
            t_prev = schedule[current_idx].time
            t_next = schedule[current_idx + 1].time
        time = t_prev  # used by filter_accesses_horizon (1-hr lookahead window)
        orbit_now = propagate_orbit(orbit, time)
        r, v = kepler2eci(orbit_now)
        pos_ecef = eci2ecef(r, time)

        acc_converted = [(latlong2ecef([a.lat, a.long]), a.angle, a.time, a, idx)
                         for idx, a in enumerate(accesses)]
        acc_filt = filter_accesses_horizon(orbit_now, time, acc_converted,
                                            pos_ecef, FIELD_OF_REGARD)
        if not acc_filt:
            current_idx += 1
            continue

        # Iterate over candidate pitches; score with renewal value.
        # Match the notebook: 50 candidates from 0 to 53 deg.
        pitch_candidates = np.linspace(0, 53, 50)

        best_value = -np.inf
        best_pitch = None
        best_in_box = []
        best_acc_filt = []
        best_obs_time = None
        best_r_obs = None

        for pitch_deg in pitch_candidates:
            t_man = (agility(pitch_deg - BORESIGHT_PITCH) - agility(0)) * 2
            # Paper's optimal-timing proposition: maneuver ends at t_next, so
            # it STARTS at t_next - t_man and the OBSERVATION (instant, mid
            # maneuver) is at t_next - t_man / 2.
            t_decision = t_next - datetime.timedelta(seconds=t_man)
            obs_time = t_next - datetime.timedelta(seconds=t_man / 2)
            orbit_obs = propagate_orbit(orbit, obs_time)
            r_obs, _ = kepler2eci(orbit_obs)
            pos_ecef_obs = eci2ecef(r_obs, obs_time)

            acc_filt_obs = filter_accesses_horizon(
                orbit_obs, obs_time,
                [(latlong2ecef([a.lat, a.long]), a.angle, a.time, a, idx)
                 for idx, a in enumerate(accesses)],
                pos_ecef_obs, FIELD_OF_REGARD)
            if not acc_filt_obs:
                continue

            points_obs = np.array([rr for rr, _, _, _, _ in acc_filt_obs])
            _, in_box, _ = project_in_box(
                pitch_deg, 0, orbit_obs, obs_time,
                acc_filt_obs, points_obs, WIDTH, HEIGHT, K)
            if not len(in_box):
                continue

            # Filter to UNOBSERVED accesses only — already-observed accesses
            # carry no information for a new lookahead.
            in_box = [j for j in in_box
                      if not acc_filt_obs[j][-2].state.get("observed", False)]
            if not in_box:
                continue

            n_visible = len(in_box)
            vis_times = [(acc_filt_obs[i][-2].time - obs_time).total_seconds()
                         for i in in_box]
            t_window = max(vis_times) - min(vis_times) if len(vis_times) > 1 else 1.0

            min_obs = obs_time + datetime.timedelta(seconds=min(vis_times))
            max_obs = obs_time + datetime.timedelta(seconds=max(vis_times))
            n_sched = sum(1 for s in schedule if min_obs <= s.time <= max_obs)
            avg_u = np.mean([acc_filt_obs[i][-2].utility for i in in_box])

            pN = P_CLEAR * n_visible
            chain = (E_internal_two_boundary(pN / t_window, t_window, g_eff)
                     if pN > 0 and t_window > 0 else 0.0)
            advantage = (chain - P_CLEAR * n_sched) * avg_u

            # With optimal timing the maneuver ends at t_next; it can still
            # overflow the gap if t_man > (t_next - t_prev), in which case
            # the maneuver starts before the previous imaging finishes and
            # we'd have to forfeit some tasks. Skip infeasible pitches here
            # (they'd be feasibility-rejected in the actual policy).
            gap = (t_next - t_prev).total_seconds()
            if t_man > gap:
                continue
            opp_cost = 0.0  # optimal timing zeros opportunity cost (Prop)
            value = advantage - opp_cost

            if value > best_value:
                best_value = value
                best_pitch = pitch_deg
                best_in_box = in_box
                best_acc_filt = acc_filt_obs
                best_obs_time = obs_time
                best_r_obs = r_obs

        if best_pitch is None or best_value <= 0:
            current_idx += 1
            continue

        # Take the lookahead: log box, mark observed, splice schedule.
        # Draw the polygon at the OBSERVATION time/position (after slew),
        # which is what project_in_box actually used — not at the decision
        # time. The spacecraft moves enough during the maneuver that drawing
        # at the wrong time shifts the polygon away from the actual FoV.
        n_lookaheads += 1
        t_man_chosen = (agility(best_pitch - BORESIGHT_PITCH) - agility(0)) * 2
        orbit_at_obs = propagate_orbit(orbit, best_obs_time)
        ray_points = unproject_from_orbit(
            box_hires, -1, K, orbit_at_obs, best_obs_time,
            pitch_angle=best_pitch, roll_angle=0)
        unproj_dirs = best_r_obs - ray_points
        eci_intersections = [earth_line_intersection(best_r_obs, d, horizon_snap=True)[0]
                             for d in unproj_dirs]
        latlong_box = [ecef2latlong(eci2ecef(p, best_obs_time))
                       for p in eci_intersections]
        lookahead_boxes.append((best_pitch, best_obs_time, latlong_box))

        for j in best_in_box:
            best_acc_filt[j][-2].state["observed"] = True

        # With optimal timing, the maneuver ends at t_next so no scheduled
        # task is displaced (Proposition: opportunity cost = 0).

        obs_times = [best_acc_filt[j][-2].time for j in best_in_box]
        if not obs_times:
            current_idx += 1
            continue
        min_t, max_t = min(obs_times), max(obs_times)
        slice_idx = [i for i, a in enumerate(schedule)
                     if min_t <= a.time <= max_t]
        if not slice_idx:
            current_idx += 1
            continue
        i_s, i_e = min(slice_idx), max(slice_idx)

        scheduled_outside = set(a.requestid for a in
                                schedule[:i_s] + schedule[i_e + 1:])
        access_slice = [a for a in accesses
                        if min_t <= a.time <= max_t
                        and not (a.state.get("observed", False) and
                                 not cloud_states.get(id(a), True))
                        and a.requestid not in scheduled_outside]
        force_in = []
        if i_s > 0:
            access_slice.insert(0, schedule[i_s - 1])
            force_in.append(schedule[i_s - 1])
        if i_e < len(schedule) - 1:
            access_slice.append(schedule[i_e + 1])
            force_in.append(schedule[i_e + 1])
        try:
            new_slice = milp_schedule(access_slice, requests, agility,
                                       force_in if force_in else None)
            if force_in:
                new_slice = [a for a in new_slice if a not in force_in]
            schedule = schedule[:i_s] + new_slice + schedule[i_e + 1:]
        except Exception as e:
            print(f"  MILP repair failed: {e}")
        current_idx += 1

    print(f"  lookaheads taken: {n_lookaheads}")
    for pitch, t, box in lookahead_boxes:
        c_lat = np.mean([lat for lat, _ in box])
        c_lon = np.mean([lon for _, lon in box])
        print(f"    {t.isoformat()}  pitch={pitch:.1f}  centroid=({c_lat:.1f}, {c_lon:.1f})")

    # Pick the longest temporally-contiguous cluster (gap < 30 min between
    # consecutive lookaheads) — this gives the Europe → Asia → Indonesia
    # overflight that includes the suspected nadir-side magenta cases.
    lookahead_boxes.sort(key=lambda x: x[1])
    clusters = [[lookahead_boxes[0]]] if lookahead_boxes else []
    for la in lookahead_boxes[1:]:
        if (la[1] - clusters[-1][-1][1]).total_seconds() > 30 * 60:
            clusters.append([la])
        else:
            clusters[-1].append(la)
    # Single overflight: use the full T0..T_END window (matches notebook).
    t_win_start, t_win_end = T0, T_END
    print(f"  window: {t_win_start.isoformat()} → {t_win_end.isoformat()} "
          f"({len(lookahead_boxes)} lookaheads)")

    # ---- Plot ----
    fig = plt.figure(figsize=(7.0, 5.0), dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Orbit track in window
    horizon_dt = (t_win_end - T0).total_seconds()
    win_start_dt = (t_win_start - T0).total_seconds()
    track = [kepler2latlong(propagate_orbit(orbit, dt), T0 + datetime.timedelta(seconds=dt))
             for dt in np.linspace(win_start_dt, horizon_dt, 200)]
    for k, seg in enumerate(split_orbit_track(track)):
        ax.plot([lon for _, lon in seg], [lat for lat, _ in seg], 'k-', lw=1.0,
                transform=ccrs.PlateCarree(),
                label="Orbit Track" if k == 0 else None)

    # All requests (grey)
    ax.plot([r.long for r in requests], [r.lat for r in requests],
            '.', color='lightgrey', markersize=2, transform=ccrs.PlateCarree(),
            label="All Requests", zorder=1)

    # Accesses whose access-time falls in the overflight window
    win_acc = [a for a in accesses if t_win_start <= a.time <= t_win_end]

    # ---- Diagnostic: any window access inside a lookahead bbox AND within
    # ±60 s of that lookahead's obs time that is NOT marked observed?
    # Use the same camera projection as project_in_box: only flag accesses
    # that ACTUALLY project into the camera's pixel rectangle. Drawn polygon
    # uses horizon_snap and overstates the ground footprint near high-pitch
    # horizons, so don't trust it for marking checks.
    flagged = []
    for pitch, t_la, box in lookahead_boxes:
        t_man = (agility(pitch - BORESIGHT_PITCH) - agility(0)) * 2
        obs_time = t_la + datetime.timedelta(seconds=t_man / 2)
        candidates = [a for a in win_acc
                      if abs((a.time - obs_time).total_seconds()) <= 60
                      and not a.state.get("observed", False)]
        if not candidates:
            continue
        cand_pts = np.array([latlong2ecef([a.lat, a.long]) for a in candidates])
        orbit_obs = propagate_orbit(orbit, obs_time)
        _, in_box, _ = project_in_box(pitch, 0, orbit_obs, obs_time,
                                       [(p, 0, a.time, a, i) for i, (p, a) in
                                        enumerate(zip(cand_pts, candidates))],
                                       cand_pts, WIDTH, HEIGHT, K)
        for j in in_box:
            flagged.append((candidates[j], t_la, pitch))
    if flagged:
        print(f"  WARNING: {len(flagged)} accesses inside a lookahead bbox + ±60s "
              f"but UNOBSERVED — possible projection/marking bug.")
        for a, t_la, pitch in flagged[:5]:
            dt = (a.time - t_la).total_seconds()
            print(f"    access at ({a.lat:.1f}, {a.long:.1f}) t={a.time.strftime('%H:%M:%S')} "
                  f"vs lookahead t={t_la.strftime('%H:%M:%S')} (Δt={dt:+.0f}s) pitch={pitch:.0f}")
    else:
        print("  marking diagnostic: no unobserved accesses inside lookahead bboxes.")

    obs = [a for a in win_acc if a.state.get("observed")]
    unobs = [a for a in win_acc if not a.state.get("observed")]
    ax.plot([a.long for a in obs], [a.lat for a in obs],
            '.', color='#4a5de9', markersize=4, transform=ccrs.PlateCarree(),
            label="Observed Accesses", zorder=3)
    ax.plot([a.long for a in unobs], [a.lat for a in unobs],
            '.', color='#ea1a69', markersize=3, transform=ccrs.PlateCarree(),
            label="Unobserved Accesses", zorder=2)

    # Lookahead boxes (Europe-window only)
    for k, (pitch, t, box) in enumerate(lookahead_boxes):
        ax.plot([lon for _, lon in box], [lat for lat, _ in box],
                color='#d62728', lw=1.0, transform=ccrs.PlateCarree(),
                label="Lookahead" if k == 0 else None, zorder=4)

    # Start / end of the full notebook window (T0 .. T_END).
    s_lat, s_lon = kepler2latlong(propagate_orbit(orbit, T0), T0)
    e_lat, e_lon = kepler2latlong(propagate_orbit(orbit, T_END), T_END)
    ax.plot([s_lon], [s_lat], '*', color='green', markersize=7,
            markeredgecolor="black", markeredgewidth=0.4,
            transform=ccrs.PlateCarree(), label="Start", zorder=5)
    ax.plot([e_lon], [e_lat], '*', color='red', markersize=7,
            markeredgecolor="black", markeredgewidth=0.4,
            transform=ccrs.PlateCarree(), label="End", zorder=5)

    ax.stock_img(alpha=0.15)
    ax.coastlines(linewidth=0.6)
    if lookahead_boxes:
        all_lats = [lat for _, _, box in lookahead_boxes for lat, _ in box]
        all_lons = [lon for _, _, box in lookahead_boxes for _, lon in box]
        pad_lat, pad_lon = 8, 10
        ax.set_extent([min(all_lons) - pad_lon, max(all_lons) + pad_lon,
                       min(all_lats) - pad_lat, max(all_lats) + pad_lat],
                      crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.4)
    gl.top_labels = False
    gl.right_labels = False
    ax.legend(loc='lower left', frameon=True, ncol=2, fontsize=8)

    out_path = Path(__file__).parent.parent / "figures" / "lookahead_chain_overflight.pdf"
    plt.savefig(out_path, bbox_inches='tight')
    plt.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches='tight')
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
