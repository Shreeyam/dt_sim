"""South-America overflight case study with the bug-fixed renewal heuristic.

Differences from generate_eu_overflight.py:
- Orbit set to put the spacecraft on a daylit South-America descending pass
  (RAAN = 0, M = pi/2, 24-hour simulation horizon — the South-Atlantic
  overflight at ~19:24 UTC on 2025-01-01 emerges from cluster selection).
- Cluster selection picks the longest contiguous lookahead run (no Europe
  filter).
- Saves to paper1/figures/lookahead_chain_sa.pdf.

Carries the same fixes as the Europe script:
- Renewal scoring on UNOBSERVED accesses only (in_box_unobs).
- Optimal timing per Proposition (maneuver ends at t_next, opp_cost = 0,
  infeasible pitches skipped).
- Polygon drawn at obs_time / r_obs (after slew), not decision time.
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
    horizon_angle, latlong2ecef, eci2ecef,
    earth_line_intersection, ecef2latlong, split_orbit_track,
)
from dt_sim.cameras import (
    get_intrinsics_from_fov, project_in_box, filter_accesses_horizon,
    create_box, unproject_from_orbit,
)
from dt_sim.scheduling import milp_schedule, load_worldcities

STYLE_PATH = Path(__file__).resolve().parent.parent.parent / "shreeyam.mplstyle"
plt.style.use(["science", str(STYLE_PATH)])

# ---- Orbit / scenario ----
H_KM = 400
INC_DEG = 51.6
RAAN = 0.0
MEAN_ANOM = np.pi / 2
T0 = datetime.datetime(2025, 1, 1, 0, 0, 0)
T_END = T0 + datetime.timedelta(hours=24)
P_CLEAR = 0.34
FIELD_OF_REGARD = 45  # paper-1 default (the Europe script matched a 30-deg notebook)
FOV_DEG = 45
WIDTH = 800
HEIGHT = 600
BORESIGHT_PITCH = FOV_DEG / 2  # 22.5

# ---- Agility (paper-1 design point: t_slew(theta_FoR=45) ≈ 27.7 s) ----
T_SETTLE = 10
BETA = 25 / np.sqrt(90)


def agility(theta):
    return T_SETTLE + BETA * np.sqrt(np.abs(theta))


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
    print(f"  {len(accesses)} accesses over {(T_END - T0).total_seconds() / 3600:.0f} h")

    cloud_states = {id(a): bool(rng.random() < P_CLEAR) for a in accesses}
    for a in accesses:
        a.state = {"observed": False, "cloudy": not cloud_states[id(a)]}

    K = get_intrinsics_from_fov(FOV_DEG, WIDTH, HEIGHT)
    box_hires = create_box(WIDTH, HEIGHT, points_per_edge=12)
    g_eff = agility(FIELD_OF_REGARD)

    schedule = milp_schedule(accesses, requests, agility)
    print(f"  initial schedule: {len(schedule)} tasks, "
          f"clear: {sum(1 for s in schedule if cloud_states[id(s)])}")

    # Each entry: (pitch, obs_time, latlong_box, sweep_dict).
    # sweep_dict has 'pitches', 'value', 'advantage', 'chain', 'n_visible',
    # 'n_sched', 't_window', 'feasible' as parallel arrays over the sweep.
    lookahead_boxes = []
    n_lookaheads = 0
    # current_idx = -1 acts as a virtual anchor at T0 so the heuristic can
    # propose a lookahead in the [T0, schedule[0]] dead zone.
    current_idx = -1

    while current_idx < len(schedule) - 1:
        if current_idx < 0:
            t_prev = T0
            t_next = schedule[0].time
        else:
            t_prev = schedule[current_idx].time
            t_next = schedule[current_idx + 1].time
        time = t_prev
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

        # 50 candidate pitches (matches the notebook), with optimal timing.
        pitch_candidates = np.linspace(0, 53, 50)

        best_value = -np.inf
        best_pitch = None
        best_in_box = []
        best_acc_filt = []
        best_obs_time = None
        best_r_obs = None

        gap = (t_next - t_prev).total_seconds()
        # per-pitch diagnostic
        sweep = {k: np.full(len(pitch_candidates), np.nan)
                 for k in ("value", "advantage", "chain", "n_visible",
                           "n_sched", "t_window")}
        sweep["pitches"] = pitch_candidates.copy()
        sweep["feasible"] = np.zeros(len(pitch_candidates), dtype=bool)

        for pi, pitch_deg in enumerate(pitch_candidates):
            t_man = (agility(pitch_deg - BORESIGHT_PITCH) - agility(0)) * 2
            if t_man > gap:
                continue
            sweep["feasible"][pi] = True
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
            value = advantage  # opp_cost = 0 under optimal timing

            sweep["value"][pi] = value
            sweep["advantage"][pi] = advantage
            sweep["chain"][pi] = chain
            sweep["n_visible"][pi] = n_visible
            sweep["n_sched"][pi] = n_sched
            sweep["t_window"][pi] = t_window

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
                best_acc_filt = acc_filt_obs
                best_obs_time = obs_time
                best_r_obs = r_obs
                # capture diagnostic for the winning pitch
        if best_pitch is None or best_value <= 0:
            current_idx += 1
            continue

        n_lookaheads += 1
        orbit_at_obs = propagate_orbit(orbit, best_obs_time)
        ray_points = unproject_from_orbit(
            box_hires, -1, K, orbit_at_obs, best_obs_time,
            pitch_angle=best_pitch, roll_angle=0)
        unproj_dirs = best_r_obs - ray_points
        eci_intersections = [earth_line_intersection(best_r_obs, d, horizon_snap=True)[0]
                             for d in unproj_dirs]
        latlong_box = [ecef2latlong(eci2ecef(p, best_obs_time))
                       for p in eci_intersections]
        lookahead_boxes.append((best_pitch, best_obs_time, latlong_box, sweep))

        for j in best_in_box:
            best_acc_filt[j][-2].state["observed"] = True

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
    for pitch, t, box, _ in lookahead_boxes:
        c_lat = float(np.mean([lat for lat, _ in box]))
        c_lon = float(np.mean([lon for _, lon in box]))
        print(f"    {t.isoformat()}  pitch={pitch:.1f}  centroid=({c_lat:6.1f},{c_lon:7.1f})")

    # Cluster lookaheads by time (gap < 30 min = same overflight). Prefer
    # clusters whose centroid is over South America (lon -80..-30, lat -50..10);
    # fall back to longest cluster overall if none qualify.
    lookahead_boxes.sort(key=lambda x: x[1])
    if not lookahead_boxes:
        raise RuntimeError("No lookaheads triggered — nothing to plot.")
    clusters = [[lookahead_boxes[0]]]
    for la in lookahead_boxes[1:]:
        if (la[1] - clusters[-1][-1][1]).total_seconds() > 30 * 60:
            clusters.append([la])
        else:
            clusters[-1].append(la)

    def cluster_in_sa(cluster):
        c_lats = [np.mean([lat for lat, _ in box]) for _, _, box, _ in cluster]
        c_lons = [np.mean([lon for _, lon in box]) for _, _, box, _ in cluster]
        c_lat = float(np.mean(c_lats))
        c_lon = float(np.mean(c_lons))
        return -50 <= c_lat <= 10 and -80 <= c_lon <= -30

    sa_clusters = [c for c in clusters if cluster_in_sa(c)]
    if sa_clusters:
        sa_clusters.sort(key=len, reverse=True)
        best = sa_clusters[0]
        scope = "South America"
    else:
        clusters.sort(key=len, reverse=True)
        best = clusters[0]
        scope = "longest (no SA cluster)"
    # The plot time window must cover every access that any cluster lookahead
    # could have observed; the lookaheads see ahead by ~horizon_time(orbit)
    # ≈ 5 min, so accesses observed by the last lookahead can have access
    # times that long after its obs_time. Without this, observed dots near
    # the far edge of high-pitch boxes get filtered out of win_acc.
    from dt_sim.orbits import horizon_time
    pad = datetime.timedelta(minutes=2)
    t_win_start = best[0][1] - pad
    t_win_end = best[-1][1] + datetime.timedelta(seconds=horizon_time(orbit))
    lookahead_boxes = best
    print(f"  cluster ({scope}): {t_win_start.isoformat()} → "
          f"{t_win_end.isoformat()} ({len(best)} lookaheads)")

    # Window-restricted accesses for plotting.
    win_acc = [a for a in accesses if t_win_start <= a.time <= t_win_end]

    # Marking diagnostic via project_in_box logic.
    flagged = []
    for pitch, t_la, _box, _sweep in lookahead_boxes:
        t_man = (agility(pitch - BORESIGHT_PITCH) - agility(0)) * 2
        obs_time = t_la
        candidates = [a for a in win_acc
                      if abs((a.time - obs_time).total_seconds()) <= 60
                      and not a.state.get("observed", False)]
        if not candidates:
            continue
        cand_pts = np.array([latlong2ecef([a.lat, a.long]) for a in candidates])
        orbit_obs = propagate_orbit(orbit, obs_time)
        _, in_box, _ = project_in_box(
            pitch, 0, orbit_obs, obs_time,
            [(p, 0, a.time, a, i) for i, (p, a) in
             enumerate(zip(cand_pts, candidates))],
            cand_pts, WIDTH, HEIGHT, K)
        for j in in_box:
            flagged.append(candidates[j])
    if flagged:
        print(f"  WARNING: {len(flagged)} unobserved accesses inside lookahead "
              f"FoV — possible bug")
    else:
        print("  marking diagnostic: clean")

    # ---- Plot ----
    fig = plt.figure(figsize=(7.0, 5.5), dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree())

    horizon_dt = (t_win_end - T0).total_seconds()
    win_start_dt = (t_win_start - T0).total_seconds()
    track = [kepler2latlong(propagate_orbit(orbit, dt),
                             T0 + datetime.timedelta(seconds=dt))
             for dt in np.linspace(win_start_dt, horizon_dt, 200)]
    for k, seg in enumerate(split_orbit_track(track)):
        ax.plot([lon for _, lon in seg], [lat for lat, _ in seg], 'k-', lw=1.0,
                transform=ccrs.PlateCarree(),
                label="Orbit Track" if k == 0 else None)

    ax.plot([r.long for r in requests], [r.lat for r in requests],
            '.', color='lightgrey', markersize=2, transform=ccrs.PlateCarree(),
            label="All Requests", zorder=1)

    obs = [a for a in win_acc if a.state.get("observed")]
    unobs = [a for a in win_acc if not a.state.get("observed")]
    ax.plot([a.long for a in obs], [a.lat for a in obs],
            '.', color='#4a5de9', markersize=4, transform=ccrs.PlateCarree(),
            label="Observed Accesses", zorder=3)
    ax.plot([a.long for a in unobs], [a.lat for a in unobs],
            '.', color='#ea1a69', markersize=3, transform=ccrs.PlateCarree(),
            label="Unobserved Accesses", zorder=2)

    for k, (pitch, t, box, _sweep) in enumerate(lookahead_boxes):
        ax.plot([lon for _, lon in box], [lat for lat, _ in box],
                color='#d62728', lw=1.0, transform=ccrs.PlateCarree(),
                label="Lookahead" if k == 0 else None, zorder=4)

    s_lat, s_lon = kepler2latlong(propagate_orbit(orbit, t_win_start), t_win_start)
    e_lat, e_lon = kepler2latlong(propagate_orbit(orbit, t_win_end), t_win_end)
    ax.plot([s_lon], [s_lat], '*', color='green', markersize=7,
            markeredgecolor="black", markeredgewidth=0.4,
            transform=ccrs.PlateCarree(), label="Start", zorder=5)
    ax.plot([e_lon], [e_lat], '*', color='red', markersize=7,
            markeredgecolor="black", markeredgewidth=0.4,
            transform=ccrs.PlateCarree(), label="End", zorder=5)

    ax.stock_img(alpha=0.15)
    ax.coastlines(linewidth=0.6)
    all_lats = [lat for _, _, box, _ in lookahead_boxes for lat, _ in box]
    all_lons = [lon for _, _, box, _ in lookahead_boxes for _, lon in box]
    pad_lat, pad_lon = 8, 10
    ax.set_extent([min(all_lons) - pad_lon, max(all_lons) + pad_lon,
                   min(all_lats) - pad_lat, max(all_lats) + pad_lat],
                  crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.4)
    gl.top_labels = False
    gl.right_labels = False
    ax.legend(loc='lower left', frameon=True, ncol=2, fontsize=8)

    out_path = Path(__file__).parent.parent / "figures" / "lookahead_chain_sa.pdf"
    plt.savefig(out_path, bbox_inches='tight')
    plt.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches='tight')
    print(f"Saved {out_path}")

    # ---- Re-run the projection at the last lookahead to inspect WHICH
    #      accesses end up in_box, with their lat/lon and access times. ----
    last_pitch, last_t, last_box, last_sweep = lookahead_boxes[-1]
    last_orbit_obs = propagate_orbit(orbit, last_t)
    last_r_obs, _ = kepler2eci(last_orbit_obs)
    last_pos_ecef = eci2ecef(last_r_obs, last_t)
    sat_lat, sat_lon = kepler2latlong(last_orbit_obs, last_t)
    print(f"\nLast lookahead: t={last_t.strftime('%H:%M:%S')} pitch={last_pitch:.1f} "
          f"sat=({sat_lat:.1f},{sat_lon:.1f})")

    last_acc_filt = filter_accesses_horizon(
        last_orbit_obs, last_t,
        [(latlong2ecef([a.lat, a.long]), a.angle, a.time, a, idx)
         for idx, a in enumerate(accesses)],
        last_pos_ecef, FIELD_OF_REGARD)
    last_pts = np.array([rr for rr, _, _, _, _ in last_acc_filt])
    _, last_in_box, _ = project_in_box(
        last_pitch, 0, last_orbit_obs, last_t,
        last_acc_filt, last_pts, WIDTH, HEIGHT, K)

    in_box_accs = [last_acc_filt[j][-2] for j in last_in_box]
    print(f"  {len(in_box_accs)} accesses in box")
    print(f"  poly bbox: lat [{min(lat for lat,_ in last_box):.1f}, "
          f"{max(lat for lat,_ in last_box):.1f}], "
          f"lon [{min(lon for _,lon in last_box):.1f}, "
          f"{max(lon for _,lon in last_box):.1f}]")
    print(f"  access lat range: [{min(a.lat for a in in_box_accs):.1f}, "
          f"{max(a.lat for a in in_box_accs):.1f}]")
    print(f"  access lon range: [{min(a.long for a in in_box_accs):.1f}, "
          f"{max(a.long for a in in_box_accs):.1f}]")
    print(f"  access time range: [{min(a.time for a in in_box_accs).strftime('%H:%M:%S')}, "
          f"{max(a.time for a in in_box_accs).strftime('%H:%M:%S')}]")
    # First 5 by time
    for a in sorted(in_box_accs, key=lambda x: x.time)[:5]:
        in_win = t_win_start <= a.time <= t_win_end
        print(f"    ({a.lat:6.1f},{a.long:7.1f}) at {a.time.strftime('%H:%M:%S')} "
              f"obs={a.state.get('observed',False)} in_win={in_win}")

    pitches = last_sweep["pitches"]
    feas = last_sweep["feasible"]
    fig2, axes = plt.subplots(2, 1, figsize=(5.5, 5.0), sharex=True,
                               gridspec_kw={"hspace": 0.1})
    ax_v, ax_n = axes
    ax_v.plot(pitches[feas], last_sweep["value"][feas], "-",
              color="#4a5de9", lw=1.2, label="value (advantage)")
    ax_v.plot(pitches[feas], last_sweep["chain"][feas], ":",
              color="#4a5de9", lw=1.0, alpha=0.7, label=r"$\bar c \cdot E[K]$ (chain term)")
    ax_v.plot(pitches[feas], -P_CLEAR * last_sweep["n_sched"][feas]
              * np.where(last_sweep["n_visible"][feas] > 0, 1, np.nan), ":",
              color="#ea1a69", lw=1.0, alpha=0.7,
              label=r"$-p \cdot N_{\mathrm{sched}}$ (anchor cost)")
    ax_v.axvline(last_pitch, color="black", ls="--", lw=0.7,
                 label=fr"chosen pitch = ${last_pitch:.0f}^\circ$")
    ax_v.axhline(0, color="gray", lw=0.5)
    ax_v.set_ylabel("heuristic value")
    ax_v.legend(loc="upper left", frameon=False, fontsize=8)
    ax_v.set_title(f"last lookahead in cluster, t = {last_t.strftime('%H:%M:%S')}",
                   fontsize=9)

    ax_n.plot(pitches[feas], last_sweep["n_visible"][feas], "-",
              color="#4a5de9", lw=1.2, label=r"$N_{\mathrm{obs}}$ (unobserved)")
    ax_n.plot(pitches[feas], last_sweep["n_sched"][feas], "-",
              color="#ea1a69", lw=1.2, label=r"$N_{\mathrm{sched}}$ (in window)")
    ax_n.plot(pitches[feas], last_sweep["t_window"][feas] / 10, "--",
              color="gray", lw=1.0, label=r"$\Delta t_{\mathrm{obs}}$ / 10 [s]")
    ax_n.axvline(last_pitch, color="black", ls="--", lw=0.7)
    ax_n.set_xlabel(r"pitch [$^\circ$]")
    ax_n.set_ylabel("count / scaled time")
    ax_n.legend(loc="upper left", frameon=False, fontsize=8)

    diag_path = Path(__file__).parent.parent / "figures" / "sa_last_lookahead_sweep.pdf"
    plt.savefig(diag_path, bbox_inches='tight')
    plt.savefig(diag_path.with_suffix(".png"), dpi=200, bbox_inches='tight')
    print(f"Saved {diag_path}")

    # Print the per-pitch table for the last lookahead
    print(f"\nLast-lookahead pitch sweep (t = {last_t.isoformat()}):")
    print(f"  pitch  N_obs  N_sched  t_win[s]   chain   advantage   value")
    for i in range(len(pitches)):
        if not feas[i]:
            continue
        if np.isnan(last_sweep["value"][i]):
            continue
        print(f"  {pitches[i]:5.1f}  "
              f"{last_sweep['n_visible'][i]:5.0f}  "
              f"{last_sweep['n_sched'][i]:7.0f}  "
              f"{last_sweep['t_window'][i]:8.1f}  "
              f"{last_sweep['chain'][i]:6.3f}  "
              f"{last_sweep['advantage'][i]:9.3f}  "
              f"{last_sweep['value'][i]:7.3f}"
              + ("  <- CHOSEN" if abs(pitches[i] - last_pitch) < 1e-6 else ""))


if __name__ == "__main__":
    main()
