"""Regenerate lookahead system design figures for paper1 at arxiv widths.

Figures produced in paper1/figures/:
    - lookahead_time_constraints_agile.pdf
    - lookahead_time_constraints_ultra_agile.pdf
    - horizontal_fov_design_map.pdf
    - rolled_body_fov_design_map.pdf
    - camera_fov_limit_boxes.pdf

Max width 4 in. Uses shreeyam.mplstyle from the repo root.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
MPL_STYLE = REPO_ROOT / "shreeyam.mplstyle"
OUT_DIR = Path(__file__).resolve().parents[1] / "figures"

from dt_sim.constants import Constants

import scienceplots  # noqa: F401  # registers the "science" style
plt.style.use(["science", "grid", str(MPL_STYLE)])

LIMB_COLOR = "#ea1a69"
OPPOSITE_HORIZON_COLOR = "#00c5b3"


def subtending_angle_from_fov(fov: float, h: float) -> float:
    """Half-FOV to half subtending angle on Earth."""
    R_E = Constants.R_E
    sign = np.sign(fov)
    angle = np.abs(fov)
    beta = np.pi - np.arcsin((R_E + h) / R_E * np.sin(angle))
    theta = np.pi - (angle + beta)
    return theta * sign


def calc_lookahead_time(angle_rad: float, h: float) -> float:
    """Time (s) for a point pitched ``angle_rad`` ahead to reach nadir.

    ``mu`` is in km^3/s^2 and ``R_E``/``h`` are in km, so ``thetadot`` stays
    in consistent (km) units---the original notebook mixed km with meters and
    produced angular rates ~31 623x too small.
    """
    R_E = Constants.R_E
    thetadot = np.sqrt(Constants.mu / (R_E + h) ** 3)
    max_angle = np.arcsin(R_E / (R_E + h))
    if angle_rad > max_angle:
        angle_rad = max_angle - 1e-9
    return subtending_angle_from_fov(angle_rad, h) / thetadot


def plot_time_constraints(
    out_path: Path,
    *,
    field_of_regard_deg: float,
    altitudes_km: list[float],
) -> None:
    """Lookahead time curves at several altitudes (viridis), with symbolic constraint lines.

    Overlays a vertical line at the field-of-regard angle and a horizontal line at a
    representative slew time, both labeled symbolically.
    """
    angle = np.deg2rad(np.linspace(0, 90, 400))
    cmap = plt.get_cmap("viridis")
    colors = [cmap(x) for x in np.linspace(0.15, 1.0, len(altitudes_km))]

    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    y_top = 0.0
    for h, c in zip(altitudes_km, colors):
        times = np.array([calc_lookahead_time(x, h) for x in angle])
        max_angle_deg = np.rad2deg(np.arcsin(Constants.R_E / (Constants.R_E + h)))
        ax.plot(
            np.rad2deg(angle),
            times,
            color=c,
            linewidth=1.2,
            label=rf"${int(h)}$ km",
            zorder=2,
        )
        y_top = max(y_top, calc_lookahead_time(np.deg2rad(max_angle_deg), h))

    # Symbolic constraint lines.
    t_slew_sym = 30.0  # representative; labeled symbolically
    ax.axvline(
        field_of_regard_deg,
        linestyle="--",
        color="0.35",
        linewidth=0.8,
        zorder=1,
    )
    ax.axhline(t_slew_sym, linestyle="--", color="0.35", linewidth=0.8, zorder=1)

    ax.text(
        field_of_regard_deg + 1,
        y_top * 1.03,
        r"$\alpha_{\mathrm{FoR}}$",
        fontsize=9,
        color="0.2",
        va="top",
    )
    ax.text(
        79,
        t_slew_sym + 14,
        r"$t_{\mathrm{slew}}$",
        fontsize=9,
        color="0.2",
        ha="right",
    )

    ax.set_xlim(0, 80)
    ax.set_ylim(0, y_top * 1.1)
    ax.set_xlabel(r"Lookahead angle $\alpha$ [deg]")
    ax.set_ylabel(r"Lookahead time $t(\alpha)$ [s]")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        labels[::-1],
        title="Altitude $h$",
        fontsize=7,
        title_fontsize=7,
        loc="upper left",
    )
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path)
    plt.close(fig)


def ground_offset(q_rad: np.ndarray | float, h_km: float) -> np.ndarray | float:
    """Ground-plane offset from nadir for an off-nadir look angle."""
    R_E = Constants.R_E
    root_arg = R_E**2 - (R_E + h_km) ** 2 * np.sin(q_rad) ** 2
    root = np.sqrt(np.where(root_arg >= 0, root_arg, np.nan))
    return np.sin(q_rad) * (
        (R_E + h_km) * np.cos(q_rad)
        - root
    )


def horizontal_fov_requirement(
    vertical_fov_deg: np.ndarray,
    field_of_regard_deg: np.ndarray,
    *,
    h_km: float,
    boresight_pitch_deg: float | np.ndarray | None = None,
) -> np.ndarray:
    """Minimum full horizontal FoV for nadir-edge-aligned FoR coverage.

    The detector lower edge is aligned with nadir by default, so the boresight
    pitch is half the vertical FoV and the upper detector edge is at full
    off-nadir angle ``vertical_fov_deg``. The returned angle is the rectilinear
    horizontal detector requirement, not a viewing-sphere longitude.
    """
    R_E = Constants.R_E
    A = R_E + h_km
    theta_v, theta_for = np.broadcast_arrays(
        np.deg2rad(vertical_fov_deg),
        np.deg2rad(field_of_regard_deg),
    )
    if boresight_pitch_deg is None:
        beta = theta_v / 2
    else:
        beta = np.broadcast_to(np.deg2rad(boresight_pitch_deg), theta_v.shape)
    upper_angle = beta + theta_v / 2
    rho_for = ground_offset(theta_for, h_km)
    horizon_flat_sq = R_E**2 * (1 - (R_E / (R_E + h_km)) ** 2)
    remaining_horizon_sq = horizon_flat_sq - rho_for**2
    valid = np.isfinite(rho_for) & (remaining_horizon_sq >= 0)

    horizon_angle = np.arcsin(
        np.clip(np.sqrt(np.maximum(R_E**2 - rho_for**2, 0.0)) / A, 0.0, 1.0)
    )
    edge_root_arg = R_E**2 - rho_for**2 - A**2 * np.sin(upper_angle) ** 2
    edge_visible = valid & (upper_angle <= horizon_angle) & (edge_root_arg >= 0)
    x_edge = np.sin(upper_angle) * (
        A * np.cos(upper_angle) - np.sqrt(np.maximum(edge_root_arg, 0.0))
    )
    x_horizon = np.sqrt(np.maximum(remaining_horizon_sq, 0.0))
    x_star = np.where(edge_visible, x_edge, x_horizon)
    z_star = A - np.sqrt(np.maximum(R_E**2 - x_star**2 - rho_for**2, 0.0))
    depth = x_star * np.sin(beta) + z_star * np.cos(beta)
    requirement = 2 * np.rad2deg(np.arctan2(rho_for, depth))
    return np.where(valid & (depth > 0), requirement, np.nan)


def horizon_visible_vertical_fov(
    field_of_regard_deg: np.ndarray,
    *,
    h_km: float,
) -> np.ndarray:
    """Vertical FoV where the nadir-edge-aligned upper detector edge reaches the limb."""
    R_E = Constants.R_E
    A = R_E + h_km
    rho_for = ground_offset(np.deg2rad(field_of_regard_deg), h_km)
    horizon_flat_sq = R_E**2 * (1 - (R_E / A) ** 2)
    valid = np.isfinite(rho_for) & (rho_for**2 <= horizon_flat_sq)
    ratio = np.sqrt(np.maximum(R_E**2 - rho_for**2, 0.0)) / A
    horizon = np.rad2deg(np.arcsin(np.clip(ratio, 0.0, 1.0)))
    return np.where(valid, horizon, np.nan)


def rolled_horizon_visible_vertical_fov(
    field_of_regard_deg: np.ndarray,
    *,
    h_km: float,
    edge_sigmas: tuple[int, ...] = (-1, 1),
) -> np.ndarray:
    """Vertical FoV where the rolled detector first sees the limb on a FoR edge."""
    R_E = Constants.R_E
    A = R_E + h_km
    field = np.asarray(field_of_regard_deg, dtype=float)
    out = np.full(field.shape, np.nan, dtype=float)
    horizon_flat_sq = R_E**2 * (1 - (R_E / A) ** 2)
    beta_grid = np.deg2rad(np.linspace(0.05, 120.0, 2400) / 2)

    for index, field_value in np.ndenumerate(field):
        phi = np.deg2rad(field_value)
        rho_for = float(ground_offset(phi, h_km))
        remaining_horizon_sq = horizon_flat_sq - rho_for**2
        if not np.isfinite(rho_for) or remaining_horizon_sq < 0:
            continue

        x_horizon = np.sqrt(max(remaining_horizon_sq, 0.0))
        z_horizon = A - np.sqrt(max(R_E**2 - x_horizon**2 - rho_for**2, 0.0))
        visible = np.zeros_like(beta_grid, dtype=bool)
        for sigma in edge_sigmas:
            along = z_horizon * np.cos(phi) + sigma * rho_for * np.sin(phi)
            eta = x_horizon * np.cos(beta_grid) - along * np.sin(beta_grid)
            depth = x_horizon * np.sin(beta_grid) + along * np.cos(beta_grid)
            visible |= (depth > 1e-9) & (np.abs(eta) <= depth * np.tan(beta_grid))

        if not visible.any():
            continue

        first = int(np.argmax(visible))
        if first == 0:
            out[index] = np.rad2deg(2 * beta_grid[first])
            continue

        lo = beta_grid[first - 1]
        hi = beta_grid[first]
        for _ in range(28):
            mid = 0.5 * (lo + hi)
            is_visible = False
            for sigma in edge_sigmas:
                along = z_horizon * np.cos(phi) + sigma * rho_for * np.sin(phi)
                eta = x_horizon * np.cos(mid) - along * np.sin(mid)
                depth = x_horizon * np.sin(mid) + along * np.cos(mid)
                is_visible |= (depth > 1e-9) and (abs(eta) <= depth * np.tan(mid))
            if is_visible:
                hi = mid
            else:
                lo = mid
        out[index] = np.rad2deg(2 * hi)

    return out


def rolled_body_horizontal_requirement(
    vertical_fov_deg: np.ndarray,
    field_of_regard_deg: np.ndarray,
    *,
    h_km: float,
    samples: int = 5000,
) -> np.ndarray:
    """Rolled-body contact requirement in rectilinear camera coordinates.

    The roll angle is set equal to the field-of-regard angle. For each FoR
    edge, the requirement is the smallest horizontal angle reached by that edge
    inside the rectangular focal-plane detector band; the detector must be wide
    enough to touch both edges.
    """
    vertical = np.asarray(vertical_fov_deg, dtype=float)
    field = np.asarray(field_of_regard_deg, dtype=float)
    vertical_b, field_b = np.broadcast_arrays(vertical, field)
    out = np.full(vertical_b.shape, np.nan, dtype=float)

    flat_vertical = vertical_b.ravel()
    flat_field = field_b.ravel()
    for field_value in np.unique(flat_field[np.isfinite(flat_field)]):
        selector = flat_field == field_value
        out.ravel()[selector] = _rolled_body_horizontal_requirement_for_field(
            flat_vertical[selector],
            float(field_value),
            h_km=h_km,
            samples=samples,
        )
    return out


def _rolled_body_horizontal_requirement_for_field(
    vertical_fov_deg: np.ndarray,
    field_of_regard_deg: float,
    *,
    h_km: float,
    samples: int = 5000,
) -> np.ndarray:
    """Evaluate the rolled-body requirement for one field-of-regard value."""
    R_E = Constants.R_E
    vertical = np.asarray(vertical_fov_deg, dtype=float)
    beta = np.deg2rad(vertical) / 2
    tan_beta = np.tan(beta)
    phi = np.deg2rad(field_of_regard_deg)
    rho_for = float(ground_offset(phi, h_km))
    requirement = np.full(vertical.shape, np.nan, dtype=float)
    if not np.isfinite(rho_for):
        return requirement

    horizon_flat_sq = R_E**2 * (1 - (R_E / (R_E + h_km)) ** 2)
    remaining_horizon_sq = horizon_flat_sq - rho_for**2
    if remaining_horizon_sq < 0:
        return requirement

    x_horizon = np.sqrt(max(remaining_horizon_sq, 0.0))
    x = np.linspace(0.0, x_horizon, samples)[:, None]
    flat_sq = x**2 + rho_for**2
    z = h_km + R_E * (
        1 - np.sqrt(np.maximum(0.0, 1 - flat_sq / R_E**2))
    )

    cos_beta = np.cos(beta)[None, :]
    sin_beta = np.sin(beta)[None, :]
    tan_beta = tan_beta[None, :]
    max_edge_contact = np.zeros_like(beta, dtype=float)
    touches_edges = np.ones_like(beta, dtype=bool)

    for sigma in (-1, 1):
        delta = sigma * rho_for * np.cos(phi) - z * np.sin(phi)
        along = z * np.cos(phi) + sigma * rho_for * np.sin(phi)
        eta = x * cos_beta - along * sin_beta
        depth = x * sin_beta + along * cos_beta
        vertical_ok = (depth > 1e-9) & (np.abs(eta) <= (tan_beta + 1e-12) * depth)
        longitude = np.abs(np.arctan2(delta, depth))
        edge_contact = np.min(
            np.where(vertical_ok, longitude, np.inf),
            axis=0,
        )
        edge_touched = np.isfinite(edge_contact)
        touches_edges &= edge_touched
        max_edge_contact = np.maximum(
            max_edge_contact,
            np.where(edge_touched, edge_contact, np.inf),
        )

    finite = touches_edges & np.isfinite(max_edge_contact)
    requirement[finite] = 2 * np.rad2deg(max_edge_contact[finite])
    return requirement

def plot_horizontal_fov_design_map(out_path: Path, *, h_km: float = 400) -> None:
    """Heatmap linking required horizontal FoV to vertical FoV and FoR."""
    vertical = np.linspace(5, 120, 520)
    field_of_regard = np.linspace(4, 60, 285)
    V, F = np.meshgrid(vertical, field_of_regard)
    H = horizontal_fov_requirement(V, F, h_km=h_km)

    fig, ax = plt.subplots(figsize=(4.0, 3.05))
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad(color="0.86")
    mesh = ax.imshow(
        np.ma.masked_invalid(H),
        origin="lower",
        extent=(vertical.min(), vertical.max(), field_of_regard.min(), field_of_regard.max()),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=20,
        vmax=180,
    )
    infeasible = ~np.isfinite(H)
    if np.any(infeasible):
        ax.contourf(
            V,
            F,
            infeasible.astype(float),
            levels=[0.5, 1.5],
            colors=["0.86"],
            hatches=["///"],
        )
    levels = np.arange(20, 181, 20)
    contours = ax.contour(V, F, H, levels=levels, colors="white", linewidths=0.45, alpha=0.85)
    ax.clabel(contours, fmt=rf"$%d^\circ$", fontsize=6, inline=True)

    horizon_line = horizon_visible_vertical_fov(field_of_regard, h_km=h_km)
    ax.plot(horizon_line, field_of_regard, color=LIMB_COLOR, linestyle="-.", linewidth=1.05, zorder=5)
    label_field = 28.0
    label_vertical = np.interp(label_field, field_of_regard, horizon_line)
    ax.text(
        label_vertical + 2.0,
        label_field,
        "limb",
        fontsize=7,
        color=LIMB_COLOR,
        va="center",
        rotation=63,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.5},
        zorder=6,
    )

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label(r"Required $\Theta_h$ [deg]")

    ax.axhline(45, color="0.2", linestyle="--", linewidth=0.8)
    ax.plot(45, 45, "o", color="white", markeredgecolor="0.1", markersize=5, zorder=5)
    ax.plot(60, 45, "s", color="white", markeredgecolor="0.1", markersize=5, zorder=5)
    ax.annotate(
        r"$45^\circ\times45^\circ$",
        xy=(45, 45),
        xytext=(12, 47),
        fontsize=7,
        color="0.1",
        arrowprops={"arrowstyle": "-", "linewidth": 0.6, "color": "0.1"},
    )
    ax.annotate(
        r"$60^\circ\times60^\circ$",
        xy=(60, 45),
        xytext=(84, 55),
        fontsize=7,
        color="0.1",
        arrowprops={"arrowstyle": "-", "linewidth": 0.6, "color": "0.1"},
    )

    ax.set_xlim(vertical.min(), vertical.max())
    ax.set_ylim(field_of_regard.min(), field_of_regard.max())
    ax.set_xlabel(r"Vertical FoV $\Theta_v$ [deg]")
    ax.set_ylabel(r"Field of regard $\theta_{\mathrm{FoR}}$ [deg]")
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path)
    plt.close(fig)


def plot_rolled_body_fov_design_map(out_path: Path, *, h_km: float = 400) -> None:
    """Heatmap for the rolled-body horizontal containment requirement."""
    vertical = np.linspace(5, 120, 520)
    field_of_regard = np.linspace(4, 60, 285)
    V, F = np.meshgrid(vertical, field_of_regard)
    H = np.vstack(
        [
            _rolled_body_horizontal_requirement_for_field(
                vertical,
                float(field),
                h_km=h_km,
            )
            for field in field_of_regard
        ]
    )

    fig, ax = plt.subplots(figsize=(4.0, 3.05))
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad(color="0.86")
    mesh = ax.imshow(
        np.ma.masked_invalid(H),
        origin="lower",
        extent=(vertical.min(), vertical.max(), field_of_regard.min(), field_of_regard.max()),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=20,
        vmax=180,
    )
    infeasible = ~np.isfinite(H)
    if np.any(infeasible):
        ax.contourf(
            V,
            F,
            infeasible.astype(float),
            levels=[0.5, 1.5],
            colors=["0.86"],
            hatches=["///"],
            zorder=-2,
        )
    levels = np.arange(20, 181, 20)
    contours = ax.contour(V, F, H, levels=levels, colors="white", linewidths=0.45, alpha=0.85)
    ax.clabel(contours, fmt=rf"$%d^\circ$", fontsize=6, inline=True)

    horizon_line = rolled_horizon_visible_vertical_fov(field_of_regard, h_km=h_km)
    ax.plot(horizon_line, field_of_regard, color=LIMB_COLOR, linestyle="-.", linewidth=1.05, zorder=5)
    opposite_horizon_line = rolled_horizon_visible_vertical_fov(
        field_of_regard,
        h_km=h_km,
        edge_sigmas=(-1,),
    )
    ax.plot(
        opposite_horizon_line,
        field_of_regard,
        color=OPPOSITE_HORIZON_COLOR,
        linestyle="--",
        linewidth=1.05,
        zorder=5,
    )
    label_field = 54.0
    if np.isfinite(horizon_line).any():
        label_vertical = np.interp(label_field, field_of_regard, horizon_line)
        ax.text(
            label_vertical,
            label_field,
            "limb",
            fontsize=7,
            color=LIMB_COLOR,
            va="center",
            rotation=-80,
            zorder=6,
        )
    label_field = 55.0
    if np.isfinite(opposite_horizon_line).any():
        label_vertical = np.interp(label_field, field_of_regard, opposite_horizon_line)
        ax.text(
            label_vertical - 10.0,
            label_field,
            "opp. limb",
            fontsize=7,
            color=OPPOSITE_HORIZON_COLOR,
            va="center",
            rotation=58,
            zorder=6,
        )

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label(r"Required $\Theta_h$ [deg]")

    ax.axhline(45, color="0.2", linestyle="--", linewidth=0.8)
    ax.plot(45, 45, "o", color="white", markeredgecolor="0.1", markersize=5, zorder=5)
    ax.plot(60, 45, "s", color="white", markeredgecolor="0.1", markersize=5, zorder=5)
    ax.annotate(
        r"$45^\circ\times45^\circ$",
        xy=(45, 45),
        xytext=(12, 53),
        fontsize=7,
        color="0.1",
        arrowprops={"arrowstyle": "-", "linewidth": 0.6, "color": "0.1"},
    )
    ax.annotate(
        r"$60^\circ\times60^\circ$",
        xy=(60, 45),
        xytext=(28, 57),
        fontsize=7,
        color="0.1",
        arrowprops={"arrowstyle": "-", "linewidth": 0.6, "color": "0.1"},
    )

    ax.set_xlim(vertical.min(), vertical.max())
    ax.set_ylim(field_of_regard.min(), field_of_regard.max())
    ax.set_xlabel(r"Vertical FoV $\Theta_v$ [deg]")
    ax.set_ylabel(r"Field of regard $\theta_{\mathrm{FoR}}$ [deg]")
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path)
    plt.close(fig)


def plot_camera_fov_limit_boxes(out_dir: Path, *, h_km: float = 400) -> None:
    """Common off-axis camera projections for representative FoV requirements."""
    example_vertical = np.array([30.0, 45.0, 60.0])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    nadir_xlim = (-1.35, 1.35)
    nadir_ylim = (-0.10, 1.85)
    rolled_xlim = (-7.2, 7.2)
    rolled_ylim = (-1.25, 2.3)

    def surface_depth(flat_sq: np.ndarray) -> np.ndarray:
        root_arg = 1 - flat_sq / Constants.R_E**2
        return h_km + Constants.R_E * (
            1 - np.sqrt(np.where(root_arg >= 0, root_arg, np.nan))
        )

    def detector_box(
        theta_h_deg: float,
        theta_v_deg: float,
        *,
        horizontal_center_deg: float = 0.0,
    ) -> np.ndarray:
        left = np.tan(np.deg2rad(horizontal_center_deg - theta_h_deg / 2))
        right = np.tan(np.deg2rad(horizontal_center_deg + theta_h_deg / 2))
        half_v = np.tan(np.deg2rad(theta_v_deg / 2))
        return np.array(
            [
                [left, -half_v],
                [right, -half_v],
                [right, half_v],
                [left, half_v],
                [left, -half_v],
            ]
        )

    def off_axis_deg(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return u, v

    def project_nadir_body(
        x_ground: np.ndarray,
        y_ground: np.ndarray,
        *,
        beta: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        z = surface_depth(x_ground**2 + y_ground**2)
        depth = x_ground * np.sin(beta) + z * np.cos(beta)
        u = y_ground / depth
        v = (x_ground * np.cos(beta) - z * np.sin(beta)) / depth
        return u, v

    def project_rolled_body(
        x_ground: np.ndarray,
        y_ground: np.ndarray,
        *,
        beta: float,
        phi: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        z = surface_depth(x_ground**2 + y_ground**2)
        depth = x_ground * np.sin(beta) + np.cos(beta) * (
            z * np.cos(phi) + y_ground * np.sin(phi)
        )
        u = (y_ground * np.cos(phi) - z * np.sin(phi)) / depth
        v = (
            x_ground * np.cos(beta)
            - np.sin(beta) * (z * np.cos(phi) + y_ground * np.sin(phi))
        ) / depth
        return u, v

    def plot_masked_curve(
        ax,
        u: np.ndarray,
        v: np.ndarray,
        mask: np.ndarray,
        *,
        max_jump: float = 12.0,
        **kwargs,
    ) -> None:
        jump = np.hypot(np.diff(u), np.diff(v)) > max_jump
        connected = mask & np.r_[True, ~jump] & np.r_[~jump, True]
        starts = np.flatnonzero(connected & np.r_[True, ~connected[:-1]])
        stops = np.flatnonzero(connected & np.r_[~connected[1:], True]) + 1
        for start, stop in zip(starts, stops):
            if stop - start > 1:
                ax.plot(u[start:stop], v[start:stop], **kwargs)

    def horizon_envelope(
        u: np.ndarray,
        v: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        u_valid = u[mask]
        v_valid = v[mask]
        if u_valid.size < 2:
            return None, None

        bins = np.linspace(max(-7.2, np.nanmin(u_valid)), min(7.2, np.nanmax(u_valid)), 560)
        centers = 0.5 * (bins[:-1] + bins[1:])
        upper = np.full_like(centers, np.nan)
        for idx in range(len(centers)):
            in_bin = (u_valid >= bins[idx]) & (u_valid < bins[idx + 1])
            if np.any(in_bin):
                upper[idx] = np.nanmax(v_valid[in_bin])

        finite = np.isfinite(upper)
        if finite.sum() < 2:
            return None, None

        upper = np.interp(centers, centers[finite], upper[finite])
        return centers, upper

    def shade_above_horizon(
        ax,
        u: np.ndarray,
        v: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        centers, upper = horizon_envelope(u, v, mask)
        if centers is None:
            return None, None
        ax.fill_between(centers, upper, 2.3, color="0.92", linewidth=0, zorder=0)
        return centers, upper

    def horizon_terms(
        u: np.ndarray,
        v: np.ndarray,
        *,
        beta: float,
        phi: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Implicit pinhole image of the Earth limb."""
        horizon_cos = np.sqrt(1 - (Constants.R_E / (Constants.R_E + h_km)) ** 2)
        center_u = -np.sin(phi)
        center_v = -np.sin(beta) * np.cos(phi)
        center_depth = np.cos(beta) * np.cos(phi)
        center_dot = center_u * u + center_v * v + center_depth
        limb = center_dot**2 - horizon_cos**2 * (u**2 + v**2 + 1)
        return limb, center_dot

    def draw_horizon(
        ax,
        *,
        beta: float,
        phi: float,
        view_xlim: tuple[float, float],
        view_ylim: tuple[float, float],
        vertical_offset: float = 0.0,
    ) -> None:
        u_grid = np.linspace(*view_xlim, 1100)
        v_grid = np.linspace(*view_ylim, 520)
        u_mesh, v_mesh = np.meshgrid(u_grid, v_grid)
        limb, center_dot = horizon_terms(
            u_mesh,
            v_mesh - vertical_offset,
            beta=beta,
            phi=phi,
        )
        earth = (center_dot > 0) & (limb >= 0)
        ax.contourf(
            u_mesh,
            v_mesh,
            earth.astype(float),
            levels=[0.5, 1.5],
            colors=["0.92"],
            zorder=0,
        )
        visible_limb = np.where(center_dot > 0, limb, np.nan)
        ax.contour(
            u_mesh,
            v_mesh,
            visible_limb,
            levels=[0],
            colors=["0.15"],
            linewidths=0.9,
            zorder=2,
        )

    def draw_common_view(
        ax,
        *,
        theta_v_deg: float,
        theta_h_deg: float,
        field_of_regard_deg: float,
        rolled: bool,
        color: str,
        horizontal_center_deg: float = 0.0,
    ) -> None:
        beta = np.deg2rad(theta_v_deg / 2)
        phi = np.deg2rad(field_of_regard_deg if rolled else 0.0)
        rho_for = ground_offset(np.deg2rad(field_of_regard_deg), h_km)
        horizon_flat_sq = Constants.R_E**2 * (
            1 - (Constants.R_E / (Constants.R_E + h_km)) ** 2
        )
        x_for_horizon = np.sqrt(np.maximum(horizon_flat_sq - rho_for**2, 0))
        x_ground = np.linspace(-x_for_horizon, x_for_horizon, 900)
        view_xlim = rolled_xlim if rolled else nadir_xlim
        view_ylim = rolled_ylim if rolled else nadir_ylim
        vertical_offset = 0.0 if rolled else np.tan(beta)

        project = project_rolled_body if rolled else project_nadir_body
        draw_horizon(
            ax,
            beta=beta,
            phi=phi,
            view_xlim=view_xlim,
            view_ylim=view_ylim,
            vertical_offset=vertical_offset,
        )

        for sigma in (-1, 1):
            y_ground = np.full_like(x_ground, sigma * rho_for)
            if rolled:
                u, v = project(x_ground, y_ground, beta=beta, phi=phi)
            else:
                u, v = project(x_ground, y_ground, beta=beta)
            u, v = off_axis_deg(u, v)
            v = v + vertical_offset
            valid = np.isfinite(u) & np.isfinite(v)
            plot_masked_curve(
                ax,
                u,
                v,
                valid,
                color="k",
                linestyle="--",
                linewidth=0.85,
                zorder=4,
            )

        detector = detector_box(
            theta_h_deg,
            theta_v_deg,
            horizontal_center_deg=horizontal_center_deg,
        )
        detector[:, 1] += vertical_offset
        ax.plot(detector[:, 0], detector[:, 1], color=color, linewidth=1.2, zorder=3)
        left = np.tan(np.deg2rad(horizontal_center_deg - theta_h_deg / 2))
        right = np.tan(np.deg2rad(horizontal_center_deg + theta_h_deg / 2))
        center_u = np.tan(np.deg2rad(horizontal_center_deg))
        half_v = np.tan(np.deg2rad(theta_v_deg / 2))
        centerlines = [
            np.array([[center_u, -half_v + vertical_offset], [center_u, half_v + vertical_offset]]),
            np.array([[left, vertical_offset], [right, vertical_offset]]),
        ]
        for centerline in centerlines:
            ax.plot(centerline[:, 0], centerline[:, 1], color="0.72", linewidth=0.5, zorder=1)
        label_x = 0.96 if rolled else 0.04
        label_ha = "right" if rolled else "left"
        ax.text(
            label_x,
            0.94,
            rf"${theta_h_deg:.0f}^\circ\times{theta_v_deg:.0f}^\circ$",
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            ha=label_ha,
            color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.6},
        )
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(*view_xlim)
        ax.set_ylim(*view_ylim)

    nadir_for_deg = 45.0
    nadir_curve = horizontal_fov_requirement(
        example_vertical,
        np.full_like(example_vertical, nadir_for_deg),
        h_km=h_km,
    )

    rolled_for_deg = 40.0
    rolled_curve = rolled_body_horizontal_requirement(
        example_vertical,
        np.full_like(example_vertical, rolled_for_deg),
        h_km=h_km,
    )

    fig, axs = plt.subplots(2, 3, figsize=(8.0, 4.0))
    for col, theta_v in enumerate(example_vertical):
        color = colors[col % len(colors)]
        draw_common_view(
            axs[0, col],
            theta_v_deg=theta_v,
            theta_h_deg=nadir_curve[col],
            field_of_regard_deg=nadir_for_deg,
            rolled=False,
            color=color,
        )
        draw_common_view(
            axs[1, col],
            theta_v_deg=theta_v,
            theta_h_deg=rolled_curve[col],
            field_of_regard_deg=rolled_for_deg,
            rolled=True,
            color=color,
        )

    for ax, title in zip(axs[0, :], [r"$\Theta_v=30^\circ$", r"$\Theta_v=45^\circ$", r"$\Theta_v=60^\circ$"]):
        ax.set_title(title, fontsize=8)
    axs[0, 0].set_ylabel("Nadir FoR")
    axs[1, 0].set_ylabel("Rolled body")
    fig.tight_layout(pad=0.25, h_pad=0.25, w_pad=0.20)
    fig.savefig(out_dir / "camera_fov_limit_boxes.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_time_constraints(
        OUT_DIR / "lookahead_time_constraints.pdf",
        field_of_regard_deg=45,
        altitudes_km=[300, 400, 550, 700, 1000],
    )
    # optimal_timing.pdf is generated by generate_optimal_timing_real.py
    # using real MILP schedules rather than the synthetic stand-in below.
    plot_horizontal_fov_design_map(OUT_DIR / "horizontal_fov_design_map.pdf")
    plot_rolled_body_fov_design_map(OUT_DIR / "rolled_body_fov_design_map.pdf")
    plot_camera_fov_limit_boxes(OUT_DIR)

    print(f"Regenerated figures in {OUT_DIR}")


if __name__ == "__main__":
    main()
