"""Footprint comparison between a narrow primary instrument (~1 deg FoV) and a
wider, forward-pitched lookahead instrument.

Adapted from ``phd_code/notebooks/plots/plot_camera_animation_final.ipynb``
(cell 3). Renders both ground footprints on a single Cartopy axis using the
magenta and teal entries from ``shreeyam.mplstyle``. Axis decorations are
suppressed so the figure can drop straight into a paper figure.
"""
from __future__ import annotations

import datetime
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  # registers the "science" style

from dt_sim.cameras import create_box, get_intrinsics_from_fov, unproject_from_orbit
from dt_sim.constants import Constants
from dt_sim.orbits import (circular_orbit, earth_line_intersection,
                           ecef2latlong, eci2ecef, kepler2eci, kepler2latlong)

REPO_ROOT = Path(__file__).resolve().parents[2]
MPL_STYLE = REPO_ROOT / "shreeyam.mplstyle"
OUT_DIR = Path(__file__).resolve().parents[1] / "figures"

plt.style.use(["science", "grid", str(MPL_STYLE)])

MAGENTA = "#ea1a69"
TEAL = "#00c5b3"

# Instruments
PRIMARY_FOV_DEG = 1.0
PRIMARY_PITCH_DEG = 0.0
LOOKAHEAD_FOV_DEG = 45.0
LOOKAHEAD_PITCH_DEG = 22.5

# Imaging geometry
ALTITUDE_KM = 400
WIDTH = 800
HEIGHT = 800
POINTS_PER_EDGE = 200


def footprint_latlon(orbit, r, fov_deg: float, pitch_deg: float, roll_deg: float = 0.0):
    """Return (lons, lats) of the camera ground footprint for a given FoV/pitch."""
    K = get_intrinsics_from_fov(fov_deg, WIDTH, HEIGHT)
    box = create_box(WIDTH, HEIGHT, POINTS_PER_EDGE)
    ray_points = unproject_from_orbit(
        box, -1, K, orbit, orbit.t,
        pitch_angle=pitch_deg, roll_angle=roll_deg)
    dirs = r - ray_points
    pts_eci = [earth_line_intersection(r, d, horizon_snap=True)[0] for d in dirs]
    latlons = [ecef2latlong(eci2ecef(pt, orbit.t)) for pt in pts_eci]
    lats, lons = zip(*latlons)
    return np.asarray(lons), np.asarray(lats)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    orbit = circular_orbit(
        a=Constants.R_E + ALTITUDE_KM,
        i=np.deg2rad(0),
        Omega=np.pi,
        M=2 * np.pi - 1.4,
        t=datetime.datetime(2025, 6, 6, 16, 0, 0),
    )
    r, _ = kepler2eci(orbit)
    sub_lat, sub_lon = kepler2latlong(orbit, orbit.t)

    lon_p, lat_p = footprint_latlon(orbit, r, PRIMARY_FOV_DEG, PRIMARY_PITCH_DEG)
    lon_l, lat_l = footprint_latlon(orbit, r, LOOKAHEAD_FOV_DEG, LOOKAHEAD_PITCH_DEG)

    # Set extent around both footprints with a small pad
    all_lon = np.concatenate([lon_p, lon_l])
    all_lat = np.concatenate([lat_p, lat_l])
    pad = 0.4
    extent = [all_lon.min() - pad, all_lon.max() + pad,
              all_lat.min() - pad, all_lat.max() + pad]

    fig = plt.figure(figsize=(2.0, 2.0), dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.stock_img(alpha=0.15)
    ax.coastlines(linewidth=0.6)
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.fill(lon_l, lat_l, color=TEAL, alpha=0.35, transform=ccrs.PlateCarree(),
            zorder=2, label="Lookahead")
    ax.plot(lon_l, lat_l, color=TEAL, linewidth=1.6, transform=ccrs.PlateCarree(),
            zorder=3)

    ax.fill(lon_p, lat_p, color=MAGENTA, alpha=0.6, transform=ccrs.PlateCarree(),
            zorder=4, label="Primary")
    ax.plot(lon_p, lat_p, color=MAGENTA, linewidth=1.6, transform=ccrs.PlateCarree(),
            zorder=5)

    # Gridlines without ticks/labels
    gl = ax.gridlines(draw_labels=False, linestyle="--", color="gray",
                      alpha=0.5, linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.tight_layout(pad=0.1)
    out_pdf = OUT_DIR / "footprint_primary_vs_lookahead.pdf"
    out_png = OUT_DIR / "footprint_primary_vs_lookahead.png"
    out_svg = OUT_DIR / "footprint_primary_vs_lookahead.svg"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.02)
    print(f"Sub-satellite point: lat={sub_lat:.3f}, lon={sub_lon:.3f}")
    print(f"Primary footprint extent: lon=[{lon_p.min():.3f}, {lon_p.max():.3f}], "
          f"lat=[{lat_p.min():.3f}, {lat_p.max():.3f}]")
    print(f"Lookahead footprint extent: lon=[{lon_l.min():.3f}, {lon_l.max():.3f}], "
          f"lat=[{lat_l.min():.3f}, {lat_l.max():.3f}]")
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")
    print(f"Saved {out_svg}")


if __name__ == "__main__":
    main()
