"""Keplerian orbits, frame conversions, horizon geometry."""
import datetime
from collections import namedtuple
from typing import Union

import numpy as np

from .constants import Constants
from .geometry import rotmat_z

Keplerian = namedtuple("Keplerian", ["a", "e", "i", "omega", "Omega", "M", "t"])


# ---------- Frame conversions ----------

def latlong2ecef(latlong) -> np.ndarray:
    lat_deg, lon_deg = latlong
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    return Constants.R_E * np.array([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ])


def latlong2ecef_vec(latlong: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(latlong[:, 0])
    lon = np.deg2rad(latlong[:, 1])
    cl, sl = np.cos(lat), np.sin(lat)
    co, so = np.cos(lon), np.sin(lon)
    return Constants.R_E * np.column_stack((cl * co, cl * so, sl))


def _era_rad(time: datetime.datetime) -> float:
    days = (time - Constants.J2000).total_seconds() / Constants.seconds_per_day
    return np.deg2rad((Constants.ERA_J2000 + Constants.gamma * days) % 360)


def ecef2eci(ecef: np.ndarray, time: datetime.datetime) -> np.ndarray:
    return rotmat_z(_era_rad(time)) @ ecef


def eci2ecef(eci: np.ndarray, time: datetime.datetime) -> np.ndarray:
    return rotmat_z(-_era_rad(time)) @ eci


def ecef2eci_vec(ecef: np.ndarray, time: datetime.datetime) -> np.ndarray:
    if ecef.ndim != 2 or ecef.shape[1] != 3:
        raise ValueError("ecef must have shape (N, 3)")
    R = rotmat_z(_era_rad(time))
    return ecef @ R.T


def ecef2latlong(ecef: np.ndarray) -> np.ndarray:
    X, Y, Z = ecef
    lat = np.arcsin(Z / np.linalg.norm(ecef))
    lon = np.arctan2(Y, X)
    return np.array([np.rad2deg(lat), np.rad2deg(lon)])


# ---------- Orbital mechanics ----------

def kepler2eci(elements: Keplerian):
    a, e, i, omega, Omega, M, _ = elements
    mu = Constants.mu

    if e != 0:
        E = M
        for _ in range(50):
            E_new = M + e * np.sin(E)
            if abs(E_new - E) < 1e-9:
                E = E_new
                break
            E = E_new
        nu = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
        r = a * (1 - e * np.cos(E))
    else:
        nu = M
        r = a

    r_perifocal = np.array([r * np.cos(nu), r * np.sin(nu), 0.0])
    h = np.sqrt(mu * a * (1 - e ** 2))
    v_perifocal = np.array([-mu / h * np.sin(nu), mu / h * (e + np.cos(nu)), 0.0])

    R_Omega = np.array([[np.cos(Omega), -np.sin(Omega), 0],
                        [np.sin(Omega),  np.cos(Omega), 0],
                        [0, 0, 1]])
    R_i = np.array([[1, 0, 0],
                    [0, np.cos(i), -np.sin(i)],
                    [0, np.sin(i),  np.cos(i)]])
    R_omega = np.array([[np.cos(omega), -np.sin(omega), 0],
                        [np.sin(omega),  np.cos(omega), 0],
                        [0, 0, 1]])
    R = R_Omega @ R_i @ R_omega
    return R @ r_perifocal, R @ v_perifocal


def circular_orbit(a: float, i: float, Omega: float, M: float,
                   t: datetime.datetime) -> Keplerian:
    return Keplerian(a, 0.0, i, 0.0, Omega, M, t)


def propagate_orbit(orbit: Keplerian,
                    time: Union[float, datetime.datetime, datetime.timedelta]) -> Keplerian:
    a, e, i, omega, Omega, M, t = orbit
    n = np.sqrt(Constants.mu / a ** 3)
    if isinstance(time, datetime.timedelta):
        dt = time.total_seconds()
    elif isinstance(time, datetime.datetime):
        dt = (time - t).total_seconds()
    else:
        dt = float(time)
    return Keplerian(a, e, i, omega, Omega, M + n * dt,
                     t + datetime.timedelta(seconds=dt))


# ---------- Geometry / horizon ----------

def v_orb(h: float) -> float:
    return np.sqrt(Constants.mu / (h + Constants.R_E))


def t_orb(elements: Keplerian) -> float:
    return 2 * np.pi * np.sqrt(elements.a ** 3 / Constants.mu)


def horizon_angle(elements: Keplerian) -> float:
    return np.arcsin(Constants.R_E / elements.a)


def horizon_distance(elements: Keplerian) -> float:
    return np.sqrt(elements.a ** 2 - Constants.R_E ** 2)


def horizon_spherical_angle(elements: Keplerian) -> float:
    return np.arccos(Constants.R_E / elements.a)


def horizon_time(elements: Keplerian) -> float:
    h = elements.a - Constants.R_E
    return horizon_spherical_angle(elements) * elements.a / v_orb(h)


# ---------- Sun (used for eclipse check in access) ----------

def sunvec_eci(time: datetime.datetime) -> np.ndarray:
    d = (time - Constants.J2000).total_seconds() / Constants.seconds_per_day
    L = 280.4606184 + (36000.77005361 / 36525) * d
    g = 357.5277233 + (35999.05034 / 36525) * d
    p = L + 1.914666471 * np.sin(np.deg2rad(g)) + 0.918994643 * np.sin(np.deg2rad(2 * g))
    q = 23.43929 - (46.8093 / 3600) * (d / 36525)
    p_r, q_r = np.deg2rad(p), np.deg2rad(q)
    return np.array([np.cos(p_r),
                     np.cos(q_r) * np.sin(p_r),
                     np.sin(q_r) * np.sin(p_r)])


# ---------- Convenience wrappers used by figure scripts ----------

def kepler2latlong(orbit: Keplerian, time: datetime.datetime) -> np.ndarray:
    """Sub-satellite lat/lon at `time` (uses orbit's epoch as reference)."""
    dt = (time - orbit.t).total_seconds()
    r_eci, _ = kepler2eci(propagate_orbit(orbit, dt))
    return ecef2latlong(eci2ecef(r_eci, time))


def latlong2eci(lat: float, lon: float, time: datetime.datetime) -> np.ndarray:
    return ecef2eci(latlong2ecef((lat, lon)), time)


def split_orbit_track(latlongs, threshold: float = 180):
    """Split a list of (lat, lon) along longitude wraps (for cartopy plotting)."""
    arr = np.asarray(latlongs)
    delta_long = np.abs(np.diff(arr[:, 1]))
    jump_indices = np.where(delta_long > threshold)[0] + 1
    return np.split(arr, jump_indices)


# ---------- Ray / Earth intersection ----------

def intersect_ray_sphere(P, u, x0, r, horizon_snap: bool = False):
    """Intersect the ray P + t*u with a sphere centered at x0 with radius r.

    Returns (pt1, pt2, t1, t2). When the ray misses the sphere, returns None
    unless `horizon_snap=True`, in which case returns the tangent point twice.
    """
    P = np.asarray(P, dtype=float)
    u = np.asarray(u, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    if np.allclose(u, 0):
        raise ValueError("Direction vector u must be non-zero")
    u = u / np.linalg.norm(u)

    d = P - x0
    A = 1.0
    B = 2.0 * np.dot(u, d)
    C = np.dot(d, d) - r * r
    disc = B * B - 4 * A * C

    if disc > 0:
        sqrt_disc = np.sqrt(disc)
        t1 = (-B + sqrt_disc) / (2 * A)
        t2 = (-B - sqrt_disc) / (2 * A)
        return P + t1 * u, P + t2 * u, t1, t2

    if np.isclose(disc, 0):
        t = -B / (2 * A)
        pt = P + t * u
        return pt, pt, t, t

    if not horizon_snap:
        return None

    L2 = np.dot(d, d)
    if L2 <= r * r:
        return None

    n = np.cross(u, d)
    if np.linalg.norm(n) < 1e-12:
        n = np.cross(d, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(n) < 1e-12:
            n = np.cross(d, np.array([0.0, 1.0, 0.0]))
    k = np.cross(n, d)
    k_hat = k / np.linalg.norm(k)

    L = np.sqrt(L2)
    delta = np.sqrt(L2 - r * r)

    T1 = x0 + (r * r / L2) * d + (r * delta / L) * k_hat
    T2 = x0 + (r * r / L2) * d - (r * delta / L) * k_hat

    dot1 = -np.dot(T1 - P, u)
    dot2 = -np.dot(T2 - P, u)

    if dot1 >= 0 and dot2 < 0:
        T, t = T1, -dot1
    elif dot2 >= 0 and dot1 < 0:
        T, t = T2, -dot2
    else:
        ang1 = np.arccos(dot1 / np.linalg.norm(T1 - P))
        ang2 = np.arccos(dot2 / np.linalg.norm(T2 - P))
        T, t = (T1, dot1) if ang1 < ang2 else (T2, dot2)

    return T, T, t, t


def earth_line_intersection(P, u, horizon_snap: bool = False):
    res = intersect_ray_sphere(P, u, np.zeros(3), Constants.R_E, horizon_snap)
    if res is None:
        return None
    p1, p2, t1, t2 = res
    if t1 < 0 and t2 < 0:
        return p1, p2
    return None
