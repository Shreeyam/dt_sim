"""Imaging requests and access generation.

`Access` is a single visibility opportunity for a `Request`. `get_accesses`
runs a recursive spatial bisection search to enumerate all accesses for a
request set against a given orbit over a horizon.
"""
import datetime
import itertools
from typing import List, Optional

import numpy as np

from .constants import Constants
from .geometry import dist, dist2plane
from .orbits import (ecef2eci, ecef2eci_vec, kepler2eci, latlong2ecef,
                     latlong2ecef_vec, propagate_orbit, sunvec_eci, v_orb)


class Request:
    __slots__ = ("id", "lat", "long", "name", "utility")

    def __init__(self, id: int, lat: float, long: float, name: str, utility: float = 1):
        self.id = id
        self.lat = lat
        self.long = long
        self.name = name
        self.utility = utility

    def __repr__(self):
        return f"Request({self.id}, {self.lat}, {self.long}, {self.name}, {self.utility})"


class Access:
    """A single visibility opportunity for a Request.

    `aid` is a process-stable monotonic id assigned at construction. Use it
    for dict keys (e.g., cloud-state lookup) instead of `id(a)` — survives
    pickling and deep copies, unlike CPython object ids.

    `ecef` is the request's ECEF position cached at construction. lat/long
    don't change over an access's lifetime, so we pay the trig once instead
    of every time the simulator re-projects the access.
    """
    __slots__ = ("aid", "request", "requestid", "lat", "long", "name", "time",
                 "angle", "state", "utility", "ecef")

    _counter = 0

    def __init__(self, request: Request, time: datetime.datetime, angle: float,
                 state: Optional[dict] = None, utility: Optional[float] = None):
        Access._counter += 1
        self.aid = Access._counter
        self.request = request
        self.requestid = request.id
        self.lat = request.lat
        self.long = request.long
        self.name = request.name
        self.time = time
        self.angle = angle
        self.state = state
        self.utility = request.utility if utility is None else utility
        self.ecef = latlong2ecef((request.lat, request.long))

    def __repr__(self):
        return (f"Access(aid={self.aid}, req={self.requestid}, t={self.time}, "
                f"angle={self.angle:.1f}, u={self.utility})")


def task_not_in_eclipse(time: datetime.datetime, r: np.ndarray) -> bool:
    return float(np.dot(r, sunvec_eci(time))) > 0


def _bisect_accesses(requests, req_latlongs, field_of_regard, t0_s, t0,
                     t1, t2, r1, r2, d1, d2, orbit) -> List[Access]:
    t3 = (t1 + t2) / 2
    if t2 - t1 < 1:
        out = []
        for x in requests:
            t_obs = t0 + datetime.timedelta(seconds=t3 - t0_s)
            task_eci = ecef2eci(latlong2ecef((x.lat, x.long)), t_obs)
            r, v = kepler2eci(propagate_orbit(orbit, t1))
            angle_diff = np.arccos(np.dot(r, r - task_eci) /
                                   (np.linalg.norm(r) * np.linalg.norm(task_eci - r)))
            sign = np.sign(dist2plane(r, np.cross(r, v), task_eci))
            angle_diff = np.rad2deg(angle_diff) * sign
            if abs(angle_diff) < field_of_regard and task_not_in_eclipse(t_obs, task_eci):
                out.append(Access(x, t_obs, angle_diff))
        return out

    r3, v3 = kepler2eci(propagate_orbit(orbit, t3))
    tasks_eci_3 = ecef2eci_vec(latlong2ecef_vec(req_latlongs),
                               t0 + datetime.timedelta(seconds=t3 - t0_s))
    d3 = dist2plane(r3, v3, tasks_eci_3)

    mask1 = d1 * d3 < 0
    mask2 = d2 * d3 < 0
    out = []
    if np.any(mask1):
        out += _bisect_accesses(
            [requests[i] for i in range(len(requests)) if mask1[i]],
            req_latlongs[mask1], field_of_regard, t0_s, t0, t1, t3,
            r1, r3, d1[mask1], d3[mask1], orbit)
    if np.any(mask2):
        out += _bisect_accesses(
            [requests[i] for i in range(len(requests)) if mask2[i]],
            req_latlongs[mask2], field_of_regard, t0_s, t0, t3, t2,
            r3, r2, d3[mask2], d2[mask2], orbit)
    return out


def get_accesses(requests, orbit, t_coarse: int, field_of_regard: float,
                 t0: datetime.datetime, t_end: datetime.datetime) -> List[Access]:
    """Enumerate all accesses for `requests` against `orbit` between `t0` and `t_end`.

    `t_coarse` controls the coarse search step (seconds). Smaller -> more accurate
    but slower; 500 has been the working default.
    """
    h = orbit.a - Constants.R_E
    v = v_orb(h)
    theta = np.arctan(h * np.tan(np.deg2rad(field_of_regard)) / Constants.R_E)
    theta_total = theta + np.deg2rad((Constants.gamma / 86400) * t_coarse)
    filter_radius = np.sqrt(
        (v * t_coarse / 2) ** 2
        + (h + Constants.R_E * (1 - np.cos(theta_total))) ** 2
        + (Constants.R_E * np.sin(theta_total)) ** 2
    )

    seconds_since_epoch = 0
    accesses: List[Access] = []
    req_latlongs = np.array([[r.lat, r.long] for r in requests])

    r1, v1 = kepler2eci(propagate_orbit(orbit, seconds_since_epoch))
    horizon_s = int((t_end - t0).total_seconds()) - t_coarse
    for i in range(0, horizon_s, t_coarse):
        t1 = seconds_since_epoch + i
        t2 = t1 + t_coarse

        r2, v2 = kepler2eci(propagate_orbit(orbit, t2))
        tasks_eci_1 = ecef2eci_vec(latlong2ecef_vec(req_latlongs),
                                    t0 + datetime.timedelta(seconds=i))
        tasks_eci_2 = ecef2eci_vec(latlong2ecef_vec(req_latlongs),
                                    t0 + datetime.timedelta(seconds=i + t_coarse))

        mask = (dist(r1, tasks_eci_1) <= filter_radius) | \
               (dist(r2, tasks_eci_2) <= filter_radius)
        if np.any(mask):
            reqs_pre = list(itertools.compress(requests, mask))
            ll_pre = list(itertools.compress(req_latlongs, mask))
            d1 = dist2plane(r1, v1, tasks_eci_1[mask])
            d2 = dist2plane(r2, v2, tasks_eci_2[mask])
            cross = [d1[k] * d2[k] < 0 and d1[k] > d2[k] for k in range(len(reqs_pre))]
            if any(cross):
                reqs_f = list(itertools.compress(reqs_pre, cross))
                ll_f = np.array(list(itertools.compress(ll_pre, cross)))
                accesses += _bisect_accesses(
                    reqs_f, ll_f, field_of_regard, seconds_since_epoch, t0,
                    t1, t2, r1, r2,
                    d1[cross], d2[cross], orbit)
        r1, v1 = r2, v2

    return accesses
