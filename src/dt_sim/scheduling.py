"""MILP scheduler and request loaders.

Provides uniformly sampled request generation, world-city request loading, and
single-satellite MILP scheduling under slew-time and one-per-request constraints.
"""
import datetime
import re
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from pyscipopt import Model, quicksum

from .access import Access, Request

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def generate_requests(n: int, rng: Optional[np.random.Generator] = None) -> List[Request]:
    """Generate `n` Requests sampled uniformly over the sphere.

    Uses the cylindrical equal-area sampling (z = sin(lat) ~ U[-1, 1])
    that gives unbiased lat/long coverage. Pass an `rng` for reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng()
    z = rng.uniform(-1.0, 1.0, size=n)
    theta = rng.uniform(0.0, 2 * np.pi, size=n)
    lat = np.degrees(np.arcsin(z))
    lon = np.degrees(theta)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    return [Request(i, float(lat[i]), float(lon[i]), f"request_{i}") for i in range(n)]


def load_worldcities(n: int = 1000, csv_path: Optional[Path] = None) -> List[Request]:
    """Load the top-n cities from `data/worldcities.csv` as Request objects."""
    path = Path(csv_path) if csv_path else DATA_DIR / "worldcities.csv"
    out: List[Request] = []
    with open(path, "r", encoding="utf8") as f:
        f.readline()  # header
        for line in f:
            parts = re.split(r'(?:",")|"', line)
            lat, lon, city = float(parts[3]), float(parts[4]), parts[2]
            out.append(Request(len(out), lat, lon, city))
            if len(out) >= n:
                break
    return out


def _build_conflict_pairs(accesses, agility):
    """Return (i, j) pairs that must be mutually exclusive: same request, or
    insufficient slew time between them. Vectorised over j per i."""
    n = len(accesses)
    if n < 2:
        return []
    angles = np.array([a.angle for a in accesses], dtype=float)
    times = np.array([a.time.timestamp() for a in accesses], dtype=float)
    req_ids = np.array([a.requestid for a in accesses], dtype=np.int64)

    pairs = []
    for i in range(n - 1):
        dtheta = angles[i + 1:] - angles[i]
        slew = agility(dtheta)
        if np.isscalar(slew):
            slew = np.full_like(dtheta, slew)
        time_gap = times[i + 1:] - times[i]
        conflict = (time_gap < slew) | (req_ids[i + 1:] == req_ids[i])
        for k in np.flatnonzero(conflict):
            pairs.append((i, i + 1 + int(k)))
    return pairs


def milp_schedule(accesses: List[Access], requests: List[Request],
                  agility: Callable[[float], float],
                  force_in_schedule: Optional[List[Access]] = None,
                  objective_utilities: Optional[List[float]] = None) -> List[Access]:
    """MILP: maximise total utility subject to slew-time and one-per-request constraints.

    `agility(theta)` returns the slew time (seconds) for a slew of `theta` degrees.
    `objective_utilities`, when supplied, overrides `Access.utility` in the
    objective without mutating the access objects.  This is used by schedule
    repair to assign expected utility to accesses whose cloud state is unknown.
    """
    if not accesses:
        return []
    if objective_utilities is not None and len(objective_utilities) != len(accesses):
        raise ValueError("objective_utilities must match the access list length")

    pairs = _build_conflict_pairs(accesses, agility)
    force_set = {a.aid for a in (force_in_schedule or [])}
    utilities = (objective_utilities if objective_utilities is not None
                 else [a.utility for a in accesses])

    model = Model("Scheduler")
    model.hideOutput()
    x = [model.addVar(vtype="B", name=f"x_{i}") for i in range(len(accesses))]
    for i, a in enumerate(accesses):
        if a.aid in force_set:
            model.addCons(x[i] == 1)
    for i, j in pairs:
        model.addCons(x[i] + x[j] <= 1)
    model.setObjective(quicksum(x[i] * utilities[i]
                                for i in range(len(accesses))),
                       "maximize")
    model.optimize()
    sol = model.getBestSol()
    return [a for i, a in enumerate(accesses) if sol[x[i]] == 1]
