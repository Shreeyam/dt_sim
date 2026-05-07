"""Scenario setup: orbit, accesses, cloud states, MILP baselines."""
import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .access import Access, get_accesses
from .config import SimConfig
from .constants import Constants
from .orbits import circular_orbit
from .scheduling import milp_schedule


@dataclass
class Scenario:
    requests: List[Any]
    orbit: Any
    accesses: List[Access]
    cloud_states: Dict[int, bool]   # access.aid -> True if clear
    schedule_conv: List[Access]
    schedule_omni: List[Access]
    t0: datetime.datetime
    t_end: datetime.datetime
    raan: float


def build_scenario(*, requests, raan: float, cfg: SimConfig,
                   rng: np.random.Generator) -> Optional[Scenario]:
    t0 = cfg.t0
    t_end = t0 + datetime.timedelta(hours=cfg.horizon_h)
    orbit = circular_orbit(
        a=Constants.R_E + cfg.altitude_km,
        i=np.deg2rad(cfg.inclination_deg),
        Omega=raan,
        M=rng.uniform(0, 2 * np.pi),
        t=t0,
    )
    accesses = get_accesses(requests, orbit, t_coarse=500,
                            field_of_regard=cfg.field_of_regard_deg,
                            t0=t0, t_end=t_end)
    if len(accesses) < 5:
        return None

    p_clear = 1.0 - cfg.cloud_prob
    cloud_states = {a.aid: bool(rng.random() < p_clear) for a in accesses}
    for a in accesses:
        a.state = {"observed": False, "cloudy": not cloud_states[a.aid]}

    schedule_conv = milp_schedule(accesses, requests, cfg.agility())
    if len(schedule_conv) < 3:
        return None
    clear = [a for a in accesses if cloud_states[a.aid]]
    schedule_omni = milp_schedule(clear, requests, cfg.agility()) if clear else []

    return Scenario(
        requests=requests,
        orbit=orbit,
        accesses=accesses,
        cloud_states=cloud_states,
        schedule_conv=schedule_conv,
        schedule_omni=schedule_omni,
        t0=t0,
        t_end=t_end,
        raan=float(raan),
    )


def reset_states(scenario: Scenario) -> None:
    for a in scenario.accesses:
        a.state = {"observed": False, "cloudy": not scenario.cloud_states[a.aid]}
