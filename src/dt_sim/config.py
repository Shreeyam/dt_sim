"""Single source of truth for simulation knobs."""
import datetime
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

VARIANTS = ("never", "always", "renewal", "omniscient")


@dataclass
class SimConfig:
    t0_iso: str = "2025-01-01T00:00:00"
    horizon_h: float = 12.0
    altitude_km: float = 400.0
    inclination_deg: float = 51.6
    n_cities: int = 10000
    cloud_prob: float = 0.66

    slew_at_for_s: float = 25.0
    field_of_regard_deg: float = 45.0
    t_s_ratio: float = 0.36

    fov_deg: float = 45.0
    boresight_pitch_deg: Optional[float] = None
    pitch_candidates: int = 8

    variant: str = "renewal"

    n_trials: int = 20
    seed: int = 42

    out_dir: str = "dt_sim/runs"
    tag: str = "default"

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {self.variant!r}")

    @property
    def t0(self) -> datetime.datetime:
        return datetime.datetime.fromisoformat(self.t0_iso)

    @property
    def t_s(self) -> float:
        return self.t_s_ratio * self.slew_at_for_s

    @property
    def beta(self) -> float:
        return (self.slew_at_for_s - self.t_s) / np.sqrt(self.field_of_regard_deg)

    @property
    def boresight_pitch(self) -> float:
        return self.fov_deg / 2 if self.boresight_pitch_deg is None else self.boresight_pitch_deg

    def agility(self):
        t_s, beta = self.t_s, self.beta
        return lambda theta: t_s + beta * np.sqrt(np.abs(theta))

    def to_dict(self):
        return asdict(self)
