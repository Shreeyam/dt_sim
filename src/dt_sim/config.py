"""Single source of truth for simulation knobs."""
import datetime
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np

VARIANTS = (
    "never", "greedy", "two_anchor", "break_even", "omniscient",
)
CLOUD_SOURCES = ("iid", "bcm")
TRIAL_SAMPLING = ("dates", "date_grid", "month_starts", "explicit_dates", "raan")


@dataclass
class SimConfig:
    t0_iso: str = "2025-01-01T00:00:00"
    horizon_h: float = 24.0
    altitude_km: float = 400.0
    inclination_deg: float = 51.6
    n_cities: int = 10000
    cloud_prob: float = 0.66
    cloud_source: str = "iid"
    bcm_data_dir: Optional[str] = None
    bcm_download_missing: bool = False
    bcm_cache_access_states: bool = True
    bcm_prune_products_after_state_cache: bool = False

    slew_at_for_s: float = 25.0
    field_of_regard_deg: float = 45.0
    t_s_ratio: Optional[float] = None
    t_s: float = 10.0

    fov_deg: float = 45.0
    boresight_pitch_deg: Optional[float] = None
    pitch_candidates: int = 64
    cap_far_edge_at_limb: bool = False

    variant: str = "two_anchor"

    n_trials: int = 20
    seed: int = 42
    trial_sampling: str = "dates"
    trial_start_iso: str = "2025-01-01T00:00:00"
    trial_end_iso: str = "2025-12-31T23:30:00"
    trial_dates_iso: Optional[Tuple[str, ...]] = None
    match_ground_track: bool = False
    raan_deg: float = 0.0
    mean_anom_deg: float = 0.0

    out_dir: str = "runs"
    tag: str = "default"

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {self.variant!r}")
        if self.cloud_source not in CLOUD_SOURCES:
            raise ValueError(f"cloud_source must be one of {CLOUD_SOURCES}, got {self.cloud_source!r}")
        if self.trial_sampling not in TRIAL_SAMPLING:
            raise ValueError(f"trial_sampling must be one of {TRIAL_SAMPLING}, got {self.trial_sampling!r}")

    @property
    def t0(self) -> datetime.datetime:
        return datetime.datetime.fromisoformat(self.t0_iso)

    @property
    def trial_start(self) -> datetime.datetime:
        return datetime.datetime.fromisoformat(self.trial_start_iso)

    @property
    def trial_end(self) -> datetime.datetime:
        return datetime.datetime.fromisoformat(self.trial_end_iso)

    @property
    def settle_s(self) -> float:
        if self.t_s_ratio is not None:
            return self.t_s_ratio * self.slew_at_for_s
        return self.t_s

    @property
    def beta(self) -> float:
        return max(self.slew_at_for_s - self.settle_s, 0.0) / np.sqrt(self.field_of_regard_deg)

    @property
    def boresight_pitch(self) -> float:
        return self.fov_deg / 2 if self.boresight_pitch_deg is None else self.boresight_pitch_deg

    def agility(self):
        t_s, beta = self.settle_s, self.beta
        return lambda theta: t_s + beta * np.sqrt(np.abs(theta))

    def to_dict(self):
        return asdict(self)
