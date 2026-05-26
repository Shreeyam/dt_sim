"""Scenario setup: orbit, accesses, cloud states, MILP baselines."""
import datetime
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
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
    mean_anom: float


def round_to_half_hour(t: datetime.datetime) -> datetime.datetime:
    """Round a datetime to the nearest 30-minute trial-start slot."""
    minute_total = t.minute + t.second / 60 + t.microsecond / 6e7
    base = t.replace(minute=0, second=0, microsecond=0)
    if minute_total >= 45:
        return base + datetime.timedelta(hours=1)
    if minute_total >= 15:
        return base.replace(minute=30)
    return base


def sample_trial_datetimes(cfg: SimConfig) -> List[datetime.datetime]:
    """Deterministically sample unique 30-minute trial starts in cfg's window."""
    if cfg.trial_sampling == "explicit_dates":
        if not cfg.trial_dates_iso:
            raise ValueError(
                "explicit_dates sampling requires cfg.trial_dates_iso.")
        dates = [
            datetime.datetime.fromisoformat(value)
            for value in cfg.trial_dates_iso
        ]
        if len(dates) != cfg.n_trials:
            raise ValueError(
                f"explicit_dates sampling received {len(dates)} dates, "
                f"but n_trials={cfg.n_trials}.")
        if len(set(dates)) != len(dates):
            raise ValueError("explicit_dates sampling received duplicate dates.")
        return dates

    rng = np.random.default_rng(cfg.seed)
    step = datetime.timedelta(minutes=30)
    slots = []
    t = round_to_half_hour(cfg.trial_start)
    end = round_to_half_hour(cfg.trial_end)
    while t <= end:
        slots.append(t)
        t += step
    if cfg.n_trials > len(slots):
        raise ValueError(
            f"Requested {cfg.n_trials} trials, but only {len(slots)} unique "
            "30-minute slots exist in the configured trial window.")
    if cfg.trial_sampling == "date_grid":
        if cfg.n_trials == 1:
            indices = np.array([0], dtype=int)
        else:
            indices = np.rint(
                np.linspace(0, len(slots) - 1, cfg.n_trials)).astype(int)
        if len(np.unique(indices)) != len(indices):
            raise ValueError(
                "date_grid sampling produced duplicate trial starts; reduce "
                "n_trials or widen the trial window.")
        return [slots[int(i)] for i in indices]
    if cfg.trial_sampling == "month_starts":
        months = []
        current = datetime.datetime(cfg.trial_start.year, cfg.trial_start.month, 1)
        if current < cfg.trial_start:
            year = current.year + (1 if current.month == 12 else 0)
            month = 1 if current.month == 12 else current.month + 1
            current = datetime.datetime(year, month, 1)
        while current <= cfg.trial_end:
            months.append(current)
            year = current.year + (1 if current.month == 12 else 0)
            month = 1 if current.month == 12 else current.month + 1
            current = datetime.datetime(year, month, 1)
        if cfg.n_trials != len(months):
            raise ValueError(
                f"month_starts sampling found {len(months)} month starts in "
                f"the configured trial window, but n_trials={cfg.n_trials}.")
        return months
    indices = np.sort(rng.choice(len(slots), size=cfg.n_trials, replace=False))
    return [slots[int(i)] for i in indices]


def _configure_bcm_data_dir(cfg: SimConfig) -> None:
    if cfg.bcm_data_dir:
        from .imagery import set_data_dir
        set_data_dir(cfg.bcm_data_dir)


@lru_cache(maxsize=256)
def _load_global_bcm_cached(t: datetime.datetime, data_dir: str,
                            download_missing: bool):
    from . import imagery

    if data_dir:
        imagery.set_data_dir(data_dir)
    cache = imagery.DERIVED_DIR / f"bcm_{t.strftime(imagery.DATE_FORMAT)}.npz"
    if not download_missing and not cache.exists():
        raise FileNotFoundError(
            f"Missing BCM cache {cache}. Either precompute it or set "
            "bcm_download_missing=True to let dt_sim derive/download it.")
    return imagery.load_global_bcm(t)


def _access_state_fingerprint(accesses: List[Access]) -> str:
    """Hash the access sequence and BCM source metadata used for truth labels."""
    from . import imagery

    payload = {
        "version": 1,
        "sources": {
            name: {
                "lon": imagery.IMAGE_SOURCES[name]["lon"],
                "cadence_min": imagery.IMAGE_SOURCES[name]["cadence_min"],
            }
            for name in sorted(imagery.IMAGE_SOURCES)
        },
        "accesses": [
            [
                a.requestid,
                a.time.isoformat(timespec="seconds"),
                round(float(a.lat), 7),
                round(float(a.long), 7),
                round(float(a.angle), 7),
            ]
            for a in accesses
        ],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:20]


def _access_state_cache_path(accesses: List[Access]) -> tuple[Any, str]:
    from . import imagery

    fingerprint = _access_state_fingerprint(accesses)
    cache_dir = imagery.DERIVED_DIR / "access_states"
    return cache_dir / f"bcm_access_states_{fingerprint}.npz", fingerprint


def _read_access_state_cache(accesses: List[Access]) -> Optional[Dict[int, bool]]:
    path, fingerprint = _access_state_cache_path(accesses)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as ds:
        cached_fingerprint = str(ds["fingerprint"])
        clear = np.array(ds["clear"], dtype=bool)
    if cached_fingerprint != fingerprint or len(clear) != len(accesses):
        return None
    return {a.aid: bool(clear[i]) for i, a in enumerate(accesses)}


def _write_access_state_cache(
    accesses: List[Access],
    states: Dict[int, bool],
    source_by_aid: Dict[int, str],
    sample_time_by_aid: Dict[int, datetime.datetime],
    product_by_aid: Dict[int, str],
) -> None:
    path, fingerprint = _access_state_cache_path(accesses)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        fingerprint=fingerprint,
        clear=np.array([states[a.aid] for a in accesses], dtype=bool),
        source=np.array([source_by_aid[a.aid] for a in accesses], dtype="U32"),
        sample_time=np.array(
            [sample_time_by_aid[a.aid].isoformat(timespec="seconds")
             for a in accesses],
            dtype="U19"),
        product=np.array([product_by_aid[a.aid] for a in accesses], dtype="U512"),
    )


def _source_product_files() -> List[Path]:
    from . import imagery

    return sorted(
        list(imagery.PRODUCTS_DIR.glob("*.nc"))
        + list(imagery.PRODUCTS_DIR.glob("*.grb"))
    )


def _prune_products(product_paths) -> None:
    n_deleted = 0
    paths = sorted({Path(path) for path in product_paths})
    for path in paths:
        try:
            path.unlink()
            n_deleted += 1
        except FileNotFoundError:
            pass
    print(f"[bcm] pruned {n_deleted}/{len(paths)} source products", flush=True)


def make_cloud_states_iid(accesses: List[Access], p_clear: float,
                          seed: int) -> Dict[int, bool]:
    rng = np.random.default_rng(seed)
    return {a.aid: bool(rng.random() < p_clear) for a in accesses}


def make_cloud_states_bcm(accesses: List[Access], cfg: SimConfig) -> Dict[int, bool]:
    """Sample cloud truth at each access imaging time and location."""
    _configure_bcm_data_dir(cfg)
    if cfg.bcm_cache_access_states:
        cached = _read_access_state_cache(accesses)
        if cached is not None:
            path, _ = _access_state_cache_path(accesses)
            print(f"[bcm] access-state cache hit: {path.name} "
                  f"({len(accesses)} accesses)", flush=True)
            if cfg.bcm_prune_products_after_state_cache:
                _prune_products(_source_product_files())
            return cached

    from . import imagery

    states = {}
    source_by_aid = {}
    sample_time_by_aid = {}
    product_by_aid = {}
    product_paths = set()
    groups = defaultdict(list)

    for a in accesses:
        source = imagery.source_for_lon(a.long)
        sample_time = imagery.source_temporal_sample_time(source, a.time)
        groups[(source, sample_time)].append(a)
        source_by_aid[a.aid] = source
        sample_time_by_aid[a.aid] = sample_time

    print(f"[bcm] access-state cache miss: {len(accesses)} accesses, "
          f"{len(groups)} source-time products", flush=True)
    group_items = sorted(groups.items(), key=lambda item: (item[0][1], item[0][0]))
    for idx, ((source, sample_time), group) in enumerate(group_items, start=1):
        product_path = imagery.source_product_path(source, sample_time)
        status = "cached" if product_path.exists() else "download"
        print(f"[bcm] product {idx}/{len(group_items)} {status}: "
              f"{source} {sample_time.isoformat()} "
              f"({len(group)} accesses)", flush=True)
        bcm, lats, lons, product_path = imagery.load_source_bcm(
            source, sample_time, download_missing=cfg.bcm_download_missing)
        product_paths.add(product_path)
        for a in group:
            cloudy = imagery.get_closest_latlong_sample(
                bcm, lats, lons, (a.lat, a.long))
            states[a.aid] = not bool(cloudy)
            product_by_aid[a.aid] = str(product_path)

    if cfg.bcm_cache_access_states:
        _write_access_state_cache(
            accesses, states, source_by_aid, sample_time_by_aid, product_by_aid)
        path, _ = _access_state_cache_path(accesses)
        print(f"[bcm] wrote access-state cache: {path.name} "
              f"({len(accesses)} accesses)", flush=True)
        if cfg.bcm_prune_products_after_state_cache:
            _prune_products(_source_product_files())
    return states


def build_scenario(*, requests, raan: float, cfg: SimConfig,
                   rng: np.random.Generator,
                   t0: Optional[datetime.datetime] = None,
                   mean_anom: Optional[float] = None) -> Optional[Scenario]:
    t0 = cfg.t0 if t0 is None else t0
    t_end = t0 + datetime.timedelta(hours=cfg.horizon_h)
    if mean_anom is None:
        mean_anom = rng.uniform(0, 2 * np.pi)
    orbit = circular_orbit(
        a=Constants.R_E + cfg.altitude_km,
        i=np.deg2rad(cfg.inclination_deg),
        Omega=raan,
        M=mean_anom,
        t=t0,
    )
    accesses = get_accesses(requests, orbit, t_coarse=500,
                            field_of_regard=cfg.field_of_regard_deg,
                            t0=t0, t_end=t_end)
    if len(accesses) < 5:
        return None

    p_clear = 1.0 - cfg.cloud_prob
    if cfg.cloud_source == "iid":
        cloud_states = make_cloud_states_iid(
            accesses, p_clear, seed=int(t0.timestamp()))
    else:
        cloud_states = make_cloud_states_bcm(accesses, cfg)
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
        mean_anom=float(mean_anom),
    )


def reset_states(scenario: Scenario) -> None:
    for a in scenario.accesses:
        a.state = {"observed": False, "cloudy": not scenario.cloud_states[a.aid]}
