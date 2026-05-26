"""Realistic cloud imagery: GOES / Meteosat / Himawari fetchers + BCM loaders.

Provides Binary Cloud Mask (BCM) data for replacing the synthetic Bernoulli
cloud model with actual meteorological satellite observations.

Pipeline:
    download_*  →  raw image file in data/products/
    load_bcm    →  (data, lat, lon) per-source mask
    sample_global_bcm / derive_global_bcm  →  global gridded BCM
    load_global_bcm  →  cached global BCM (auto-derives if not present)

External dependencies:
    requests, xarray, pyproj, pygrib (Linux only — Meteosat GRIB)

Meteosat downloads need EUMETSAT credentials passed to `get_auth_token_meteosat`.
"""
import datetime
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import xarray as xr
from requests.auth import HTTPBasicAuth

try:
    import pyproj
except ImportError:
    pyproj = None

try:
    import pygrib
except ImportError:
    pygrib = None  # Meteosat GRIB unavailable on this platform

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PRODUCTS_DIR = DATA_DIR / "products"
TMP_DIR = DATA_DIR / "tmp"
DERIVED_DIR = DATA_DIR / "derived"

DATE_FORMAT = "%Y-%m-%d_%H%M%S"

GOES_WEST_URL = "https://noaa-goes18.s3.amazonaws.com/"
GOES_EAST_URL = "https://noaa-goes16.s3.amazonaws.com/"
GOES_EAST_GOES19_URL = "https://noaa-goes19.s3.amazonaws.com/"
GOES_EAST_GOES19_START = datetime.datetime(2025, 4, 7)
HIMAWARI_URL = "https://noaa-himawari9.s3.amazonaws.com/"
EUMETSAT_BASE = "https://api.eumetsat.int/data/download/1.0.0"
EUMETSAT_TOKEN_URL = "https://api.eumetsat.int/token"

GOES_PRODUCT = "ABI-L2-ACMF"
GOES_MAX_NEAREST_DELTA_MIN = 15.0
HIMAWARI_PRODUCT = "AHI-L2-FLDK-Clouds"

S3_NAMESPACE = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

MeteosatAuthToken = namedtuple("MeteosatAuthToken", ["token", "expires"])
_METEOSAT_AUTH_TOKEN: Optional[MeteosatAuthToken] = None
_DOTENV_LOADED = False


def _ensure_dirs() -> None:
    for d in (PRODUCTS_DIR, TMP_DIR, DERIVED_DIR):
        d.mkdir(parents=True, exist_ok=True)


def set_data_dir(path: str | Path) -> None:
    """Point imagery loaders at a different data cache root."""
    global DATA_DIR, PRODUCTS_DIR, TMP_DIR, DERIVED_DIR
    DATA_DIR = Path(path)
    PRODUCTS_DIR = DATA_DIR / "products"
    TMP_DIR = DATA_DIR / "tmp"
    DERIVED_DIR = DATA_DIR / "derived"


# ---------- GOES ----------

def _goes_key_start_time(key: str) -> Optional[datetime.datetime]:
    match = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})", key)
    if match is None:
        return None
    year, day_of_year, hour, minute, second = map(int, match.groups())
    return (datetime.datetime(year, 1, 1)
            + datetime.timedelta(days=day_of_year - 1,
                                 hours=hour, minutes=minute, seconds=second))


def _list_goes_keys(time: datetime.datetime, url: str,
                    product: str) -> List[str]:
    day_of_year = time.timetuple().tm_yday
    listing = requests.get(
        f"{url.rstrip('/')}/?prefix={product}/{time.year}/"
        f"{day_of_year:03d}/{time.hour:02d}/")
    listing.raise_for_status()
    root = ET.fromstring(listing.text)
    return [c.find("s3:Key", S3_NAMESPACE).text
            for c in root.findall("s3:Contents", S3_NAMESPACE)]


def _list_goes_day_key_times(time: datetime.datetime, url: str,
                             product: str
                             ) -> List[tuple[str, datetime.datetime]]:
    day_of_year = time.timetuple().tm_yday
    listing = requests.get(
        f"{url.rstrip('/')}/?prefix={product}/{time.year}/{day_of_year:03d}/")
    listing.raise_for_status()
    root = ET.fromstring(listing.text)
    keyed_times = []
    for content in root.findall("s3:Contents", S3_NAMESPACE):
        key = content.find("s3:Key", S3_NAMESPACE).text
        key_time = _goes_key_start_time(key)
        if key_time is not None:
            keyed_times.append((key, key_time))
    return keyed_times


def _resolve_goes_product(time: datetime.datetime, url: str,
                          product: str) -> tuple[str, datetime.datetime]:
    keyed_times = _list_goes_day_key_times(time, url, product)
    if not keyed_times:
        bucket = url.rstrip("/").rsplit("/", 1)[-1]
        raise RuntimeError(
            f"No GOES {product} files for {time.date().isoformat()} in {bucket}.")
    filename, product_time = min(
        keyed_times, key=lambda kt: abs((kt[1] - time).total_seconds()))
    delta_min = abs((product_time - time).total_seconds()) / 60
    if delta_min > GOES_MAX_NEAREST_DELTA_MIN:
        before = [t for _, t in keyed_times if t < time]
        after = [t for _, t in keyed_times if t > time]
        prev_s = max(before).isoformat() if before else "none"
        next_s = min(after).isoformat() if after else "none"
        bucket = url.rstrip("/").rsplit("/", 1)[-1]
        raise RuntimeError(
            f"No GOES {product} file close enough to {time.isoformat()} in "
            f"{bucket}. Nearest is {product_time.isoformat()} "
            f"({delta_min:.1f} min away); previous={prev_s}, next={next_s}.")
    return filename, product_time


def _goes_east_url_for_time(time: datetime.datetime) -> str:
    return GOES_EAST_GOES19_URL if time >= GOES_EAST_GOES19_START else GOES_EAST_URL


def download_goes_image(time: datetime.datetime, url: str, product: str,
                        savefile: str) -> Path:
    """Fetch a GOES ACM (cloud-mask) NetCDF closest to `time`."""
    _ensure_dirs()
    filename, _ = _resolve_goes_product(time, url, product)

    blob = requests.get(f"{url.rstrip('/')}/{filename}")
    blob.raise_for_status()
    out = PRODUCTS_DIR / f"{savefile}.nc"
    out.write_bytes(blob.content)
    return out


def download_goes_east(time: datetime.datetime) -> Path:
    return download_goes_image(time, _goes_east_url_for_time(time), GOES_PRODUCT,
                               f"goes_east_{time.strftime(DATE_FORMAT)}")


def download_goes_west(time: datetime.datetime) -> Path:
    return download_goes_image(time, GOES_WEST_URL, GOES_PRODUCT,
                               f"goes_west_{time.strftime(DATE_FORMAT)}")


# ---------- Himawari ----------

def _list_himawari_keys(time: datetime.datetime) -> List[str]:
    listing = requests.get(
        f"{HIMAWARI_URL.rstrip('/')}/?prefix={HIMAWARI_PRODUCT}/{time.year}/"
        f"{time.month:02d}/{time.day:02d}/{time.hour:02d}{time.minute:02d}")
    listing.raise_for_status()
    root = ET.fromstring(listing.text)
    return [c.find("s3:Key", S3_NAMESPACE).text
            for c in root.findall("s3:Contents", S3_NAMESPACE)]


def _resolve_himawari_product(time: datetime.datetime) -> str:
    cmsk = [k for k in _list_himawari_keys(time) if "AHI-CMSK" in k]
    if not cmsk:
        raise RuntimeError(f"No Himawari CMSK file for {time.isoformat()}")
    return cmsk[0]


def download_himawari(time: datetime.datetime,
                      savefile: Optional[str] = None) -> Path:
    """Fetch a Himawari AHI cloud-mask NetCDF closest to `time`."""
    _ensure_dirs()
    if savefile is None:
        savefile = f"himawari_{time.strftime(DATE_FORMAT)}"

    filename = _resolve_himawari_product(time)
    blob = requests.get(f"{HIMAWARI_URL.rstrip('/')}/{filename}")
    blob.raise_for_status()
    out = PRODUCTS_DIR / f"{savefile}.nc"
    out.write_bytes(blob.content)
    return out


# ---------- Meteosat ----------

def get_auth_token_meteosat(key: str, secret: str) -> MeteosatAuthToken:
    """Fetch an EUMETSAT API access token. Pass your client credentials."""
    response = requests.post(
        EUMETSAT_TOKEN_URL,
        data={"grant_type": "client_credentials"},
        auth=HTTPBasicAuth(key, secret),
    )
    if response.status_code != 200:
        raise RuntimeError(f"EUMETSAT auth failed: {response.status_code} {response.text}")
    body = response.json()
    expires = datetime.datetime.now() + datetime.timedelta(seconds=body["expires_in"])
    return MeteosatAuthToken(body["access_token"], expires)


def revoke_auth_token_meteosat(auth_token: MeteosatAuthToken) -> bool:
    response = requests.post("https://api.eumetsat.int/revoke",
                             data={"token": auth_token.token})
    return response.status_code == 200


def _dotenv_paths() -> List[Path]:
    package_root = Path(__file__).resolve().parents[2]
    candidates = [
        Path.cwd() / ".env",
        Path.cwd() / "dt_sim" / ".env",
        package_root / ".env",
    ]
    candidates.extend(parent / ".env" for parent in Path.cwd().parents)
    out = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            out.append(path)
    return out


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export "):].strip()
    if "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None
    if ((value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))):
        value = value[1:-1]
    return key, value


def _load_dotenv_once() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    for path in _dotenv_paths():
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                parsed = _parse_dotenv_line(line)
                if parsed is None:
                    continue
                key, value = parsed
                os.environ.setdefault(key, value)
        return


def _env_any(names: tuple[str, ...]) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _get_meteosat_auth_token_from_env() -> MeteosatAuthToken:
    """Get/reuse an EUMETSAT token from environment or a local .env file."""
    global _METEOSAT_AUTH_TOKEN
    if (_METEOSAT_AUTH_TOKEN is not None
            and _METEOSAT_AUTH_TOKEN.expires
            > datetime.datetime.now() + datetime.timedelta(minutes=2)):
        return _METEOSAT_AUTH_TOKEN
    _load_dotenv_once()
    key = _env_any(("EUMETSAT_KEY", "EUMETSAT_CONSUMER_KEY", "EUMETSAT_API_KEY"))
    secret = _env_any((
        "EUMETSAT_SECRET",
        "EUMETSAT_CONSUMER_SECRET",
        "EUMETSAT_API_SECRET",
    ))
    if not key or not secret:
        raise RuntimeError(
            "Missing EUMETSAT credentials. Set EUMETSAT_KEY/EUMETSAT_SECRET "
            "in the environment or a local .env file.")
    _METEOSAT_AUTH_TOKEN = get_auth_token_meteosat(key, secret)
    return _METEOSAT_AUTH_TOKEN


def _meteosat_download_url(time: datetime.datetime,
                           auth_token: MeteosatAuthToken,
                           product: str = "zds") -> str:
    collection = "EO:EUM:DAT:MSG:CLM" + ("-IODC" if product == "iodc" else "")
    coll_safe = collection.replace(":", "%3A")
    return (f"{EUMETSAT_BASE}/collections/{coll_safe}/dates/{time.year}/"
            f"{time.month:02d}/{time.day:02d}/times/{time.hour:02d}/"
            f"{time.minute:02d}?access_token={auth_token.token}")


def download_meteosat_image(time: datetime.datetime, auth_token: MeteosatAuthToken,
                            product: str = "zds") -> Path:
    """Fetch a Meteosat cloud-mask GRIB closest to `time`. product: 'zds' or 'iodc'."""
    _ensure_dirs()
    url = _meteosat_download_url(time, auth_token, product)
    response = requests.get(url, headers={"Authorization": f"Bearer {auth_token.token}"})
    if response.status_code != 200:
        raise RuntimeError(f"Meteosat download failed: {response.status_code} {response.text}")

    zip_path = TMP_DIR / f"meteosat_{product}_{time.strftime(DATE_FORMAT)}.zip"
    zip_path.write_bytes(response.content)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(TMP_DIR)

    msg_id = 2 if product == "iodc" else 3
    prefix = (f"MSG{msg_id}-SEVI-MSGCLMK-0100-0100-{time.year}{time.month:02d}"
              f"{time.day:02d}{time.hour:02d}{time.minute:02d}")
    target = PRODUCTS_DIR / f"meteosat_{product}_{time.strftime(DATE_FORMAT)}.grb"
    for root_dir, _, files in os.walk(TMP_DIR):
        for f in files:
            if f.startswith(prefix) and f.endswith(".grb"):
                Path(root_dir, f).rename(target)
                return target
    raise RuntimeError(f"Meteosat GRIB not found after extraction: {prefix}")


def download_meteosat_zds(time: datetime.datetime) -> Path:
    return download_meteosat_image(
        time, _get_meteosat_auth_token_from_env(), product="zds")


def download_meteosat_iodc(time: datetime.datetime) -> Path:
    return download_meteosat_image(
        time, _get_meteosat_auth_token_from_env(), product="iodc")


# ---------- Source registry ----------

# Defined after the download functions so we can reference them directly.
IMAGE_SOURCES: Dict[str, dict] = {
    "goes_west": {
        "description": "GOES West (18)",
        "lon": -137.2,
        "cadence_min": 10,
        "download": download_goes_west,
    },
    "goes_east": {
        "description": "GOES East (16)",
        "lon": -75.2,
        "cadence_min": 10,
        "download": download_goes_east,
    },
    "meteosat_zds": {
        "description": "Meteosat Zero Degree Service",
        "lon": 0.0,
        "cadence_min": 15,
        "download": download_meteosat_zds,
    },
    "meteosat_iodc": {
        "description": "Meteosat Indian Ocean Data Coverage",
        "lon": 45.5,
        "cadence_min": 15,
        "download": download_meteosat_iodc,
    },
    "himawari": {
        "description": "Himawari 9",
        "lon": 140.7,
        "cadence_min": 10,
        "download": download_himawari,
    },
}


def round_to_nearest_minutes(time: datetime.datetime,
                             minutes: int) -> datetime.datetime:
    """Round a timestamp to the nearest fixed-minute cadence."""
    if minutes <= 0:
        raise ValueError("minutes must be positive")
    step_s = minutes * 60
    day_start = time.replace(hour=0, minute=0, second=0, microsecond=0)
    elapsed_s = (time - day_start).total_seconds()
    rounded_s = int(np.floor(elapsed_s / step_s + 0.5) * step_s)
    return day_start + datetime.timedelta(seconds=rounded_s)


def source_temporal_sample_time(source: str,
                                time: datetime.datetime) -> datetime.datetime:
    """Nearest available cloud-mask time for a source."""
    return round_to_nearest_minutes(time, IMAGE_SOURCES[source]["cadence_min"])


def _source_names_by_lon() -> List[str]:
    return sorted(IMAGE_SOURCES, key=lambda name: IMAGE_SOURCES[name]["lon"])


def source_for_lon(lon: float) -> str:
    """Pick the geostationary source whose sub-satellite longitude is closest."""
    lon = ((lon + 180.0) % 360.0) - 180.0
    source_names = _source_names_by_lon()
    center_lons = np.array([IMAGE_SOURCES[name]["lon"] for name in source_names])
    cutoffs = (center_lons + np.roll(center_lons, 1)) / 2
    cutoffs[0] -= 180
    sat_idx = np.searchsorted(cutoffs, lon)
    if sat_idx == 0:
        sat_idx = len(source_names)
    return source_names[sat_idx - 1]


def source_product_filename(source: str, time: datetime.datetime) -> str:
    ext = "grb" if source.startswith("meteosat") else "nc"
    return f"{source}_{time.strftime(DATE_FORMAT)}.{ext}"


def source_product_path(source: str, time: datetime.datetime) -> Path:
    return PRODUCTS_DIR / source_product_filename(source, time)


def source_product_status(source: str,
                          time: datetime.datetime) -> tuple[bool, str]:
    """Check whether the source product exists locally or remotely."""
    source_time = source_temporal_sample_time(source, time)
    path = source_product_path(source, source_time)
    if path.exists():
        return True, f"local {path.name}"

    try:
        if source == "goes_east":
            key, product_time = _resolve_goes_product(
                source_time, _goes_east_url_for_time(source_time), GOES_PRODUCT)
            return True, f"remote {key} ({product_time.isoformat()})"
        if source == "goes_west":
            key, product_time = _resolve_goes_product(
                source_time, GOES_WEST_URL, GOES_PRODUCT)
            return True, f"remote {key} ({product_time.isoformat()})"
        if source == "himawari":
            key = _resolve_himawari_product(source_time)
            return True, f"remote {key}"
        if source == "meteosat_zds":
            token = _get_meteosat_auth_token_from_env()
            response = requests.get(
                _meteosat_download_url(source_time, token, "zds"),
                headers={"Authorization": f"Bearer {token.token}"},
                stream=True,
            )
            if response.status_code == 200:
                response.close()
                return True, "remote EUMETSAT zds"
            text = response.text[:200]
            response.close()
            return False, f"EUMETSAT zds status {response.status_code}: {text}"
        if source == "meteosat_iodc":
            token = _get_meteosat_auth_token_from_env()
            response = requests.get(
                _meteosat_download_url(source_time, token, "iodc"),
                headers={"Authorization": f"Bearer {token.token}"},
                stream=True,
            )
            if response.status_code == 200:
                response.close()
                return True, "remote EUMETSAT iodc"
            text = response.text[:200]
            response.close()
            return False, f"EUMETSAT iodc status {response.status_code}: {text}"
    except Exception as exc:
        return False, str(exc)

    return False, f"unknown BCM source {source!r}"


def load_source_bcm(source: str, time: datetime.datetime,
                    download_missing: bool = False
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    """Load one source at its nearest cadence time for `time`."""
    source_time = source_temporal_sample_time(source, time)
    filename = source_product_filename(source, source_time)
    path = PRODUCTS_DIR / filename
    if not download_missing and not path.exists():
        raise FileNotFoundError(
            f"Missing BCM product {path}. Precompute it or set "
            "bcm_download_missing=True.")
    data, lats, lons = load_bcm(filename, lazy_load=download_missing)
    return data, lats, lons, path


# ---------- BCM loaders ----------

def load_bcm(filename: str, lazy_load: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a per-source binary cloud mask. Returns (mask, lat_grid, lon_grid).

    `filename` is relative to data/products/. With `lazy_load=True`, the file
    is downloaded if missing (requires no Meteosat auth — those need explicit
    download_meteosat_image first).
    """
    pure = os.path.basename(filename)
    full = PRODUCTS_DIR / filename

    if lazy_load and not full.exists():
        date = datetime.datetime.strptime(
            "_".join(pure.split(".")[0].split("_")[-2:]), DATE_FORMAT)
        source = "_".join(pure.split("_")[:-2])
        downloader = IMAGE_SOURCES[source]["download"]
        if downloader is None:
            raise RuntimeError(
                f"Source {source!r} requires manual download with auth credentials.")
        print(f"[bcm] downloading {source} {date.isoformat()} -> {full.name}",
              flush=True)
        downloader(date)
        print(f"[bcm] downloaded {source} {date.isoformat()} -> {full.name}",
              flush=True)

    if pure.startswith("goes"):
        if pyproj is None:
            raise ImportError("pyproj required for GOES BCMs")
        with xr.open_dataset(full) as ds:
            data = ds["BCM"]
            proj_info = ds["goes_imager_projection"].attrs
            h = proj_info["perspective_point_height"]
            lon_origin = proj_info["longitude_of_projection_origin"]
            sweep_axis = proj_info["sweep_angle_axis"]
            proj = pyproj.Proj(proj="geos", h=h, lon_0=lon_origin, sweep=sweep_axis)
            x = ds["x"].values * h
            y = ds["y"].values * h
        x_mesh, y_mesh = np.meshgrid(x, y)
        lon, lat = proj(x_mesh, y_mesh, inverse=True)
        return data.to_numpy(), lat, lon

    if pure.startswith("meteosat"):
        if pygrib is None:
            raise ImportError("pygrib required for Meteosat BCMs (Linux only)")
        with pygrib.open(str(full)) as grbs:
            data = grbs[1].values
            lat = grbs[1].latitudes.reshape((3712, 3712))
            lon = grbs[1].longitudes.reshape((3712, 3712))
        # Meteosat cloud-mask convention: code 3 = no-data, code 2 = cloudy.
        lat[data == 3] = np.nan
        lon[data == 3] = np.nan
        lon[lon > 180] -= 360
        data[data == 3] = np.nan
        data[~np.isnan(data)] = (data[~np.isnan(data)] == 2)
        return data[::-1, ::-1], lat, lon

    if pure.startswith("himawari"):
        with xr.open_dataset(full) as ds:
            data = ds["CloudMaskBinary"].values
            lat = ds["Latitude"].values
            lon = ds["Longitude"].values
        return data, lat, lon

    raise ValueError(f"Unknown image source for filename: {pure}")


def get_closest_latlong_sample(data: np.ndarray, lats: np.ndarray,
                               lons: np.ndarray, point) -> float:
    lat, lon = point
    lats_1d = lats[:, lats.shape[0] // 2]
    lat_idx = np.nanargmin(np.abs(lats_1d - lat))
    lon_idx = np.nanargmin(np.abs(lons[lat_idx, :] - lon))
    return data[lat_idx, lon_idx]


def sample_global_bcm(all_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                      points: np.ndarray) -> np.ndarray:
    """Sample BCM at each (lat, lon) point. Picks the geo satellite whose
    sub-satellite longitude is closest."""
    source_names = _source_names_by_lon()
    center_lons = np.array([IMAGE_SOURCES[name]["lon"] for name in source_names])
    cutoffs = (center_lons + np.roll(center_lons, 1)) / 2
    cutoffs[0] -= 180

    mask = np.zeros(len(points))
    for i, p in enumerate(points):
        sat_idx = np.searchsorted(cutoffs, p[1])
        if sat_idx == 0:
            sat_idx = len(source_names)
        data, lats, lons = all_data[sat_idx - 1]
        mask[i] = get_closest_latlong_sample(data, lats, lons, p)
    return mask


def derive_global_bcm(time: datetime.datetime,
                      all_data: Optional[List] = None,
                      n_lat: int = 1000, n_lon: int = 3000,
                      max_lat: float = 60.0
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive a (n_lat, n_lon) global cloud-mask grid for `time`."""
    lat_grid = np.linspace(-max_lat, max_lat, n_lat)
    lon_grid = np.linspace(-180, 180, n_lon)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    points = np.stack([lat_mesh.flatten(), lon_mesh.flatten()], axis=1)

    if all_data is None:
        all_data = []
        for name in _source_names_by_lon():
            data, source_lats, source_lons, _ = load_source_bcm(
                name, time, download_missing=True)
            all_data.append((data, source_lats, source_lons))

    mask = sample_global_bcm(all_data, points).reshape(lat_mesh.shape)
    return mask, lat_mesh, lon_mesh


def load_global_bcm(time: datetime.datetime,
                    n_lat: int = 1000, n_lon: int = 3000
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (or derive + cache) the global BCM grid for `time`."""
    _ensure_dirs()
    cache = DERIVED_DIR / f"bcm_{time.strftime(DATE_FORMAT)}.npz"
    if cache.exists():
        ds = np.load(cache)
        return ds["BCM"], ds["Latitude"], ds["Longitude"]
    data, lats, lons = derive_global_bcm(time, None, n_lat, n_lon)
    np.savez(cache, BCM=data, Latitude=lats, Longitude=lons)
    return data, lats, lons
