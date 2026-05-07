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
HIMAWARI_URL = "https://noaa-himawari9.s3.amazonaws.com/"
EUMETSAT_BASE = "https://api.eumetsat.int/data/download/1.0.0"
EUMETSAT_TOKEN_URL = "https://api.eumetsat.int/token"

GOES_PRODUCT = "ABI-L2-ACMF"
HIMAWARI_PRODUCT = "AHI-L2-FLDK-Clouds"

S3_NAMESPACE = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

MeteosatAuthToken = namedtuple("MeteosatAuthToken", ["token", "expires"])


def _ensure_dirs() -> None:
    for d in (PRODUCTS_DIR, TMP_DIR, DERIVED_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------- GOES ----------

def download_goes_image(time: datetime.datetime, url: str, product: str,
                        savefile: str) -> Path:
    """Fetch a GOES ACM (cloud-mask) NetCDF closest to `time`."""
    _ensure_dirs()
    if time.minute == 0:
        time = time - datetime.timedelta(hours=1)

    day_of_year = time.timetuple().tm_yday
    listing = requests.get(
        f"{url}/?prefix={product}/{time.year}/{day_of_year:03d}/{time.hour:02d}/")
    listing.raise_for_status()
    root = ET.fromstring(listing.text)
    keys = [c.find("s3:Key", S3_NAMESPACE).text
            for c in root.findall("s3:Contents", S3_NAMESPACE)]
    if not keys:
        raise RuntimeError(f"No GOES files for {time.isoformat()}")
    filename = keys[time.minute // 10 if time.minute > 0 else -1]

    blob = requests.get(f"{url}/{filename}")
    blob.raise_for_status()
    out = PRODUCTS_DIR / f"{savefile}.nc"
    out.write_bytes(blob.content)
    return out


def download_goes_east(time: datetime.datetime) -> Path:
    return download_goes_image(time, GOES_EAST_URL, GOES_PRODUCT,
                               f"goes_east_{time.strftime(DATE_FORMAT)}")


def download_goes_west(time: datetime.datetime) -> Path:
    return download_goes_image(time, GOES_WEST_URL, GOES_PRODUCT,
                               f"goes_west_{time.strftime(DATE_FORMAT)}")


# ---------- Himawari ----------

def download_himawari(time: datetime.datetime,
                      savefile: Optional[str] = None) -> Path:
    """Fetch a Himawari AHI cloud-mask NetCDF closest to `time`."""
    _ensure_dirs()
    if savefile is None:
        savefile = f"himawari_{time.strftime(DATE_FORMAT)}"

    listing = requests.get(
        f"{HIMAWARI_URL}/?prefix={HIMAWARI_PRODUCT}/{time.year}/"
        f"{time.month:02d}/{time.day:02d}/{time.hour:02d}{time.minute:02d}")
    listing.raise_for_status()
    root = ET.fromstring(listing.text)
    keys = [c.find("s3:Key", S3_NAMESPACE).text
            for c in root.findall("s3:Contents", S3_NAMESPACE)]
    cmsk = [k for k in keys if "AHI-CMSK" in k]
    if not cmsk:
        raise RuntimeError(f"No Himawari CMSK file for {time.isoformat()}")

    blob = requests.get(f"{HIMAWARI_URL}/{cmsk[0]}")
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


def download_meteosat_image(time: datetime.datetime, auth_token: MeteosatAuthToken,
                            product: str = "zds") -> Path:
    """Fetch a Meteosat cloud-mask GRIB closest to `time`. product: 'zds' or 'iodc'."""
    _ensure_dirs()
    collection = "EO:EUM:DAT:MSG:CLM" + ("-IODC" if product == "iodc" else "")
    coll_safe = collection.replace(":", "%3A")

    url = (f"{EUMETSAT_BASE}/collections/{coll_safe}/dates/{time.year}/"
           f"{time.month:02d}/{time.day:02d}/times/{time.hour:02d}/"
           f"{time.minute:02d}?access_token={auth_token.token}")
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


# ---------- Source registry ----------

# Defined after the download functions so we can reference them directly.
IMAGE_SOURCES: Dict[str, dict] = {
    "goes_west": {
        "description": "GOES West (18)",
        "lon": -137.2,
        "download": download_goes_west,
    },
    "goes_east": {
        "description": "GOES East (16)",
        "lon": -75.2,
        "download": download_goes_east,
    },
    "meteosat_zds": {
        "description": "Meteosat Zero Degree Service",
        "lon": 0.0,
        "download": None,  # requires auth_token; call download_meteosat_image directly
    },
    "meteosat_iodc": {
        "description": "Meteosat Indian Ocean Data Coverage",
        "lon": 45.5,
        "download": None,
    },
    "himawari": {
        "description": "Himawari 9",
        "lon": 140.7,
        "download": download_himawari,
    },
}


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
        downloader(date)

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
    center_lons = np.array([s["lon"] for s in IMAGE_SOURCES.values()])
    center_lon_sorted = np.sort(center_lons)
    cutoffs = (center_lon_sorted + np.roll(center_lon_sorted, 1)) / 2
    cutoffs[0] -= 180

    mask = np.zeros(len(points))
    for i, p in enumerate(points):
        sat_idx = np.searchsorted(cutoffs, p[1])
        if sat_idx == 0:
            sat_idx = len(IMAGE_SOURCES)
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
    lons, lats = np.meshgrid(lon_grid, lat_grid)
    points = np.stack([lats.flatten(), lons.flatten()], axis=1)

    if all_data is None:
        all_data = []
        for name in IMAGE_SOURCES.keys():
            ext = "grb" if name.startswith("meteosat") else "nc"
            all_data.append(load_bcm(
                f"{name}_{time.strftime(DATE_FORMAT)}.{ext}", lazy_load=True))

    mask = sample_global_bcm(all_data, points).reshape(lats.shape)
    return mask, lats, lons


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
