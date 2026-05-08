"""
Smoke test: cold cache download for every feature, then warm cache, compare times.

Usage:  python3 test_ncar_solar.py
"""

import logging
import tempfile
import time
from pathlib import Path

import atlite

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

BBOX = dict(x=slice(18.4, 20.4), y=slice(41.8, 43.6))
TIME = dict(time=slice("2022-01-01", "2022-01-02"))
FEATURES = ["height", "wind", "influx", "temperature", "runoff"]

with tempfile.TemporaryDirectory() as cache_dir:
    cache = Path(cache_dir)

    # Cold run
    t0 = time.time()
    c1 = atlite.Cutout(path=str(cache / "cold.nc"), module="era5_ncar", **BBOX, **TIME)
    c1.prepare(features=FEATURES, tmpdir=str(cache))
    cold_time = time.time() - t0
    cached_files = sorted(cache.glob("era5ncar_*.zarr"))
    print(f"\ncold: {cold_time:.1f}s, {len(cached_files)} files cached")
    print(f"prepared: {sorted(c1.data.data_vars)}")

    # Warm run
    t0 = time.time()
    c2 = atlite.Cutout(path=str(cache / "warm.nc"), module="era5_ncar", **BBOX, **TIME)
    c2.prepare(features=FEATURES, tmpdir=str(cache))
    warm_time = time.time() - t0
    print(f"warm: {warm_time:.1f}s  (speedup: {cold_time/warm_time:.1f}x)")

    expected_vars = {
        "height",
        "wnd100m", "wnd_shear_exp", "wnd_azimuth", "roughness",
        "influx_toa", "influx_direct", "influx_diffuse",
        "albedo", "solar_altitude", "solar_azimuth",
        "temperature", "soil temperature", "dewpoint temperature",
        "runoff",
    }
    missing = expected_vars - set(c2.data.data_vars)
    assert not missing, f"missing vars: {missing}"

    assert warm_time < cold_time / 5, "warm run should be much faster than cold"
    print("OK")
