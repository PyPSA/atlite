"""
Quick smoke test: cold cache download, then warm cache, compare times.

Usage:  python3 test_ncar_solar.py
"""

import logging
import tempfile
import time
from pathlib import Path

import atlite

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

BBOX = dict(x=slice(18.4, 20.4), y=slice(41.8, 43.6))
TIME = dict(time=slice("2022-01-01", "2022-01-31"))

with tempfile.TemporaryDirectory() as cache_dir:
    cache = Path(cache_dir)

    # Cold run
    t0 = time.time()
    c1 = atlite.Cutout(path=str(cache / "cold.nc"), module="era5_ncar", **BBOX, **TIME)
    c1.prepare(features=["influx"], tmpdir=str(cache))
    cold_time = time.time() - t0
    cached_files = sorted(cache.glob("era5ncar_*.zarr"))
    print(f"\ncold: {cold_time:.1f}s, {len(cached_files)} files cached")

    # Warm run
    t0 = time.time()
    c2 = atlite.Cutout(path=str(cache / "warm.nc"), module="era5_ncar", **BBOX, **TIME)
    c2.prepare(features=["influx"], tmpdir=str(cache))
    warm_time = time.time() - t0
    print(f"warm: {warm_time:.1f}s  (speedup: {cold_time/warm_time:.1f}x)")

    assert warm_time < cold_time / 5, "warm run should be much faster than cold"
    print("OK")
