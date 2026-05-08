"""
Test: atlite Cutout.prepare with era5_ncar module (solar/influx only).

Montenegro bounding box (approx):
  N 43.6  S 41.8  W 18.4  E 20.4
"""

import logging
import time

import atlite

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

CUTOUT_PATH = "test_cutout_ncar.nc"

if __name__ == "__main__":
    t0 = time.time()

    cutout = atlite.Cutout(
        path=CUTOUT_PATH,
        module="era5_ncar",
        x=slice(18.4, 20.4),
        y=slice(41.8, 43.6),
        time=slice("2022-01-01", "2022-01-31"),
    )
    print(f"Cutout created in {time.time()-t0:.1f}s")
    print(cutout)

    t1 = time.time()
    cutout.prepare(features=["influx"])
    print(f"\nprepare() done in {time.time()-t1:.1f}s")
    print(f"total: {time.time()-t0:.1f}s")

    print("\n=== Cutout data ===")
    print(cutout.data)

    print("\n=== Prepared features ===")
    print(cutout.prepared_features)
