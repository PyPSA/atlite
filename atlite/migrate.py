# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Explicit migration of cutouts to the current schema version.

Write the migrated cutout to a new file and keep the original::

    python -m atlite.migrate path/to/cutout.nc --output path/to/migrated.nc

or, from Python::

    from atlite.migrate import migrate_cutout

    migrate_cutout("path/to/cutout.nc", output="path/to/migrated.nc")

Overwriting the original file requires ``--in-place`` (``in_place=True``).

Each migration step upgrades a cutout file by exactly one schema version and is
registered in :data:`MIGRATIONS` under the version it upgrades from.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import xarray as xr

from atlite.schema import (
    CUTOUT_SCHEMA_VERSION,
    SCHEMA_VERSION_ATTR,
    IncompatibleCutoutError,
    read_schema_version,
    require_supported_schema_version,
    schema_version_from_attrs,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from atlite._types import PathLike

logger = logging.getLogger(__name__)


def _migrate_v0_to_v1(path: Path) -> None:
    """Migrate a cutout file in place from schema version 0 to 1."""
    # v1 only adds the version attribute
    # Appending a dataset without variables updates the meta attributes in the header and leaves the data untouched
    xr.Dataset(attrs={SCHEMA_VERSION_ATTR: 1}).to_netcdf(path, mode="a")


# Maps the schema version a step upgrades from to the step itself
MIGRATIONS: dict[int, Callable[[Path], None]] = {
    0: _migrate_v0_to_v1,
}


def migrate_cutout(
    path: PathLike, output: PathLike | None = None, *, in_place: bool = False
) -> None:
    """
    Migrate a cutout file to the current schema version.

    Exactly one of ``output`` and ``in_place=True`` must be given.

    Parameters
    ----------
    path : str or Path
        Path of the cutout to migrate.
    output : str or Path, optional
        Path where to write the migrated cutout to. Must not exist yet.
    in_place : bool, default False
        Overwrite the original cutout with the migrated cutout.

    Raises
    ------
    ValueError
        If neither or both of ``output`` and ``in_place=True`` are given.
    IncompatibleCutoutError
        If the file is not an atlite cutout or its schema version cannot be
        migrated by this atlite version.
    FileExistsError
        If ``output`` already exists.

    """
    # Overwriting the original must be an explicit choice
    if (output is None) == (not in_place):
        raise ValueError("Specify either an output file or in_place=True")

    path = Path(path)
    with xr.open_dataset(path) as ds:
        attrs = dict(ds.attrs)
    # Files without a version attribute are treated as v0 cutouts, so make sure
    # it is a cutout at all before stamping a version on it
    if "module" not in attrs:
        raise IncompatibleCutoutError(
            f"'{path}' does not appear to be an atlite cutout and is missing the attribute 'module'. "
            f"Aborting."
        )
    version = schema_version_from_attrs(attrs)
    require_supported_schema_version(version, path)

    if output is None:
        target = path
    else:
        output = Path(output)
        if output.exists():
            raise FileExistsError(f"Output file '{output}' already exists.")
        # Migrate a hidden copy next to the output, so `output` only appears
        # once it is complete and the original is not touched before, avoiding
        # corruption on failed upgrades
        target = output.with_name(f".{output.name}.migration")
        shutil.copyfile(path, target)

    try:

        # Actual migration loop. Call all migration functions in order
        for current in range(version, CUTOUT_SCHEMA_VERSION):
            logger.info(
                "Migrating '%s' from schema version %d to %d.",
                path,
                current,
                current + 1,
            )
            MIGRATIONS[current](target)
    except BaseException:
        # Also on KeyboardInterrupt: don't leave a half-migrated copy behind
        if target != path:
            target.unlink(missing_ok=True)
        raise

    if output is not None:
        target.rename(output)


def main(argv: Sequence[str] | None = None) -> int:
    """
    Migrate a cutout from the command line.

    Parameters
    ----------
    argv : sequence of str, optional
        Command line arguments, defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code.

    """
    parser = argparse.ArgumentParser(
        prog="python -m atlite.migrate",
        description="Migrate an atlite cutout to the current schema version "
        f"({CUTOUT_SCHEMA_VERSION}).",
    )
    parser.add_argument("cutout", type=Path, help="Path to the cutout file.")
    # Mirrors migrate_cutout: exactly one destination is required
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument(
        "-o", "--output", type=Path, help="Write to this file, keep the original."
    )
    destination.add_argument(
        "--in-place", action="store_true", help="Overwrite the original file."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        version = read_schema_version(args.cutout)
        migrate_cutout(args.cutout, args.output, in_place=args.in_place)
    except (IncompatibleCutoutError, OSError) as err:
        # Expected failures: report them without a traceback
        logger.error("%s", err)
        return 1

    logger.info(
        "'%s' has schema version %d (was %d).",
        args.output or args.cutout,
        CUTOUT_SCHEMA_VERSION,
        version,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
