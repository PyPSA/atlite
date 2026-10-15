# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Schema version handling for cutouts.

Every cutout written by atlite carries an integer schema version in the global
attribute :data:`SCHEMA_VERSION_ATTR`. Legacy cutouts written before this attribute
was introduced are treated as having version ``0``.

atlite does not migrates cutouts between versions automatically.
Loading a cutout with an older but supported version emits a  :class:`OutdatedCutoutWarning`.
Loading a cutout with an unsupported version raises a :class:`IncompatibleCutoutError`.
Outdated cutouts need to be migrated explicitly with ``python -m atlite.migrate`` or
:func:`atlite.migrate.migrate_cutout`.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Mapping

    from atlite._types import PathLike

logger = logging.getLogger(__name__)

# Name of the attribute stored as netCDF attr
SCHEMA_VERSION_ATTR = "atlite_cutout_schema_version"

# Version written to newly created cutouts
# Bump with integer version when the schema changes
CUTOUT_SCHEMA_VERSION = 1

# Versions this release of atlite can read
SUPPORTED_SCHEMA_VERSIONS = frozenset({0, 1})

# Version assumed for cutouts written before the schema number was introduced
# keep this for legacy compatability, even if no longer supported by the current version
LEGACY_SCHEMA_VERSION = 0


class IncompatibleCutoutError(ValueError):
    """Raised when a cutout has a schema version atlite cannot read."""


class OutdatedCutoutWarning(FutureWarning):
    """Emitted when a cutout has an older, still supported schema version."""


def schema_version_from_attrs(attrs: Mapping[str, Any]) -> int:
    """
    Get the cutout schema version from the global attributes.

    Parameters
    ----------
    attrs : Mapping
        Global attributes of a cutout, e.g. ``cutout.data.attrs``.

    Returns
    -------
    int
        The schema version or `LEGACY_SCHEMA_VERSION` if not present.

    """
    if SCHEMA_VERSION_ATTR not in attrs:
        logger.debug(
            "Cutout has no attribute '%s', assuming schema version %d.",
            SCHEMA_VERSION_ATTR,
            LEGACY_SCHEMA_VERSION,
        )
        return LEGACY_SCHEMA_VERSION
    return int(attrs[SCHEMA_VERSION_ATTR])


def read_schema_version(path: PathLike) -> int:
    """
    Read the cutout schema version from a file without loading its data.

    The dataset is opened lazily, so only metadata and the small coordinate
    indexes are read. This is cheap even for large cutouts.

    Parameters
    ----------
    path : str or Path
        Path to the cutout file.

    Returns
    -------
    int
        The schema version, :data:`LEGACY_SCHEMA_VERSION` if not present.

    """
    with xr.open_dataset(Path(path)) as ds:
        return schema_version_from_attrs(ds.attrs)


def _cutout_name(path: PathLike | None) -> str:
    return "This cutout" if path is None else f"The cutout '{path}'"


def migration_instructions(path: PathLike | None = None) -> str:
    """
    Build the instructions for migrating a cutout to the current schema version.

    Parameters
    ----------
    path : str or Path, optional
        Path of the cutout. Placeholders are used if not given.

    Returns
    -------
    str
        The instructions, including the shell and Python commands to migrate.

    """
    source = Path("cutout.nc") if path is None else Path(path)
    output = source.with_name(f"{source.stem}-v{CUTOUT_SCHEMA_VERSION}{source.suffix}")

    return (
        "Migrate the cutout to the current version; "
        "the original file is not changed.\n"
        "\n"
        "  From the command line:\n"
        f'    python -m atlite.migrate "{source}" --output "{output}"\n'
        "\n"
        "  From Python:\n"
        "    from atlite.migrate import migrate_cutout\n"
        f"    migrate_cutout({str(source)!r}, output={str(output)!r})\n"
        "\n"
        "To overwrite the original file, pass --in-place on the command "
        "line or in_place=True in Python, without an output file."
    )


def outdated_cutout_message(version: int, path: PathLike | None = None) -> str:
    """
    Build the message explaining that a cutout is outdated and must be migrated to be used.

    Parameters
    ----------
    version : int
        Schema version of the cutout.
    path : str or Path, optional
        Path of the cutout. Placeholders are used if not given.

    Returns
    -------
    str
        The message, including the shell and Python commands to migrate.

    """
    return (
        f"The cutout uses cutout schema version {version}. "
        f"This version of atlite can only modify cutouts of version {CUTOUT_SCHEMA_VERSION}. "
        f"Version {version} can still be read but not modified."
        "\n" + migration_instructions(path)
    )


def require_current_schema_version(version: int, path: PathLike | None = None) -> None:
    """
    Require a cutout to have the current schema version before it is modified.

    atlite only writes the current schema version. Modifying a cutout with an
    older version would mix both versions in one file, so it has to be
    migrated explicitly first.

    Parameters
    ----------
    version : int
        Schema version of the cutout.
    path : str or Path, optional
        Path of the cutout, used in messages.

    Raises
    ------
    IncompatibleCutoutError
        If ``version`` is not :data:`CUTOUT_SCHEMA_VERSION`.

    """
    if version != CUTOUT_SCHEMA_VERSION:
        raise IncompatibleCutoutError(
            f"This cutout uses cutout schema version {version}. "
            f"This version of atlite can only modify cutouts of "
            f"version {CUTOUT_SCHEMA_VERSION}.\n"
            "\n" + migration_instructions(path)
        )


def require_supported_schema_version(
    version: int, path: PathLike | None = None
) -> None:
    """
    Require a cutout schema version to be supported by this atlite version.

    Parameters
    ----------
    version : int
        Schema version of the cutout.
    path : str or Path, optional
        Path of the cutout, used in messages.

    Raises
    ------
    IncompatibleCutoutError
        If ``version`` is not supported in `SUPPORTED_SCHEMA_VERSIONS`.

    """
    supported = ", ".join(map(str, sorted(SUPPORTED_SCHEMA_VERSIONS)))

    if version not in SUPPORTED_SCHEMA_VERSIONS:
        if version > CUTOUT_SCHEMA_VERSION:
            raise IncompatibleCutoutError(
                f"Cutout has schema version {version}, which is newer than "
                f"this version of atlite supports ({supported}). "
                "Upgrade atlite to use this cutout."
            )
        raise IncompatibleCutoutError(
            f"Cutout has schema version {version}, which is no longer "
            f"supported by this version of atlite ({supported}). "
            f"Migrate or re-create the cutout."
        )


def check_schema_version(version: int, path: PathLike | None = None) -> None:
    """
    Check whether a cutout schema version can be read by this atlite version.

    Parameters
    ----------
    version : int
        Schema version of the cutout.
    path : str or Path, optional
        Path of the cutout, used in messages.

    Raises
    ------
    IncompatibleCutoutError
        If ``version`` is not supported.

    Warns
    -----
    OutdatedCutoutWarning
        If ``version`` is supported but older than the current version.

    """  # noqa: DOC502 (IncompatibleCutoutError is raised by require_supported_schema_version)
    require_supported_schema_version(version, path)

    if version < CUTOUT_SCHEMA_VERSION:
        warnings.warn(
            outdated_cutout_message(version, path),
            OutdatedCutoutWarning,
            stacklevel=3,
        )
