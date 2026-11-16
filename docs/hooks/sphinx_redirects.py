# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""Redirect URLs of the former Sphinx documentation to the MkDocs pages."""

from __future__ import annotations

import posixpath
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.files import Files

NOTEBOOKS = [
    "create_cutout",
    "comparing_era5_sources_cds_vs_edh",
    "large_cutouts_with_era5_edh",
    "create_cutout_SARAH",
    "historic-comparison-germany",
    "building_stock_weather_aggregation",
    "landuse-availability",
    "using_gebco_heightmap",
    "plotting_with_atlite",
    "logfiles_and_messages",
    "solarpv_tracking_options",
    "working-with-csp",
    "glofas_hydro_inflow",
]

REDIRECTS = {
    "introduction.html": "home/introduction.md",
    "installation.html": "home/installation.md",
    "conventions.html": "home/conventions.md",
    "migrating_cutouts.html": "home/migrating-cutouts.md",
    "ref_api.html": "api/cutout.md",
    "release_notes.html": "release-notes.md",
    "contributing.html": "contributing/contributing.md",
    "support.html": "home/support.md",
    "license.html": "home/license.md",
    "examples/more_examples.html": "examples/examples.md",
} | {f"examples/{nb}.html": f"examples/{nb}.ipynb" for nb in NOTEBOOKS}

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Redirecting...</title>
<link rel="canonical" href="{url}">
<meta http-equiv="refresh" content="0; url={url}">
</head>
<body>This page has moved to <a href="{url}">{url}</a>.</body>
</html>
"""


_files: dict[str, Files] = {}


def on_files(files: Files, config: MkDocsConfig) -> Files:
    """
    Keep the documentation files to resolve redirect targets after the build.

    Returns
    -------
    Files
        The unchanged documentation files.
    """
    _files["docs"] = files
    return files


def on_post_build(config: MkDocsConfig) -> None:
    """
    Write an HTML redirect for each former Sphinx URL.

    Raises
    ------
    FileNotFoundError
        If a redirect target is not part of the documentation.
    """
    for old, new in REDIRECTS.items():
        target = _files["docs"].get_file_from_path(new)
        if target is None:
            msg = f"Redirect target '{new}' does not exist."
            raise FileNotFoundError(msg)
        url = posixpath.relpath(target.url, posixpath.dirname(old) or ".") + "/"
        dest = Path(config.site_dir, old)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(TEMPLATE.format(url=url))
