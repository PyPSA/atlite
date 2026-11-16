# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Add dark-theme variants of the figures in the example notebooks.

The notebooks store matplotlib figures as PNG on a white background. The light
theme shows them unchanged. For the dark theme, this hook removes the white:

- White becomes transparent.
- Thin shapes (text, ticks, lines, grids), anti-aliased fringes and near-white
  pixels are un-mixed from white, so that they keep their hue on any
  background. Neutral thin shapes next to white take the theme's text colour.
- Everything else (maps, heatmaps, bars, colorbars, line cores, text on data)
  keeps its exact original colour.

Material shows each variant only in its theme.
"""

from __future__ import annotations

import base64
import hashlib
import io
import re
from functools import cache, partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from mkdocs.utils import get_relative_url
from PIL import Image
from scipy.ndimage import binary_dilation, binary_opening, gaussian_filter, white_tophat

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.pages import Page

INK = np.array([226, 228, 233]) / 255
INK_OPACITY = 0.82
INK_CONTEXT = 0.15
CHROMA_TOLERANCE = 0.15
WHITE_ALPHA = 0.02
HAZE_ALPHA = 0.25
FRINGE_ALPHA = 0.5
CORE_ALPHA = 0.8
POCKET_CONTRAST = 0.2
RIM_WIDTH = 2
STROKE_WIDTH = 4
SIGMA = 2

_y, _x = np.ogrid[-STROKE_WIDTH : STROKE_WIDTH + 1, -STROKE_WIDTH : STROKE_WIDTH + 1]
DISK = _x**2 + _y**2 <= STROKE_WIDTH**2
SQUARE = np.ones((5, 5), dtype=bool)

FIGURES = "assets/images/notebook-figures"
IMG = re.compile(r'<img (?P<attrs>[^>]*?)src="data:image/png;base64,(?P<data>[^"]+)"')


@cache
def to_dark(png: bytes) -> bytes:
    """
    Convert a figure on white background to a transparent dark-theme figure.

    Returns
    -------
    bytes
        PNG with transparent background and light neutral strokes.
    """
    rgba = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"), dtype=float) / 255
    rgb = rgba[..., :3] * rgba[..., 3:] + 1 - rgba[..., 3:]
    alpha = 1 - rgb.min(axis=-1)
    color = 1 - (1 - rgb) / np.maximum(alpha, 1e-6)[..., None]
    chroma = color.max(axis=-1) - color.min(axis=-1)
    neutral = np.clip(1 - chroma / CHROMA_TOLERANCE, 0, 1)
    white = alpha < WHITE_ALPHA

    thin = ~binary_opening(~white, DISK)
    rim = binary_dilation(white, iterations=RIM_WIDTH)
    haze = (neutral > 0.5) & (alpha < HAZE_ALPHA)
    near_ink = binary_dilation((neutral > 0.5) & (alpha > 0.5), SQUARE)
    bright = white_tophat(1 - alpha, footprint=SQUARE) > POCKET_CONTRAST
    pocket = bright & (alpha < FRINGE_ALPHA) & ~near_ink
    unmix = thin | rim | haze | pocket

    context = np.clip(gaussian_filter(white.astype(float), SIGMA) / INK_CONTEXT, 0, 1)
    ink = neutral * thin * context
    core = np.clip((alpha - FRINGE_ALPHA) / (CORE_ALPHA - FRINGE_ALPHA), 0, 1)
    solid = np.where(unmix, core * (1 - ink), 1)[..., None]

    color = color * (1 - ink[..., None]) + INK * ink[..., None]
    alpha = alpha * (1 - ink * (1 - INK_OPACITY))
    transparent = np.dstack([color, alpha])
    opaque = np.dstack([rgb, np.ones_like(alpha)])
    out = opaque * solid + transparent * (1 - solid)

    image = Image.fromarray((out * 255).round().astype(np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, "png", optimize=True)
    return buffer.getvalue()


def _theme_variants(match: re.Match[str], page: Page, site_dir: str) -> str:
    attrs, data = match["attrs"], match["data"]
    png = to_dark(base64.b64decode(data))
    path = f"{FIGURES}/{hashlib.sha256(png).hexdigest()[:16]}.png"
    file = Path(site_dir, path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_bytes(png)
    url = get_relative_url(path, page.url)
    return (
        f'<img {attrs}src="data:image/png;base64,{data}#only-light">'
        f'<img {attrs}loading="lazy" src="{url}#only-dark"'
    )


def on_post_page(output: str, page: Page, config: MkDocsConfig) -> str:
    """
    Add a dark-theme variant to each PNG figure of a notebook page.

    The dark variants are written as separate files and load lazily, so that
    readers of the light theme do not download them.

    Returns
    -------
    str
        The page HTML with light and dark figure variants.
    """
    if not page.file.src_uri.endswith(".ipynb"):
        return output
    replace = partial(_theme_variants, page=page, site_dir=config.site_dir)
    return IMG.sub(replace, output)
