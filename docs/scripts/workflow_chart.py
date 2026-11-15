# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Draw the atlite workflow chart for the README and the documentation.

Run ``python docs/scripts/workflow_chart.py`` to write a light and a dark SVG
to ``docs/assets/images/``.
"""

from __future__ import annotations

import math
from pathlib import Path
from xml.sax.saxutils import escape

OUT_DIR = Path(__file__).parents[1] / "assets" / "images"

PALETTES = {
    "light": {
        "card": "#F7FAFB",
        "card-stroke": "#D6E2E7",
        "inset": "#EAF2F4",
        "title": "#0D3B4F",
        "text": "#2E4752",
        "muted": "#647A84",
        "line": "#B9CCD3",
        "arrow": "#00A3A3",
        "badge": "#0D3B4F",
        "badge-text": "#FFFFFF",
        "wind": "#14A39E",
        "solar": "#E39B1F",
        "temperature": "#D6533A",
        "water": "#3474D0",
        "height": "#8C9CA4",
    },
    "dark": {
        "card": "#12262E",
        "card-stroke": "#28444F",
        "inset": "#1A333D",
        "title": "#E8F3F5",
        "text": "#C3D4D9",
        "muted": "#89A2AB",
        "line": "#3A5966",
        "arrow": "#4FD1C5",
        "badge": "#4FD1C5",
        "badge-text": "#0D3B4F",
        "wind": "#3CC9C2",
        "solar": "#F2B655",
        "temperature": "#EE7B61",
        "water": "#6EA6F2",
        "height": "#A2B1B8",
    },
}

VARIABLES = {
    "wind": "Wind",
    "solar": "Solar radiation",
    "temperature": "Temperature",
    "water": "Runoff, discharge",
    "height": "Terrain height",
}

DATASETS = [
    (
        "ERA5",
        "all variables, hourly",
        ["wind", "solar", "temperature", "water", "height"],
    ),
    ("SARAH-2", "solar radiation", ["solar"]),
    ("GloFAS", "river discharge", ["water"]),
    ("GEBCO", "terrain height", ["height"]),
]

CONVERSIONS = [
    ("Wind power", ["wind"]),
    ("Solar PV", ["solar", "temperature"]),
    ("Solar thermal", ["solar", "temperature"]),
    ("Concentrated solar", ["solar"]),
    ("Hydro inflow", ["water"]),
    ("Heat & cooling demand", ["temperature"]),
    ("Heat pump COP", ["temperature"]),
    ("Line rating", ["wind", "solar", "temperature"]),
]

AGGREGATIONS = [
    ("SPACE", "regions or buses"),
    ("WEIGHTS", "capacity layout"),
    ("TIME", "series, sum or mean"),
    ("UNIT", "MW or per unit"),
]

CARDS = [
    ("Weather data", "Global datasets", "Access", "API download or local files"),
    ("Cutout", "Your region and period", "Storage", "one NetCDF file, reusable"),
    (
        "Conversion",
        "Models per grid cell",
        "Technology data",
        "turbines, solar panels, CSP",
    ),
    ("Aggregation", "From cells to regions", "Geodata", "shapes, layout, eligibility"),
]

WIDTH, MARGIN, CARD_W, GAP = 1044, 12, 222, 44
CARD_Y, CARD_H, PAD = 12, 446, 18
HEIGHT = CARD_Y + CARD_H + MARGIN
ART_Y = CARD_Y + 137
LIST_Y = CARD_Y + 212
CUTOUT_NOTE = ("Each grid cell holds one", "time series per variable.")
FONT = "'Inter', 'Segoe UI', 'Helvetica Neue', Helvetica, Arial, 'Liberation Sans', sans-serif"


def text(x: float, y: float, s: str, cls: str, anchor: str = "start") -> str:
    """Return an SVG text element."""
    return f'<text x="{x:g}" y="{y:g}" class="{cls}" text-anchor="{anchor}">{escape(s)}</text>'


def dots(x_right: float, y: float, variables: list[str]) -> str:
    """Return right-aligned coloured dots, one per weather variable."""
    return "".join(
        f'<circle cx="{x_right - 4.5 - 12 * i:g}" cy="{y:g}" r="4.5" class="f-{v}"/>'
        for i, v in enumerate(reversed(variables))
    )


def globe(cx: float, cy: float, r: float = 40) -> str:
    """Return a globe with meridians and parallels."""
    parts = [f'<circle cx="{cx}" cy="{cy}" r="{r}" class="art-fill"/>']
    for rx in (r / 3, 2 * r / 3):
        parts.append(
            f'<ellipse cx="{cx}" cy="{cy}" rx="{rx:g}" ry="{r}" class="art-line"/>'
        )
    parts.append(
        f'<line x1="{cx}" y1="{cy - r}" x2="{cx}" y2="{cy + r}" class="art-line"/>'
    )
    for dy in (-2 * r / 3, -r / 3, 0, r / 3, 2 * r / 3):
        half = math.sqrt(r**2 - dy**2)
        parts.append(
            f'<line x1="{cx - half:.1f}" y1="{cy + dy:.1f}" x2="{cx + half:.1f}" y2="{cy + dy:.1f}" class="art-line"/>'
        )
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" class="art-edge"/>')
    return "".join(parts)


def skewed(x0: float, y0: float, u: float, v: float) -> tuple[float, float]:
    """Map unit square coordinates onto a skewed grid layer."""
    w, h, skew = 92, 46, 26
    return x0 + skew * (1 - v) + w * u, y0 + h * v


def grid_layers(cx: float, cy: float) -> str:
    """Return three stacked grid layers that stand for time steps."""
    w, h, skew, nx, ny = 92, 46, 26, 5, 3
    shades = [
        0.25,
        0.45,
        0.7,
        0.9,
        0.6,
        0.35,
        0.55,
        0.8,
        1.0,
        0.75,
        0.2,
        0.4,
        0.65,
        0.85,
        0.5,
    ]
    parts = []
    for layer, (dx, dy) in enumerate([(16, -20), (8, -10), (0, 0)]):
        x0, y0 = cx - (w + skew) / 2 - 8 + dx, cy - h / 2 + 10 + dy

        def pt(
            u: float, v: float, x0: float = x0, y0: float = y0
        ) -> tuple[float, float]:
            return skewed(x0, y0, u, v)

        if layer == 2:
            for i in range(nx):
                for j in range(ny):
                    corners = [
                        pt(i / nx, j / ny),
                        pt((i + 1) / nx, j / ny),
                        pt((i + 1) / nx, (j + 1) / ny),
                        pt(i / nx, (j + 1) / ny),
                    ]
                    poly = " ".join(f"{a:.1f},{b:.1f}" for a, b in corners)
                    parts.append(
                        f'<polygon points="{poly}" class="f-wind" fill-opacity="{shades[i + nx * j]:g}"/>'
                    )
        outline = " ".join(
            f"{a:.1f},{b:.1f}" for a, b in [pt(0, 0), pt(1, 0), pt(1, 1), pt(0, 1)]
        )
        cls = "art-layer" if layer == 2 else "art-layer-back"
        parts.append(f'<polygon points="{outline}" class="{cls}"/>')
        for i in range(1, nx):
            (a, b), (c, d) = pt(i / nx, 0), pt(i / nx, 1)
            parts.append(
                f'<line x1="{a:.1f}" y1="{b:.1f}" x2="{c:.1f}" y2="{d:.1f}" class="art-grid"/>'
            )
        for j in range(1, ny):
            (a, b), (c, d) = pt(0, j / ny), pt(1, j / ny)
            parts.append(
                f'<line x1="{a:.1f}" y1="{b:.1f}" x2="{c:.1f}" y2="{d:.1f}" class="art-grid"/>'
            )
    ax, ay = cx + w / 2 + 14, cy + 22
    parts.append(
        f'<line x1="{ax}" y1="{ay}" x2="{ax + 14}" y2="{ay - 18}" class="art-axis" marker-end="url(#tip)"/>'
    )
    parts.append(text(ax + 8, ay + 14, "time", "tiny", "middle"))
    return "".join(parts)


def power_curve(cx: float, cy: float) -> str:
    """Return a wind turbine power curve with axes."""
    w, h = 124, 62
    x0, y0 = cx - w / 2 + 6, cy + h / 2 - 2

    def p(v: float) -> float:
        if v < 0.15 or v > 0.88:
            return 0.0
        return min(1.0, ((v - 0.15) / 0.4) ** 3)

    steps = [i / 200 for i in range(201)]
    path = " ".join(
        f"{'M' if i == 0 else 'L'}{x0 + w * v:.1f},{y0 - (h - 6) * p(v):.1f}"
        for i, v in enumerate(steps)
    )
    return (
        f'<line x1="{x0}" y1="{y0}" x2="{x0 + w + 6}" y2="{y0}" class="art-axis" marker-end="url(#tip)"/>'
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 - h - 4}" class="art-axis" marker-end="url(#tip)"/>'
        f'<path d="{path}" class="curve s-wind"/>'
        + text(x0 + w / 2, y0 + 16, "wind speed", "tiny", "middle")
        + f'<text x="{x0 - 8}" y="{y0 - h / 2}" class="tiny" text-anchor="middle" transform="rotate(-90 {x0 - 8} {y0 - h / 2})">power</text>'
    )


def region_map(cx: float, cy: float) -> str:
    """Return grid cells under a region outline, shaded by their overlap."""
    size, nx, ny = 17, 7, 4
    overlap = [
        [0.0, 0.2, 0.6, 0.5, 0.1, 0.0, 0.0],
        [0.1, 0.8, 1.0, 1.0, 0.7, 0.2, 0.0],
        [0.0, 0.5, 1.0, 1.0, 1.0, 0.6, 0.0],
        [0.0, 0.0, 0.3, 0.7, 0.5, 0.1, 0.0],
    ]
    x0, y0 = cx - nx * size / 2, cy - ny * size / 2
    parts = []
    for j in range(ny):
        for i in range(nx):
            x, y = x0 + i * size, y0 + j * size
            parts.append(
                f'<rect x="{x:g}" y="{y:g}" width="{size}" height="{size}" class="art-cell"/>'
            )
            if overlap[j][i]:
                parts.append(
                    f'<rect x="{x:g}" y="{y:g}" width="{size}" height="{size}" class="f-wind" fill-opacity="{overlap[j][i]:g}"/>'
                )
    shape = [
        (1.3, 0.9),
        (2.6, 0.2),
        (4.2, 0.5),
        (5.1, 1.2),
        (5.7, 2.4),
        (4.6, 3.6),
        (3.0, 3.5),
        (2.0, 2.6),
        (0.9, 1.9),
    ]
    poly = " ".join(f"{x0 + a * size:.1f},{y0 + b * size:.1f}" for a, b in shape)
    parts.append(f'<polygon points="{poly}" class="region"/>')
    return "".join(parts)


def card(i: int) -> str:
    """Return one workflow stage as a card."""
    title, subtitle, inset_label, inset_value = CARDS[i]
    x = MARGIN + i * (CARD_W + GAP)
    cx, right = x + CARD_W / 2, x + CARD_W - PAD
    parts = [
        f'<rect x="{x}" y="{CARD_Y}" width="{CARD_W}" height="{CARD_H}" rx="14" class="card"/>',
        f'<circle cx="{x + PAD + 13}" cy="{CARD_Y + 32}" r="13" class="badge"/>',
        text(x + PAD + 13, CARD_Y + 37, str(i + 1), "badge-text", "middle"),
        text(x + PAD + 34, CARD_Y + 38, title, "title"),
        text(x + PAD, CARD_Y + 66, subtitle, "subtitle"),
        f'<line x1="{x + PAD}" y1="{CARD_Y + 82}" x2="{right}" y2="{CARD_Y + 82}" class="divider"/>',
        [globe, grid_layers, power_curve, region_map][i](cx, ART_Y),
    ]
    if i == 0:
        for k, (name, desc, variables) in enumerate(DATASETS):
            y = LIST_Y + 40 * k
            parts += [
                text(x + PAD, y, name, "item-strong"),
                text(x + PAD, y + 18, desc, "note"),
                dots(right, y - 5, variables),
            ]
    elif i == 1:
        for k, (var, label) in enumerate(VARIABLES.items()):
            y = LIST_Y + 28 * k
            parts += [
                f'<circle cx="{x + PAD + 4.5}" cy="{y - 5}" r="4.5" class="f-{var}"/>',
                text(x + PAD + 18, y, label, "item"),
            ]
        for k, line in enumerate(CUTOUT_NOTE):
            parts.append(
                text(x + PAD, LIST_Y + 28 * len(VARIABLES) - 4 + 17 * k, line, "note")
            )
    elif i == 2:
        for k, (name, variables) in enumerate(CONVERSIONS):
            y = LIST_Y + 22 * k
            parts += [text(x + PAD, y, name, "item"), dots(right, y - 5, variables)]
    else:
        for k, (label, value) in enumerate(AGGREGATIONS):
            y = LIST_Y + 38 * k
            parts += [
                text(x + PAD, y - 2, label, "label"),
                text(x + PAD, y + 17, value, "item"),
            ]
    inset_y = CARD_Y + CARD_H - 66
    parts += [
        f'<rect x="{x + PAD - 6}" y="{inset_y}" width="{CARD_W - 2 * PAD + 12}" height="50" rx="8" class="inset"/>',
        text(x + PAD + 4, inset_y + 21, inset_label, "inset-label"),
        text(x + PAD + 4, inset_y + 39, inset_value, "note"),
    ]
    return "".join(parts)


def arrow(i: int) -> str:
    """Return the arrow between card ``i`` and card ``i + 1``."""
    x1 = MARGIN + (i + 1) * CARD_W + i * GAP + 8
    x2 = x1 + GAP - 16
    y = CARD_Y + CARD_H / 2
    return f'<line x1="{x1}" y1="{y}" x2="{x2 - 4}" y2="{y}" class="flow" marker-end="url(#head)"/>'


def style(palette: dict[str, str]) -> str:
    """Return the CSS that maps classes to the palette."""
    c = palette
    variables = "".join(
        f".f-{v}{{fill:{c[v]}}}.s-{v}{{stroke:{c[v]}}}" for v in VARIABLES
    )
    return (
        f"text{{font-family:{FONT}}}"
        f".card{{fill:{c['card']};stroke:{c['card-stroke']};stroke-width:1.2}}"
        f".inset{{fill:{c['inset']}}}"
        f".badge{{fill:{c['badge']}}}"
        f".badge-text{{fill:{c['badge-text']};font-size:14px;font-weight:700}}"
        f".title{{fill:{c['title']};font-size:19px;font-weight:700}}"
        f".subtitle{{fill:{c['muted']};font-size:14.5px}}"
        f".item{{fill:{c['text']};font-size:15px}}"
        f".item-strong{{fill:{c['title']};font-size:15.5px;font-weight:700}}"
        f".note{{fill:{c['muted']};font-size:13.5px}}"
        f".label{{fill:{c['muted']};font-size:11.5px;font-weight:700;letter-spacing:1px}}"
        f".inset-label{{fill:{c['title']};font-size:13.5px;font-weight:700}}"
        f".tiny{{fill:{c['muted']};font-size:11px}}"
        f".divider{{stroke:{c['card-stroke']};stroke-width:1}}"
        f".flow{{stroke:{c['arrow']};stroke-width:3;stroke-linecap:round}}"
        f".art-fill{{fill:{c['inset']}}}"
        f".art-line,.art-grid{{fill:none;stroke:{c['line']};stroke-width:1}}"
        f".art-edge,.art-layer{{fill:none;stroke:{c['muted']};stroke-width:1.4}}"
        f".art-layer-back{{fill:{c['card']};stroke:{c['line']};stroke-width:1.2}}"
        f".art-axis{{stroke:{c['muted']};stroke-width:1.2}}"
        f".art-cell{{fill:none;stroke:{c['line']};stroke-width:0.8}}"
        f".curve{{fill:none;stroke-width:2.4;stroke-linejoin:round}}"
        f".region{{fill:none;stroke:{c['title']};stroke-width:1.8;stroke-linejoin:round}}"
        f".m-head{{fill:{c['arrow']}}}.m-tip{{fill:{c['muted']}}}"
        f"{variables}"
    )


def svg(palette: dict[str, str]) -> str:
    """Return the full chart as an SVG document."""
    defs = (
        '<marker id="head" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="4" markerHeight="4" orient="auto">'
        '<path d="M0,0 L10,5 L0,10 z" class="m-head"/></marker>'
        '<marker id="tip" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="5" markerHeight="5" orient="auto">'
        '<path d="M0,1 L9,5 L0,9 z" class="m-tip"/></marker>'
    )
    body = "".join(card(i) for i in range(len(CARDS))) + "".join(
        arrow(i) for i in range(len(CARDS) - 1)
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}" '
        'role="img" aria-labelledby="t">'
        '<title id="t">atlite workflow: weather data, cutout, conversion, aggregation</title>'
        f"<style>{style(palette)}</style><defs>{defs}</defs>{body}</svg>\n"
    )


if __name__ == "__main__":
    for name, palette in PALETTES.items():
        suffix = "" if name == "light" else "-dark"
        (OUT_DIR / f"workflow_chart{suffix}.svg").write_text(svg(palette))
