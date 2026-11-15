# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
Draw the atlite workflow chart for the README and the documentation.

Run ``uv run docs/scripts/workflow_chart.py`` to write a light and a dark SVG
to ``docs/assets/images/``. The script embeds the documentation font Overpass
(OFL-1.1), which it downloads from jsDelivr.
"""

from __future__ import annotations

import base64
import math
import urllib.request
from pathlib import Path
from xml.sax.saxutils import escape

from shapely.geometry import Polygon, box

OUT_DIR = Path(__file__).parents[1] / "assets" / "images"
FONT_URL = "https://cdn.jsdelivr.net/npm/@fontsource/overpass@5.1.1/files/overpass-latin-{}-normal.woff2"

PALETTES = {
    "light": {
        "ink": "#0D3B4F",
        "secondary": "#2C6378",
        "accent": "#00787A",
        "accent-fill": "#00A3A3",
        "hairline": "#C5D3D9",
        "ramp": ["#E6F4F4", "#B3E0DF", "#66C2C1", "#00A3A3", "#00787A"],
    },
    "dark": {
        "ink": "#DCE7EA",
        "secondary": "#9DB8C1",
        "accent": "#4FD1C5",
        "accent-fill": "#7FE3D8",
        "hairline": "#2E4A55",
        "ramp": ["#16343C", "#1F5A5E", "#2E8C8A", "#4FD1C5", "#7FE3D8"],
    },
}

STAGES = [
    (
        "Weather data",
        ["Reanalysis and satellite", "data, global or regional"],
        ("Datasets", "ERA5, SARAH, GloFAS, GEBCO"),
    ),
    (
        "Cutout",
        ["A region and period, cut", "out as a grid of weather", "time series"],
        ("Storage", "one NetCDF file, reused"),
    ),
    (
        "Conversion",
        [
            "Models per grid cell: wind,",
            "solar PV, CSP, solar thermal,",
            "hydro, heat demand, heat",
            "pumps, line ratings",
        ],
        ("Technology data", "turbines, panels, CSP plants"),
    ),
    (
        "Aggregation",
        [
            "Sum of cells per region or",
            "bus, weighted by overlap",
            "and installed capacity",
        ],
        ("Geodata", "shapes, layout, eligibility"),
    ),
]

WIDTH, COL_W, GUTTER = 880, 204, 21
HEADER_Y, DESC_Y, LINE_H = 22, 50, 20
BAND_Y, BAND_H = 132, 124
MID_Y = BAND_Y + BAND_H / 2
FOOT_Y = BAND_Y + BAND_H + 40
HEIGHT = FOOT_Y + 30
CELL_W, CELL_H, NX, NY = 16, 14, 5, 4
FONT = "Overpass, 'Segoe UI', 'Helvetica Neue', Helvetica, Arial, sans-serif"


def col(i: int) -> float:
    """Return the left edge of column ``i``."""
    return i * (COL_W + GUTTER)


def text(
    x: float, y: float, s: str, cls: str, anchor: str = "start", extra: str = ""
) -> str:
    """Return an SVG text element."""
    return f'<text x="{x:g}" y="{y:g}" class="{cls}" text-anchor="{anchor}"{extra}>{escape(s)}</text>'


def line(x1: float, y1: float, x2: float, y2: float, cls: str, extra: str = "") -> str:
    """Return an SVG line element."""
    return (
        f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" class="{cls}"{extra}/>'
    )


def points(coords: list[tuple[float, float]]) -> str:
    """Return coordinates as an SVG points attribute."""
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)


def ramp_class(value: float) -> str:
    """Return the ramp class for a value between 0 and 1."""
    return f"r{min(4, int(value * 5))}"


def cells(x0: float, y0: float, values: list[list[float]]) -> str:
    """Return a grid of cells, shaded by values between 0 and 1."""
    parts = []
    for j, row in enumerate(values):
        for i, v in enumerate(row):
            fill = ramp_class(v) if v > 0.02 else "empty"
            parts.append(
                f'<rect x="{x0 + i * CELL_W:g}" y="{y0 + j * CELL_H:g}" width="{CELL_W}" height="{CELL_H}" class="cell {fill}"/>'
            )
    return "".join(parts)


def weather_field() -> list[list[float]]:
    """Return a smooth field of values between 0 and 1, like wind speeds."""
    return [
        [
            0.5 + 0.45 * math.sin(0.9 * i - 0.7 * j + 0.4) * math.cos(0.5 * j - 0.3 * i)
            for i in range(NX)
        ]
        for j in range(NY)
    ]


def weather_map(x: float) -> tuple[str, tuple[float, float, float, float]]:
    """Return a map with dataset footprints and the cut-out box."""
    mx, my, mw, mh = x + 0.5, BAND_Y + 8.5, 184, 108
    parts = [
        f'<clipPath id="map"><rect x="{mx}" y="{my}" width="{mw}" height="{mh}"/></clipPath>'
    ]
    parts += [
        line(mx + k * mw / 8, my, mx + k * mw / 8, my + mh, "hair") for k in range(1, 8)
    ]
    parts += [
        line(mx, my + k * mh / 4, mx + mw, my + k * mh / 4, "hair") for k in range(1, 4)
    ]
    parts.append(
        f'<circle cx="{mx + 64}" cy="{my + 70}" r="52" class="disk" clip-path="url(#map)"/>'
    )
    parts.append(f'<rect x="{mx}" y="{my}" width="{mw}" height="{mh}" class="frame"/>')
    parts.append(text(mx + mw - 8, my + mh - 10, "ERA5", "label", "end"))
    parts.append(text(mx + 64, my + 96, "SARAH", "label", "middle"))
    cut = (mx + 98, my + 14, 32, 24)
    parts.append(
        f'<rect x="{cut[0]}" y="{cut[1]}" width="{cut[2]}" height="{cut[3]}" class="cut"/>'
    )
    return "".join(parts), cut


def cube(x: float) -> tuple[str, tuple[float, float]]:
    """Return a cube of grid cells over time."""
    w, h, dx, dy = NX * CELL_W, NY * CELL_H, 34, -26
    fx, fy = x + 40, MID_Y - h / 2 + 12
    top = [(fx, fy), (fx + dx, fy + dy), (fx + w + dx, fy + dy), (fx + w, fy)]
    side = [
        (fx + w, fy),
        (fx + w + dx, fy + dy),
        (fx + w + dx, fy + h + dy),
        (fx + w, fy + h),
    ]
    parts = [
        f'<polygon points="{points(top)}" class="face"/>',
        f'<polygon points="{points(side)}" class="face"/>',
    ]
    for k in range(1, 4):
        ox, oy = dx * k / 4, dy * k / 4
        parts.append(line(fx + ox, fy + oy, fx + w + ox, fy + oy, "hair"))
        parts.append(line(fx + w + ox, fy + oy, fx + w + ox, fy + h + oy, "hair"))
    parts.append(cells(fx, fy, weather_field()))
    parts.append(f'<rect x="{fx}" y="{fy}" width="{w}" height="{h}" class="frame"/>')
    for poly in (top, side):
        parts.append(f'<polygon points="{points(poly)}" class="frame"/>')
    tx, ty = fx + w + dx / 2 + 10, fy + h + dy / 2 + 8
    parts.append(
        text(
            tx,
            ty,
            "time",
            "label",
            extra=f' transform="rotate({math.degrees(math.atan2(dy, dx)):.1f} {tx:g} {ty:g})"',
        )
    )
    return "".join(parts), (fx, fy)


def power_curve(x: float) -> tuple[str, float, float]:
    """Return a wind turbine power curve with labelled axes."""
    px, w, h = x + 56, 132, 72
    py = MID_Y + h / 2 - 4

    def power(v: float) -> float:
        if v < 0.14 or v > 0.86:
            return 0.0
        return min(1.0, ((v - 0.14) / 0.42) ** 3)

    path = " ".join(
        f"{'M' if k == 0 else 'L'}{px + w * v:.1f},{py - (h - 8) * power(v):.1f}"
        for k, v in enumerate(i / 240 for i in range(241))
    )
    parts = [
        f'<path d="M{px},{py - h} V{py} H{px + w}" class="axis"/>',
        f'<path d="{path}" class="curve"/>',
        text(px + w / 2, py + 20, "wind speed", "label", "middle"),
        text(
            px - 10,
            py - h / 2,
            "power",
            "label",
            "middle",
            f' transform="rotate(-90 {px - 10:g} {py - h / 2:g})"',
        ),
    ]
    return "".join(parts), px - 30, px + w


def region_overlap(
    gx: float, gy: float
) -> tuple[list[list[float]], list[tuple[float, float]]]:
    """Return the overlap of each grid cell with a region, and the region outline."""
    outline = []
    for k in range(60):
        a = 2 * math.pi * k / 60
        r = (
            1
            + 0.13 * math.sin(2 * a + 0.6)
            + 0.08 * math.sin(5 * a + 1.9)
            + 0.06 * math.sin(7 * a + 0.3)
            + 0.035 * math.sin(13 * a)
        )
        outline.append((2.55 + 1.75 * r * math.cos(a), 2.0 + 1.45 * r * math.sin(a)))
    region = Polygon(outline)
    overlap = [
        [region.intersection(box(i, j, i + 1, j + 1)).area for i in range(NX)]
        for j in range(NY)
    ]
    return overlap, [(gx + a * CELL_W, gy + b * CELL_H) for a, b in outline]


def aggregation(x: float) -> tuple[str, float]:
    """Return grid cells weighted by a region, and the resulting time series."""
    gx, gy = x + 28, MID_Y - NY * CELL_H / 2
    overlap, outline = region_overlap(gx, gy)
    parts = [
        cells(gx, gy, overlap),
        f'<polygon points="{points(outline)}" class="region"/>',
    ]
    sx, sw, sh = x + 136, 68, 56
    sy = MID_Y + sh / 2
    parts.append(
        line(
            gx + NX * CELL_W + 6,
            MID_Y,
            sx - 8,
            MID_Y,
            "flow",
            ' marker-end="url(#head)"',
        )
    )
    for k, cls in enumerate(("series", "series-2")):
        values = [
            0.5
            + 0.22 * math.sin(0.19 * t + 2.2 * k)
            + 0.09 * math.sin(0.53 * t + 1.3 + 2 * k)
            + 0.04 * math.sin(1.7 * t + k)
            for t in range(49)
        ]
        path = " ".join(
            f"{'M' if t == 0 else 'L'}{sx + sw * t / 48:.1f},{sy - sh * v:.1f}"
            for t, v in enumerate(values)
        )
        parts.append(f'<path d="{path}" class="{cls}"/>')
    parts.append(f'<path d="M{sx},{sy - sh} V{sy} H{sx + sw}" class="axis"/>')
    parts.append(text(sx + sw / 2, sy + 20, "per region", "label", "middle"))
    return "".join(parts), gx


def stage_text(i: int) -> str:
    """Return the header, description and footer of stage ``i``."""
    name, desc, (foot_label, foot_value) = STAGES[i]
    x = col(i)
    parts = [
        text(x, HEADER_Y, str(i + 1), "num"),
        text(x + 18, HEADER_Y, name, "stage"),
    ]
    parts += [text(x, DESC_Y + k * LINE_H, s, "desc") for k, s in enumerate(desc)]
    parts += [
        text(x, FOOT_Y, foot_label, "foot-label"),
        text(x, FOOT_Y + 19, foot_value, "desc"),
    ]
    return "".join(parts)


def body() -> str:
    """Return the drawing of all four stages and the flows between them."""
    map_svg, (cx, cy, cw, ch) = weather_map(col(0))
    cube_svg, (fx, fy) = cube(col(1))
    curve_svg, curve_left, curve_right = power_curve(col(2))
    agg_svg, grid_left = aggregation(col(3))
    cube_right = fx + NX * CELL_W + 34
    parts = [map_svg, cube_svg, curve_svg, agg_svg]
    parts.append(line(cx + cw, cy, fx, fy, "projection"))
    parts.append(line(cx + cw, cy + ch, fx, fy + NY * CELL_H, "projection"))
    parts.append(
        line(
            cube_right + 8, MID_Y, curve_left, MID_Y, "flow", ' marker-end="url(#head)"'
        )
    )
    parts.append(
        line(
            curve_right + 10,
            MID_Y,
            grid_left - 8,
            MID_Y,
            "flow",
            ' marker-end="url(#head)"',
        )
    )
    for i in (2, 3):
        jx = col(i) + 2.5
        parts.append(
            line(jx, FOOT_Y - 20, jx, MID_Y + 5, "flow", ' marker-end="url(#head)"')
        )
    parts += [stage_text(i) for i in range(len(STAGES))]
    return "".join(parts)


def style(palette: dict, fonts: dict[int, str]) -> str:
    """Return the CSS that maps classes to the palette."""
    c = palette
    faces = "".join(
        f"@font-face{{font-family:Overpass;font-weight:{w};src:url(data:font/woff2;base64,{data}) format('woff2')}}"
        for w, data in fonts.items()
    )
    ramp = "".join(f".r{k}{{fill:{v}}}" for k, v in enumerate(c["ramp"]))
    return (
        f"{faces}"
        f"text{{font-family:{FONT};fill:{c['ink']}}}"
        f".num{{fill:{c['accent']};font-size:18px;font-weight:600}}"
        f".stage{{font-size:18px;font-weight:600}}"
        f".desc{{fill:{c['secondary']};font-size:15px}}"
        f".foot-label{{font-size:15px;font-weight:600}}"
        f".label{{fill:{c['secondary']};font-size:14px}}"
        f".hair{{stroke:{c['hairline']};stroke-width:1}}"
        f".frame{{fill:none;stroke:{c['ink']};stroke-width:1.25;stroke-linejoin:miter}}"
        f".face{{fill:{c['ramp'][0]}}}"
        f".cell{{stroke:{c['hairline']};stroke-width:1}}.empty{{fill:none}}"
        f".disk{{fill:{c['accent-fill']};fill-opacity:0.18;stroke:{c['accent']};stroke-width:1.25;stroke-dasharray:4 3}}"
        f".cut{{fill:{c['accent-fill']};fill-opacity:0.35;stroke:{c['ink']};stroke-width:1.5}}"
        f".projection{{stroke:{c['secondary']};stroke-width:1;stroke-dasharray:3 3}}"
        f".flow{{stroke:{c['ink']};stroke-width:1.25}}"
        f".axis{{fill:none;stroke:{c['ink']};stroke-width:1.25}}"
        f".curve,.series{{fill:none;stroke:{c['accent']};stroke-width:2;stroke-linejoin:round}}"
        f".series-2{{fill:none;stroke:{c['secondary']};stroke-width:1.5;stroke-linejoin:round}}"
        f".region{{fill:none;stroke:{c['ink']};stroke-width:1.5;stroke-linejoin:round}}"
        f".head{{fill:none;stroke:{c['ink']};stroke-width:1.25}}"
        f"{ramp}"
    )


def svg(palette: dict, fonts: dict[int, str]) -> str:
    """Return the full chart as an SVG document."""
    marker = (
        '<marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" '
        'markerUnits="userSpaceOnUse" orient="auto"><path d="M1,1 L9,5 L1,9" class="head"/></marker>'
    )
    desc = (
        "1. Weather data: reanalysis and satellite data, global or regional, such as ERA5, SARAH, GloFAS and GEBCO. "
        "2. Cutout: a region and period, cut out as a grid of weather time series and stored in one NetCDF file. "
        "3. Conversion: models turn the weather in each grid cell into wind, solar, hydro, heat and line rating time series, "
        "using technology data such as turbine power curves. "
        "4. Aggregation: grid cells are summed per region or bus, weighted by overlap and installed capacity."
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT:g}" width="{WIDTH}" height="{HEIGHT:g}" '
        'role="img" aria-labelledby="title desc">'
        '<title id="title">How atlite works: weather data, cutout, conversion, aggregation</title>'
        f'<desc id="desc">{escape(desc)}</desc>'
        f"<style>{style(palette, fonts)}</style><defs>{marker}</defs>{body()}</svg>\n"
    )


def fetch_fonts() -> dict[int, str]:
    """Download Overpass and return it base64-encoded per weight."""
    fonts = {}
    for weight in (400, 600):
        with urllib.request.urlopen(FONT_URL.format(weight)) as response:
            fonts[weight] = base64.b64encode(response.read()).decode()
    return fonts


if __name__ == "__main__":
    fonts = fetch_fonts()
    for name, palette in PALETTES.items():
        suffix = "" if name == "light" else "-dark"
        (OUT_DIR / f"workflow_chart{suffix}.svg").write_text(svg(palette, fonts))
