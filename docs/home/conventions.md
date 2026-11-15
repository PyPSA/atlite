<!--
SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>

SPDX-License-Identifier: CC-BY-4.0
-->

# Conventions

atlite applies the following conventions to geospatial and temporal data.

## Grid coordinates

Following the `xarray` conventions, grid coordinates are sorted such that both
`x` and `y` are ascending.

The coordinates are points in the **centre** of the grid cells.
[`Cutout.grid`][atlite.Cutout.grid] returns the grid as a GeoPandas
`GeoDataFrame` with coordinates and geometries. The coordinates are the
centroids of the geometries, the grid cells. For example, this cutout

``` python
cutout = atlite.Cutout(
    "example", module="era5", x=slice(5, 10), y=slice(30, 35), time="2013"
)
```

has centroids starting at 5° for `x` (longitude) and 30° for `y` (latitude).
With the default resolution of 0.25° x 0.25° per grid cell, the cutout
therefore covers longitudes from 4.875° to 10.125° and latitudes from
29.875° to 35.125°. [`Cutout.extent`][atlite.Cutout.extent] returns these bounds:

``` python
>>> cutout.extent
array([ 4.875, 10.125, 29.875, 35.125])
```

## Time points

Following the ERA5 convention, the time index of time-dependent data refers to
the **end** of the period over which the data is averaged. For a time resolution
of 1 hour, the time index 12:00 refers to the average from 11:00 to 12:00.

Other datasets do not necessarily follow this convention. SARAH data, for
example, are instantaneous values: the time index 12:00 refers to the value at
12:00. atlite takes such differences into account to align datasets before it
merges them.
