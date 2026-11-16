<!--
SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>

SPDX-License-Identifier: CC-BY-4.0
-->

# Examples

The examples are Jupyter notebooks. Each page has a download button at the
top; all notebooks are also in the
[`docs/examples`](https://github.com/PyPSA/atlite/tree/master/docs/examples)
folder of the repository.

## Creating cutouts

- [Creating a cutout with ERA5](create_cutout.ipynb): the starting point for most users.
- [Comparing ERA5 sources](comparing_era5_sources_cds_vs_edh.ipynb): Climate Data Store versus Earth Data Hub.
- [Large cutouts with Earth Data Hub](large_cutouts_with_era5_edh.ipynb): cutouts for large regions and long periods.
- [Creating a cutout with SARAH-2](create_cutout_SARAH.ipynb): satellite-based solar radiation.

## Conversion

- [Historic comparison of PV and wind](historic-comparison-germany.ipynb): atlite feed-in compared with historical statistics for Germany in 2013.
- [Solar PV tracking options](solarpv_tracking_options.ipynb): fixed, single-axis and dual-axis tracking.
- [Concentrated solar power](working-with-csp.ipynb): parabolic troughs and solar towers.
- [GloFAS hydro inflow](glofas_hydro_inflow.ipynb): river discharge for hydroelectric plants.
- [Building stock weather aggregation](building_stock_weather_aggregation.ipynb): population-weighted weather for heat demand modelling.

## Geospatial analysis

- [Land use availability](landuse-availability.ipynb): eligible area per grid cell with land use exclusions.
- [GEBCO height map](using_gebco_heightmap.ipynb): precise terrain heights.
- [Plotting with atlite](plotting_with_atlite.ipynb): the figures of the [introduction](../home/introduction.md).

## Utilities

- [Log files and messages](logfiles_and_messages.ipynb): control what atlite reports.

## Contribute an example

If you miss an example for an aspect of atlite, or if you have written one
yourself, we are happy to hear from you. Please open an
[issue](https://github.com/PyPSA/atlite/issues) or a
[pull request](https://github.com/PyPSA/atlite/pulls), and see the
[contributing guidelines](../contributing/contributing.md#contributing-examples).
