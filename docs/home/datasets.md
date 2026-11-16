<!--
SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>

SPDX-License-Identifier: CC-BY-4.0
-->

# Datasets

atlite creates cutouts from one or more dataset modules. You select them with
the `module` argument of [`atlite.Cutout`][atlite.Cutout].

| Module | Dataset | Content | Access |
| --- | --- | --- | --- |
| `era5-edh` | ERA5 via Earth Data Hub | All weather variables | Download, account needed |
| `era5` | ERA5 via Climate Data Store | All weather variables | Download, account needed |
| `sarah` | SARAH-2 | Solar radiation | Manual download |
| `glofas` | GloFAS | River discharge | Download, CDS account needed |
| `gebco` | GEBCO | Terrain height | Manual download |

## ERA5

The default dataset is ERA5, the global reanalysis of the European Centre for
Medium-Range Weather Forecasts (ECMWF). It has an hourly resolution on a grid of
about 30 km x 30 km. ERA5 is free to use and needs little setup in
comparison to other datasets. See the [installation instructions](installation.md#era5-credentials)
for how to obtain credentials.

atlite offers two sources for ERA5:

- **Earth Data Hub (EDH)** hosts
  [a mirror of ERA5](https://earthdatahub.destine.eu/collections/era5/datasets/reanalysis-era5-single-levels)
  that allows immediate downloads. It is the most convenient source for most
  users, unless you need the data at an unusual resolution. The
  [comparison of ERA5 sources](../examples/comparing_era5_sources_cds_vs_edh.ipynb)
  shows how to use the EDH mirror and when it may not be the best choice.
  The [large cutouts example](../examples/large_cutouts_with_era5_edh.ipynb)
  shows how to build cutouts for large regions and long periods.
- **Climate Data Store (CDS)** of the Copernicus Climate Change Service is the
  original source of ERA5. CDS requests can wait for hours in a processing queue
  on the server. We therefore only recommend CDS if you need it. The
  [cutout creation example](../examples/create_cutout.ipynb) shows how to create
  an ERA5 cutout.

## SARAH-2

The satellite-based [SARAH-2 dataset](https://wui.cmsaf.eu/safira/action/viewDoiDetails?acronym=SARAH_V002_01)
of the EUMETSAT Satellite Application Facility on Climate Monitoring (CM SAF)
contains solar radiation variables at higher resolution than ERA5. It
complements ERA5, which provides the other variables. You must download the files
manually and pass their directory as `sarah_dir`. See the
[SARAH-2 example](../examples/create_cutout_SARAH.ipynb).

## GloFAS

The [GloFAS dataset](https://ewds.climate.copernicus.eu/datasets/cems-glofas-historical)
of ECMWF contains daily river discharge. It complements ERA5 runoff data for hydro
power. GloFAS uses the same API key as the CDS; atlite sets up the API endpoint
automatically. atlite uses GloFAS if `cutout.module` contains `"glofas"`. See the
[GloFAS hydro inflow example](../examples/glofas_hydro_inflow.ipynb).

## GEBCO

The [GEBCO](https://www.gebco.net/) bathymetry and terrain model gives a more
precise height above sea level than the geopotential height of ERA5, especially
at sea. You must download the file manually and pass its location as
`gebco_path`. See the [GEBCO example](../examples/using_gebco_heightmap.ipynb).

## Other datasets

Earlier versions of atlite supported other datasets, and they may become
usable again in the future:

- the *NCEP Climate Forecast System* dataset,
- the *EURO-CORDEX Climate Change Projection* dataset,
- weather forecasts from climate models.

Their support is currently on hold because developer time is limited.
If you need these or other data sources, please
[open an issue](https://github.com/PyPSA/atlite/issues) or, even better, a pull request.
