..
  SPDX-FileCopyrightText: 2016-2019 The Atlite Authors

  SPDX-License-Identifier: CC-BY-4.0

#############
Release Notes
#############


.. Version 0.2.1 
.. =========
.. Untoggle this and add first release note for the next version


Version 0.2
===============

**Major changes**


* Atlite now **requires Python 3.6 or higher**.
* We changed the Atlite backend for storing cutout data.
  Existing cutouts either need to be migrated with the
  appropriate functions or (what we recommended) recreated.
* The backend change also includes some changes to the API.
  Most notably:
  
  - The `xarray` for cutouts is now exposed as `Cutout.data`
  - The `Cutout.meta` attribute was deprecated in favour of
    `Cutout.data.attrs`
  - `xarray` and `dask` can now handle some data caching
    automatically.
    If you wish to preload some data before your calculation,
    you can now use `Cutout.data.load()` to load all of the
    cutouts data into memory.  
    *(Warning: Requires a large enough memory.)*
  - The `Cutout` class has a new property `grid`, a GeoPandas DataFrame 
    which combines and deprecates `grid_cells()` and `grid_coordinates()`
* The order of coordinates (indices) for `Cutouts` changed: `x` and `y` (e.g. longitude and latitude) are now both ascending (before: `x` ascending and `y` descending).
* Following the lead of geopandas, pyproj, cartopy and rasterio, atlite now uses Coordinate Reference System (`CRS`) instead of the old   fashioned projection strings. 

**New features**


* You can now use wind turbine configurations as stored in the
  `Open Energy Database <https://openenergy-platform.org/dataedit/view/supply/turbine_library>`_
  using the string prefix `"oedb:"` when specifying a turbine,
  e.g. `"oedb:Enercon_E-141/4200"`.
* Atlite now has and uses a new configuration system.
  See the new section on `configuration <https://atlite.readthedocs.io/en/latest/configuration.html>`_
  for details.
* It is possible to merge two cutouts together, using `Cutout.merge`


**Breaking changes**

* The argument `show_progress` of function `atlite.convert.convert_and_aggregate` does not take strings anymore. 
* The argument `layout` of function `atlite.convert.convert_and_aggregate` must be a `xarray.DataArray`.
* Due to the change of the order of coordinates in cutouts the order of coordinates in `matrix` passed to `convert_*` functions
    changed likewise: `x` and `y` are both ascending now.
* Due to the change of the order of coordinates in cutouts the order of elements returned by `grid_coordinates()` has changed.
* Due to the change of the order of coordinates in cutouts the order of elements in the attribute `grid_cells` has changed.


Version 0.0.4
===============

* support negative latitudes to PV panel orientation
* add support for ERA5 back extension to 1950
* add PROJ>=7 valid 'aea' projection string 



Version 0.0.3
==============

Brings a minor bug fix and prepares for the next version jump to version 0.2.

* Fix heat demand hourshift for xarray 0.15.1
* Add Travis CI and simplified release management