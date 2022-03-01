..
  SPDX-FileCopyrightText: 2016-2019 The Atlite Authors

  SPDX-License-Identifier: CC-BY-4.0

#############
Release Notes
#############


.. Upcoming Release
.. =================

Version 0.2.7 
==============

* The function `SolarPosition` does not return the atmospheric insolation anymore. This data variable was not used by any of the currently supported modules. 


Version 0.2.6 
==============

* Atlite now supports calculating dynamic line ratings based on the IEEE-738 standard (https://github.com/PyPSA/atlite/pull/189).
* The wind feature provided by ERA5 now also calculates the wind angle `wnd_azimuth` in range [0 - 2π) spanning the cirlce from north in clock-wise direction (0 is north, π/2 is east, -π is south, 3π/2 is west).
* A new intersection matrix function was added, which works similarly to incidence matrix but has boolean values.
* Atlite now supports two CSP (concentrated solar power) technologies, solar tower and parabolic trough. See (https://atlite.readthedocs.io/en/latest/examples/working-with-csp.html) for details.
* The solar position (azimuth and altitude) are now part of the cutout feature `influx`. Cutouts created with earlier versions will become incompatible with the next major.
* Automated upload of code coverage reports via Codecov.
* DataArrays returned by `.pv(...)` and `.wind(...)` now have a clearer name and 'units' attribute.
* If the `matrix` argument in conversion functions (`.pv(...)`, `.wind(...)` etc.) is a `DataArray`, the alignment of the coordinate axis with the cutout grid is double-checked. 
* Due to ambiguity, conversion functions (`.pv(...)`, `.wind(...)` etc.) now raise an `ValueError` if shapes and matrix are given. 
* Atlite now supports calculating of heat pump coefficients of performance (https://github.com/PyPSA/atlite/pull/145).
* Enabled the GitHub feature "Cite this repository" to generate a BibTeX file (Added a `CITATION.cff` file to the repository).

**Bug fixes**
* The solar position for ERA5 cutouts is now calculated for half a time step earlier (time-shift by `cutout.dt/2`) to account for the aggregated nature of
  ERA5 variables (see https://github.com/PyPSA/atlite/issues/158). The fix is only applied to newly created cutouts. Previously created cutouts do not profit
  from this fix and need to be recreated `cutout.prepare(overwrite=True)`.
* The functions `make_latitude` and `make_latitude_optimal` were not converting degrees to radian correctly. This resulted in a wrong calculation of the power output when using the orientation `latitude_optimal` or `latitude` in the `pv` conversion function. We are sorry for inconveniences.   


Version 0.2.5 
==============

* Clarification for ``ExclusionContainer.add_raster(..)`` that ``codes=..`` does not accept ``lambda``-functions in combination with ``multiprocessing``.
* Internal change: We are moving to `black` for internal code formatting.
* Fix ignored keywords in convert_and_aggregate(...) for capacity_layout=True.

Version 0.2.4 
==============

* Fix cutout merge and update for xarray ``>=v0.18.0`` (https://github.com/PyPSA/atlite/issues/147)
* Set multiprocessing context to ``spawn`` for ensuring equal computation across all platforms. 

Version 0.2.3 
==============

* The progressbar used in ``atlite.gis.availability_matrix`` is now a `tqdm` progressbar which displays better in parallel executions.
* The function ``layout_from_capacity_list`` was added to the cutout class. It is a convenience function that calculates the aggregated capacities per cutout grid cells (layout) based on a list of capacities with coordinates, e.g. list of wind turbines.    
* The dask version was fixed to a xarray-compatible versions (see https://github.com/dask/dask/issues/7583)

Version 0.2.2 
==============

This update is mainly due to fixes in the data handling of the SARAH module. If you work with the SARAH data, we encourage you to update. 

* Fixed compatibility with xarray v0.17.
* Fixed sarah data for ``dx = dy = 0.05``. Due to the float32 dtype of the sarah coordinates, the cutout coordinates were corrupted when merging. This was fixed in the sarah module by converting the coordinates to float64. This also speeds up the cutout creation for more coarse grained cutouts.  
* Fixed sarah data for a time frequency of 30 minutes. This was raising an assertion error as the (new) pandas frequency string for 30 minutes is '30T' not '30min'.
* Fix the ``regrid`` function in ``atlite.gis`` for target coords which are not having the same bounds as the original ``xarray.Dataset``. The previous implementation was leading to a small shift of coordinates in the preparation of SARAH data.



Version 0.2.1
==============
* The `regrid` function in `atlite.gis` was fixed. The previous implementation set an affine transform starting at the center of a cell at the origin. The corrected transform starts at the real origin (origin of the origin cell). Further a padding of the extent ensures that all values are taken into account in the target projection.  
* Exclusion Calculation is now possible with `atlite` (find an usage example at Examples -> Calculate Landuse Availability), Therefore 

  - a new class  `atlite.gis.ExclusionContainer`  was added. It serves as a container of rasters and geometries which should be excluded from the landuse availability.  
  - `Cutout` has a new `availabilitymatrix` function which calculates the overlap of weather cells with shapes while excluding areas based on an `ExclusionContainer`.  
  - `Cutout` has now a affine transform property (`rasterio.Affine`). 
* Fix resolution for dx and dy unequal to 0.25: Due to floating point precision errors, loading data with ERA5 corrupted the cutout coordinates. This was fixed by converting the dtype of era5 coordinates to float64 and rounding. Corresponding tests were added.
* Round cutout.dx and cutout.dy in order to prevent precision errors.    
* Allow passing keyword arguments to `dask.compute` in `convert_and_aggregate` functions. 
* The Cutout class has a new property `bounds` (same as extent but in different order).

**Breaking Change**
* `Cutout.extent` was adjusted to cover the whole cutout area. The extent is now a numpy array. Before, it indicated the coordinates of the centers of the corner cells. 

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