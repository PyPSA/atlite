..
  SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>

  SPDX-License-Identifier: CC-BY-4.0

#############
Release Notes
#############


.. Upcoming Release
.. ================

.. .. warning:: 
  
..   The features listed below are not released yet, but will be part of the next release! 
..   To use the features already you have to install the ``master`` branch, e.g. 
..   ``pip install git+https://github.com/pypsa/atlite``.


Version 0.3.0
=============

**Features**

* Add power law interpolation method as a new argument to `cutout.wind` 
  (`#402 <https://github.com/PyPSA/atlite/pull/402>`_)

* Use ``dask.array`` functions in favour of ``numpy`` functions 
  (`#367 <https://github.com/PyPSA/atlite/pull/367>`_)

* Improved CI, testing, linting and build process 
  (`#388 <https://github.com/PyPSA/atlite/pull/388>`_,
  `#392 <https://github.com/PyPSA/atlite/pull/392>`_,
  `#394 <https://github.com/PyPSA/atlite/pull/394>`_,
  `#399 <https://github.com/PyPSA/atlite/pull/399>`_,
  `#309 <https://github.com/PyPSA/atlite/pull/409>`_)

**Bug fixes**

* Adapt ERA5T merge to new CDS API (`#391 <https://github.com/PyPSA/atlite/pull/391>`_)

* Fixes issues with dependeny updates 
  (`#381 <https://github.com/PyPSA/atlite/pull/381>`_,
  `#387 <https://github.com/PyPSA/atlite/pull/387>`_)

* Use ``dask.array`` functions in favour of ``numpy`` functions.
  (`#367 <https://github.com/PyPSA/atlite/pull/367>`_,


Version 0.2.14
==============

* Compatibility with new CDS infrastructure for ERA5 cutouts. Update your API
  key at https://cds-beta.climate.copernicus.eu/how-to-api and use the new API
  endpoint ``https://cds-beta.climate.copernicus.eu/api`` in your
  ``~/.cdsapirc`` file. The old CDS infrastructure can still be accessed when
  the ``~/.cdsapirc`` uses the old endpoint.

* Adds option to toggle whether ERA5 downloads are requested in monthly or
  annual chunks with keyword argument ``cutout.prepare(monthly_requests=True)``.
  The default is now annual requests. The monthly requests can also be posted
  concurrently using ``cutout.prepare(monthly_requests=True,
  concurrent_requests=True)``.

* Improved parallelization of ``atlite.convert.build_line_rating`` by adding
  keyword arguments for ``dask.compute`` (``dask_kwargs={}``) and an option to
  disable the progressbar (``show_progress=False``).

* Default to ``show_progress=False`` for performance reasons.

* Numpy version temporarily limited to <2.

- Remove long deprecated functions and cutout arguments / attributes.

Version 0.2.13
==============

* Added solar tracking support for irradiation; e.g. ``cutout.irradiation(tracking='horizontal')``. (https://github.com/PyPSA/atlite/pull/340)
* Added SARAH-3 compatibility (SARAH-2 can still be used). (https://github.com/PyPSA/atlite/pull/352)
* Fixed passing of `disable_progressbar` argument to ``compute_availabilitymatrix()`` to disable the progress bar. (https://github.com/PyPSA/atlite/pull/356)
* Added dewpoint temperature to the list of features fetched from ERA5. (https://github.com/PyPSA/atlite/pull/342)
* Added option to compute capacity factor time series per grid cell. (https://github.com/PyPSA/atlite/pull/330)
* Fixed pandas deprecations.
* Fixed build of documentation.

Version 0.2.12
==============

* Fix: the wind turbine power curve is checked for a missing cut-out wind speed and an option to add a
  cut-out wind speed at the end of the power curve is introduced. From the next release v0.2.13, adding
  a cut-out wind speed will be the default behavior (`GH #316 <https://github.com/PyPSA/atlite/pull/316>`_)
* Compatibility with xarray >= 2023.09.: The chunked spatial dimension in `aggregate` was raising an error with the new xarray version. This is fixed now.
* Bug fix: Some wind turbine models did not include a cut-out wind speed, potentially causing overestimated power generation in windy conditions. Cut-out wind speeds were added to the following affected wind turbine models (`GH #314 <https://github.com/PyPSA/atlite/issues/314>`_):
    * NREL_ReferenceTurbine_2016CACost_10MW_offshore
    * NREL_ReferenceTurbine_2016CACost_6MW_offshore
    * NREL_ReferenceTurbine_2016CACost_8MW_offshore
    * NREL_ReferenceTurbine_2019ORCost_12MW_offshore
    * NREL_ReferenceTurbine_2019ORCost_15MW_offshore
    * NREL_ReferenceTurbine_2020ATB_12MW_offshore
    * NREL_ReferenceTurbine_2020ATB_15MW_offshore
    * NREL_ReferenceTurbine_2020ATB_18MW_offshore
* Fix: the wind turbine power curve is checked for a missing cut-out wind speed and an option to add a
  cut-out wind speed at the end of the power curve is introduced. From the next release v0.2.13, adding
  a cut-out wind speed will be the default behavior.
* A cutout can now be loaded with setting chunks to ``auto``.
* The Cutout class has a new function ``area`` which return a DataArray with dimensions (x,y) with the area of each grid cell.
* The Cutout class has a new function ``layout_from_area_density`` which returns a capacity layout with the capacity per grid cell based on the area density.

Version 0.2.11
==============


* With this release, we change the license from copyleft GPLv3 to the more liberal MIT license with the consent of all major contributors `#263 <https://github.com/PyPSA/atlite/pull/263>`_.
* Added 1-axis horizontal, 1-axis tilted horizontal, 1-axis vertical, and 2-axis tracking options for solar PV; e.g. ``cutout.pv(tracking='horizontal')``.
* Added small documentation for get_windturbineconfig
* The deprecated functions `grid_cells` and `grid_coordinates` were removed.
* Feature: Cutouts are now compressed differently during the `.prepare(...)` step using the native compression feature of netCDF files.
    This increases time to build a cutout but should reduce cutout file sizes.
    Existing cutouts are not affected. To also compress existing cutouts, load and save them using `xarray` with
    compression specified, see `the xarray documentation <https://docs.xarray.dev/en/stable/generated/xarray.Dataset.to_netcdf.html>`_
    for details.
* Feature: Cutouts from `ERA5` are now downloaded for each month rather than for each year.
  This allows for spatially larger cutouts (worldwide) which previously exceed the maximum
  download size from ERA5.
* Doc: A subsection on how to reduce `cutout` sizes has been added to the documentation.
* Bug notice: An bug in one of `atlite` package dependencies (`xarray`) can lead to `nan` values when using `atlite`.
    A workaround is implemented in `atlite` which reduces the performance when building cutouts, especially for ERA5 cutouts.
    The `nan` values in `cutouts` which are affected by the bug can not be recoevered and the `cutout` needs to be downloaded again.
    For more details on the bug, see the `xarray issue tracker <https://github.com/pydata/xarray/issues/7691>`_.
* The exclusions calculation for geometries not overlapping with the raster was fixed.
* The ExclusionContainer has new functions `compute_shape_availability` and `plot_shape_availability`. These functions ease the inspection of excluded areas within single and multiple geometries.
* Support for newly released ERA5 back extension to 1940.
* Feature: Added the `irradiation` method to `cutout` to access raw irradiation data, as well as the `irradiation=total` keyword to the `TiltedIrradiation` method for a way of accessing `direct`, `diffuse`, and `ground` reflected irradiation quantities separately when needed.
    A new example jupyter notebook `building_stock_weather_aggregation.ipynb` has also been added to demonstrate a use case for the added functionality.

Version 0.2.10
==============

* atlite now supports shapely >= v2.0.
* Bugfix: For certain time spans, the ERA5 influx data would be incorrectly shifted by 12 hours.
  This is now fixed and influx data is **always** shifted by minus 30 minutes.
  See `#256 <https://github.com/PyPSA/atlite/issues/256#issuecomment-1271446531>`_ for details.
* Bugfix: The hydro inflow calculation was relying on a wrong distance calculation in `atlite.hydro.shift_and_aggregate_runoff_for_plants`. This is now fixed.
* Add a reference to the PyPSA ecosystem community server hosted on `Discord <https://discord.gg/AnuJBk23FU>`_
* Bugfix: building cutouts spanning the most recent few months resulted in errors due to the
  mixing of `ERA5` and `ERA5T` data returned from the CDSAPI.
  See `#190 <https://github.com/PyPSA/atlite/issues/190>`_ for details.

Version 0.2.9
=============

* Enable rasterio >1.2.10. Allows now to use the new rasterio 1.3.0 version.

Version 0.2.8
=============

* Bugfix: When creating cutouts using SARAH2 data, an error was previously wrongly thrown if exactly
  the data was available as input as required. The error is now correctly thrown only if
  insufficient SARAH data is available.
* Bugfix: When only adding geometries to an `atlite.ExclusionContainer` the geometries were previously
  not opened and an error was thrown. The error did not occur if one or more shapes were included.
  Error is corrected and geometry-only exclusions can now be calculated. (GH Issue #225)
* atlite now includes the reference turbines from the NREL turbine archive (see: https://nrel.github.io/turbine-models/). Available turbines can be consulted using `atlite.windturbines` and can be passed as string argument, e.g. `coutout.wind(turbine)`.
* Bugfix: Downsampling the availability matrix (high resolution to low resolution) failed. Only rasters with 0 or 1
  were produced. Expected are also floats between 0 and 1 (GH Issue #238). Changing the rasterio version solved this.
  See solution (https://github.com/PyPSA/atlite/pull/240).
* Breaking Change: Due to better performance and memory efficiency the method of matrix summation, as well as the matrix dtpyes within `shape_availability()` in `atlite.gis`, have been changed.
  The returned object `masked` (numpy.array) is now dtype `bool` instead of `float64`. This can create broken workflows, if `masked` is not transformed ahead of certain operations (https://github.com/PyPSA/atlite/pull/243).
* Bugfix: Avoid NaN values into the hydro inflows

Version 0.2.7
==============

* The function `SolarPosition` does not return the atmospheric insolation anymore. This data variable was not used by any of the currently supported modules.


Version 0.2.6
==============

* atlite now supports calculating dynamic line ratings based on the IEEE-738 standard (https://github.com/PyPSA/atlite/pull/189).
* The wind feature provided by ERA5 now also calculates the wind angle `wnd_azimuth` in range [0 - 2π) spanning the cirlce from north in clock-wise direction (0 is north, π/2 is east, -π is south, 3π/2 is west).
* A new intersection matrix function was added, which works similarly to incidence matrix but has boolean values.
* atlite now supports two CSP (concentrated solar power) technologies, solar tower and parabolic trough. See (https://atlite.readthedocs.io/en/latest/examples/working-with-csp.html) for details.
* The solar position (azimuth and altitude) are now part of the cutout feature `influx`. Cutouts created with earlier versions will become incompatible with the next major.
* Automated upload of code coverage reports via Codecov.
* DataArrays returned by `.pv(...)` and `.wind(...)` now have a clearer name and 'units' attribute.
* If the `matrix` argument in conversion functions (`.pv(...)`, `.wind(...)` etc.) is a `DataArray`, the alignment of the coordinate axis with the cutout grid is double-checked.
* Due to ambiguity, conversion functions (`.pv(...)`, `.wind(...)` etc.) now raise an `ValueError` if shapes and matrix are given.
* atlite now supports calculating of heat pump coefficients of performance (https://github.com/PyPSA/atlite/pull/145).
* Enabled the GitHub feature "Cite this repository" to generate a BibTeX file (Added a `CITATION.cff` file to the repository).

**Bug fixes**
* The solar position for ERA5 cutouts is now calculated for half a time step earlier (time-shift by `cutout.dt/2`) to account for the aggregated nature of ERA5 variables (see https://github.com/PyPSA/atlite/issues/158). The fix is only applied to newly created cutouts. Previously created cutouts do not profit from this fix and need to be recreated `cutout.prepare(overwrite=True)`.
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


* atlite now **requires Python 3.6 or higher**.
* We changed the atlite backend for storing cutout data.
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
* atlite now has and uses a new configuration system.
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
