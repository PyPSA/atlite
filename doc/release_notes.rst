..
  SPDX-FileCopyrightText: 2016-2019 The Atlite Authors

  SPDX-License-Identifier: CC-BY-4.0

#############
Release Notes
#############

Version 0.2 (to be released)
============================

Major changes
-------------

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

New features
------------

* You can now use wind turbine configurations as stored in the
  `Open Energy Database <https://openenergy-platform.org/dataedit/view/supply/turbine_library>`_
  using the string prefix `"oedb:"` when specifying a turbine,
  e.g. `"oedb:Enercon_E-141/4200"`.
* Atlite now has and uses a new configuration system.
  See the new section on `configuration <https://atlite.readthedocs.io/en/latest/configuration.html>`_
  for details.
