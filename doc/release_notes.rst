..
  SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>

  SPDX-License-Identifier: CC-BY-4.0

.. include:: ../RELEASE_NOTES.rst

* In ``atlite/resource.py``, the functions ``get_windturbineconfig``, ``get_solarpanelconfig``, and
  ``get_cspinstallationconfig`` will now recognize if a local file was passed, and if so load
  it instead of one of the predefined ones.

* The option ``capacity_factor_timeseries`` can be selected when creating capacity factors to obtain
 the capacity factor of the selected resource per grid cell.
