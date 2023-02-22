..
  SPDX-FileCopyrightText: 2016 - 2023 The Atlite Authors

  SPDX-License-Identifier: CC-BY-4.0

.. include:: ../RELEASE_NOTES.rst

* In ``atlite/resource.py``, the functions ``get_windturbineconfig``, ``get_solarpanelconfig``, and
  ``get_cspinstallationconfig`` will now recognize if a local file was passed, and if so load
  it instead of one of the predefined ones.
