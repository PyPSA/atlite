..
  SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>

  SPDX-License-Identifier: CC-BY-4.0

.. include:: ../RELEASE_NOTES.rst

* In ``atlite/convert.py``, the function ``hydro`` is now correctly calculating the runoff in m3 per hour by multiplying the depth (in meters per hour) by the basin area (in m2).
