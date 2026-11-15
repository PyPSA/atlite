..
  SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>

  SPDX-License-Identifier: CC-BY-4.0

############
Conventions
############

atlite uses the following conventions which are applied for processing geo-spatial and temporal data.

Grid coordinates
================

According to the ``xarray`` conventions, grid coordinates are ordered such that both ``x`` and ``y`` are ascending.

The coordinates represent points in the **center** of the corresponding grid cells. Given a cutout, the geographical data of the grid is given by :py:attr:`atlite.Cutout.grid`, which returns a `GeoPandas` dataframe with coordinates and geometries. The coordinates are the geographical centroids of the geometries, the grid cells. When initializing a cutout with e.g.

>>> cutout = atlite.Cutout('example', module='era5', x=slice(5,10), y=slice(30,35), time='2013')

the cutout is built with centroids starting at :math:`5^\circ` for ``x`` (longitude) and :math:`30^\circ` for ``y`` (latitude). That means the effective covered area by the cutout spans from longitude :math:`4.875^\circ` to :math:`10.125^\circ` and from latitude :math:`29.875^\circ` to :math:`35.125^\circ`, given the default resolution of :math:`0.25^\circ\times0.25^\circ` per grid cell. This information is accessible via the the extent of the cutout:


>>> cutout.extent
array([ 4.875, 10.125, 29.875, 35.125])


Time Points
===========

Following the ERA5 convention, the time-index of time-dependent data refers to the end of the time-span over which was averaged. So, given a time resolution of 1 hour (=averaging window), the time-index 12:00 refers to the time-averaged values from 11:00 to 12:00. For datasets other than ERA5 this convention is not necessarily fulfilled. For example, the SARAH data refers to instantaneous data-points, i.e. data for the time-index 12:00 refers to the momentaneous value of the variable at 12:00. In the implementation, we try to consider this circumstance in order to appropriately align the datasets in order to merge them.
