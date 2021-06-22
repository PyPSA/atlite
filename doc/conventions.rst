..
  SPDX-FileCopyrightText: 2016-2019 The Atlite Authors

  SPDX-License-Identifier: CC-BY-4.0

############
Conventions
############

Atlite uses the following conventions which are applied for processing geo-spatial and temporal data. 

Grid coordinates 
================

According to the ``xarray`` conventions, grid coordinates are ordered such that both ``x`` and ``y`` are ascending.  

The coordinates represent points in the **middle** of the corresponding grid cells. Given a cutout, the geographical data of the grid is given by :py:attr:`atlite.Cutout.grid`, which returns a `GeoPandas` dataframe with coordinates and geometries. The coordinates are the geographical centroids of the geometries, the grid cells. When initializing a cutout with e.g. 

>>> cutout = atlite.Cutout('example', module='era5', x=slice(5,10), y=slice(30,35), time='2013')

the cutout is built with centroids starting at :math:`5` for ``x`` and :math:`30` for ``y``. That means the effective covered area spans from longitude :math:`4.875` to :math:`10.125` and from latitude :math:`29.875` to :math:`35.125`, given a standard resolution of :math:`0.25^\circ\times0.25^\circ` per grid cell. This is in alignment with the extent of the cutout 


>>> cutout.extent 
array([ 4.875, 10.125, 29.875, 35.125])


Time Points 
===========

Following the ERA5 convention, a time-index in an time-dependent data array refers to the end of an averaged time-span. So, given a time resolution of 1 hour, values at 12:00 refer to the time-span from 11:00 to 12:00 and represent the average of that time-span. For datasets other than ERA5 this convention is not necessarily fulfilled. For example, the SARAH data refers to instantaneous data-points. In the implementation, we try to consider this circumstance in order to appropriately align the datasets in order to merge them.  
