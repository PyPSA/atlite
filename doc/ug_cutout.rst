..
  SPDX-FileCopyrightText: 2016-2019 The Atlite Authors

  SPDX-License-Identifier: CC-BY-4.0

################
Creating cutouts
################

Cutouts build the core of Atlite. Once created one, you can easily derive energy system 
relevant data. 
They include all necessary weather data within your temporal and spatial boundaries. 
The following weather data fields can be processed

* Temperature
* Downward short-wave radiation
* Upward short-wave radiation
* Wind 
* Runoff
* Surface roughness
* Height maps
* Soil temperature

And from there, following energy system relevant time series can be derived

* Wind power generation for a given turbine type
* Solar PV power generation for a given panel type
* Solar thermal collector heat output
* Hydroelectric inflow (simplified)
* Heating demand (based on the degree-day approximation)


The creation of cutouts based on the ERA5 dataset is automized in Atlite. 
Using `GeoPandas <http://geopandas.org/>`_ one can easily build cutouts for 
one or more countries at a time

.. code-block:: python

    In [1]: import atlite
    In [2]: import geopandas as gpd

    
    In [3]: baltics = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))\
                 .set_index('name')\
                 .reindex(['Estonia', 'Lithuania', 'Latvia' ])

In order to define the spatial bounds of the cutout, we pass the geometrical 
bounds of the union of all shapes in the GeoPandas.GeoSeries.  

.. code-block:: python

    In [4]: cutout = atlite.Cutout(name="baltics-2011-01",
                            cutout_dir="./",
                            module="era5",
                            bounds=baltics.unary_union.bounds,
                            time="2011-01")

    In [5]: cutout.prepare()

Note that you can likewise pass the bounds as a tuple of floats in the 
form (x1, y1, x2, y2). 
The preparation of a cutout may take some time depending on the queue. 
