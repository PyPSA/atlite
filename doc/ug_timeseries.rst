..
  SPDX-FileCopyrightText: 2016-2019 The Atlite Authors

  SPDX-License-Identifier: CC-BY-4.0

####################
Deriving time-series
####################

In the same way as the capacity factors, power generation time series can directly be extracted from the cutout itself. 


.. code-block:: python

  In [7]: wind_power = cutout.wind('Vestas_V112_3MW', shapes=baltics.geometry)

  100% (1 of 1) |##########################| Elapsed Time: 0:00:00 Time:  0:00:00

  In [8]: solar_power = cutout.pv('CdTe', orientation='latitude_optimal', shapes=baltics.geometry)

  100% (1 of 1) |##########################| Elapsed Time: 0:00:00 Time:  0:00:00

Atlite comes along with a small set of wind turbine and solar configuration files, stored in 

.. code-block:: python

  In [9]: atlite.utils.construct_filepath(atlite.config.windturbine_dir)

and 

.. code-block:: python

  In [10]: atlite.utils.construct_filepath(atlite.config.solarpanel_dir)

