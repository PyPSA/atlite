#########################
Deriving Capacity Factors
#########################


Capacity factors indicate the share of exploitation of a given device. That is the average relative level that a device would run at a specific spot. So, a capacity factor of 0.8 for a 5 MW wind turbine would lead to a average power production of 4 MW. 

.. code-block:: python

    In [6]: cap_factors_wind = cutout.wind('Vestas_V112_3MW', capacity_factor=True)

    100% (1 of 1) |##########################| Elapsed Time: 0:00:00 Time:  0:00:00

    In [7]: cap_factors_pv = cutout.pv('CdTe', capacity_factor=True,
                               orientation='latitude_optimal')

    100% (1 of 1) |##########################| Elapsed Time: 0:00:00 Time:  0:00:00

