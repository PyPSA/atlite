..
  SPDX-FileCopyrightText: 2016 - 2023 The Atlite Authors

  SPDX-License-Identifier: CC-BY-4.0

############
Installation
############

There are three possibilities to install atlite:

* conda::

    conda install -c conda-forge atlite


* pypi::

    pip install atlite

* or directly from GitHub for the most recent version::

    pip install git+https://github.com/pypsa/atlite/

Requirements
============

Conda environment
-----------------

We provide a `conda environment file <https://github.com/PyPSA/atlite/blob/documentation/environment.yaml>`_
for conveniently setting up all required and optional packages.
For information on setting up conda environments from file,
`click here <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file>`_.

Python version
--------------

With the new version 0.2, Atlite requires an installed version of
**Python 3.6 or above**.

Required packages
-----------------

* bottleneck
* cdsapi
* dask (0.18.0+)
* geopandas
* netcdf4
* numexpr
* numpy
* pandas
* progressbar2
* rasterio
* rtree
* scipy
* shapely
* xarray (0.11.2+)


Computational resources
-----------------------

As for requirements on your computing equipment, we tried to keep
the resource requirements low.
The requirements obviously depend on the size of the cutouts and
datasets your parse and use.

We run our conversions on our laptops and this usually works fine
and can run in the background without clogging our computers.

With regards to

* CPU: While Atlite does some number crunching, it does not require
  special or large multicore CPUs
* Memory: For the ERA5 dataset you should be fine running Atlite with
  even 2-4 GiB.
  Other datasets can require more memory, as they sometimes need to be
  loaded fully or partially into memory for creating a cutout.
* Disk space: Is really all about the cutout and dataset sizes
  (in time and space) you use.
  We can only provide two examples for reference:

    - Small cutout (Republic of Ireland + UK + some atlantic ocean),
      1 month with hourly resolution using ERA5: 60 MiB
    - Large cutout (Western Asia),
      1 year with hourly resolution using ERA5: 6 GiB

We guess you do not need to worry about your computer being able to handle
common small or medium scenarios.

Rule of thumb:
    The requirements on your machine are increasing with the
    size of the cutout (in time and space).
