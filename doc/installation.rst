..
  SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>

  SPDX-License-Identifier: CC-BY-4.0

##############
Installation
##############


Getting Python
==============

If it is your first time with Python, we recommend `conda
<https://docs.conda.io/en/latest/miniconda.html>`_ or `pip
<https://pip.pypa.io/en/stable/>`_ as easy-to-use package managers. They are
available for Windows, Mac OS X and GNU/Linux.

It is always helpful to use dedicated `conda environments 
<https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ 
or `virtual environments <https://pypi.python.org/pypi/virtualenv>`_.


Installation with conda
=======================

If you are using ``conda`` you can install PyPSA with::

    conda install -c conda-forge atlite


Installing with pip
===================

If you have the Python package installer ``pip`` then just run::

    pip install atlite

If you're feeling adventurous, you can also install the latest master branch from github with::

    pip install git+https://github.com/PyPSA/atlite.git

ERA5 dataset credentials
========================

To download weather data, you need to obtain credentials for a source of ERA5 data. 

We currently support two methods for dataset access.

Earth Data Hub (recommended)
-----------------------------

1. Create an account at https://earthdatahub.destine.eu
2. Go to https://earthdatahub.destine.eu/account-settings#my-personal-access-tokens
3. Click on the "i" icon next to a token in the "My personal access tokens" section
4. Click on "Download .netrc". Save the resulting file in the project folder or in your home folder (or append to an existing ``.netrc``)
   Alternatively, you can store the API key in this file as a ``EARTHDATAHUB_API_KEY`` environmental variable
5. For Linux/Mac users, if using ``.netrc`` in your home folder, ensure that is has ``0600`` (owner read/write only) permissions
   This doesn't apply if storing the file in your project folder

Climate data store (fallback)
-------------------------------

1. Create an account at https://cds.climate.copernicus.eu/
2. Go to https://cds.climate.copernicus.eu/how-to-api
3. Copy the text inside the code callout in the "Setup the CDS Api key" section
4. Save it in a file called ``.cdsapirc`` in your home folder
   Alternatively, you can define ``CDSAPI_URL`` and ``CDSAPI_KEY`` environmental variables based on the
   contents of the ``.cdsapirc`` file

Computational resources
=======================

As for requirements on your computing equipment, we tried to keep
the resource requirements low.
The requirements obviously depend on the size of the cutouts and
datasets your parse and use.

We run our conversions on our laptops and this usually works fine
and can run in the background without clogging our computers.

With regards to

* CPU: While atlite does some number crunching, it does not require
  special or large multicore CPUs
* Memory: For the ERA5 dataset you should be fine running atlite with
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
