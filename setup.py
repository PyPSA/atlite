# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2016-2019 The Atlite Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

from setuptools import setup, find_packages
from codecs import open

with open("README.rst", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="atlite",
    author="The Atlite Authors",
    author_email="jonas.hoersch@posteo.de",
    description="Library for fetching and converting weather data to power systems data",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/PyPSA/atlite",
    license="GPLv3",
    packages=find_packages(exclude=["doc", "test"]),
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "scipy",
        "pandas>=0.25",
        "bottleneck",
        "numexpr",
        "xarray>=0.20",
        "netcdf4",
        "dask>=2021.10.0",
        "toolz",
        "requests",
        "pyyaml",
        "rasterio>1.2.10",
        "shapely",
        "progressbar2",
        "tqdm",
        "pyproj>=2",
        "geopandas",
        "cdsapi",
    ],
    extras_require={
        "docs": ["numpydoc", "sphinx", "sphinx_rtd_theme", "nbsphinx", "nbsphinx-link"],
        "dev": ["pre-commit", "pytest", "pytest-cov"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
    ],
)
