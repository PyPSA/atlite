# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2016 - 2023 The Atlite Authors
#
# SPDX-License-Identifier: MIT

from codecs import open

from setuptools import find_packages, setup

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
    license="MIT",
    packages=find_packages(exclude=["doc", "test"]),
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=[
        "numpy<2",
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
        "rasterio!=1.2.10",
        "shapely",
        "progressbar2",
        "tqdm",
        "pyproj>=2",
        "geopandas",
        "cdsapi>=0.7,<0.7.3",
    ],
    extras_require={
        "docs": [
            "numpydoc",
            "sphinx",
            "sphinx-book-theme",
            "nbsphinx",
            "nbsphinx-link",
            "docutils==0.20",  # Just temporarily until sphinx-docutils is updated (see https://github.com/sphinx-doc/sphinx/issues/12340)
        ],
        "dev": ["pre-commit", "pytest", "pytest-cov", "matplotlib"],
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
    ],
)
