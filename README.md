<!--
SPDX-FileCopyrightText: 2016-2019 The Atlite Authors

SPDX-License-Identifier: CC-BY-4.0
-->


# Atlite

[![PyPI version](https://img.shields.io/pypi/v/atlite.svg)](https://pypi.python.org/pypi/atlite)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/atlite.svg)](https://anaconda.org/conda-forge/atlite)
[![Documentation Status](https://readthedocs.org/projects/atlite/badge/?version=latest)](https://atlite.readthedocs.io/en/latest/?badge=latest)
[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat)](https://github.com/RichardLitt/standard-readme)
[![image](https://img.shields.io/pypi/l/atlite.svg)](License)

> Atlite is a [free software](http://www.gnu.org/philosophy/free-sw.en.html),
> [xarray](http://xarray.pydata.org/en/stable/)-based Python library for
> converting weather data (like wind speeds) into energy systems data. It
> is designed to by lightweight and work with big weather datasets while
> keeping the resource requirements especially on CPU and RAM resources
> low.

## Installation

To install you need a working installation running Python 3.6 or above
and we strongly recommend using either miniconda or anaconda for package
management.

To install the current stable version:

with `conda` from [conda-forge](https://anaconda.org/conda-forge/atlite)

```shell
    conda install -c conda-forge atlite
```

with `pip` from [pypi](https://pypi.org/project/atlite/>)

```shell
    pip install atlite
```

to install the most recent upstream version from [GitHub](https://github.com/pypsa/atlite)

```shell
    pip install git+https://github.com/pypsa/atlite.git
```

## Getting started

Please check the [documentation on getting started](https://atlite.readthedocs.io/en/latest/getting-started.html).

## Contributing

If you have any ideas, suggestions or encounter problems, feel invited
to file issues or make pull requests.

## License

Copyright (C) 2016-2019 The Atlite Authors:

* Jonas Hörsch (RLI)
* Tom Brown (KIT)
* Johannes Hampp (JLUG)
* Gorm Andresen (AU)
* David Schlachtberger (FIAS)
* Markus Schlott (FIAS)

Follow [this link for details](https://atlite.readthedocs.io/en/latest/contributing.html#contributors-and-copyrights).

This work is licensed under multiple licences:

* All original source code is licensed under [GPL-3.0-or-later](LICENSES/GPL-3.0-or-later.txt).
* The documentation is licensed under [CC-BY-4.0](LICENSES/CC-BY-4.0.txt).
* Configuration and data files are mostly licensed under [CC0-1.0](LICENSES/CC0-1.0.txt).

See the individual files for license details.
