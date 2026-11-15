<!--
SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>

SPDX-License-Identifier: CC-BY-4.0
-->

# atlite

[![PyPI version](https://img.shields.io/pypi/v/atlite.svg)](https://pypi.python.org/pypi/atlite)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/atlite.svg)](https://anaconda.org/conda-forge/atlite)
[![Documentation Status](https://readthedocs.org/projects/atlite/badge/?version=master)](https://atlite.readthedocs.io/en/master/?badge=master)
[![Tests](https://github.com/PyPSA/atlite/actions/workflows/test.yaml/badge.svg)](https://github.com/PyPSA/atlite/actions/workflows/test.yaml)
[![codecov](https://codecov.io/gh/PyPSA/atlite/branch/master/graph/badge.svg?token=TEJ16CMIHJ)](https://codecov.io/gh/PyPSA/atlite)
[![License](https://img.shields.io/pypi/l/atlite.svg)](LICENSES/MIT.txt)
[![REUSE status](https://api.reuse.software/badge/github.com/pypsa/atlite)](https://api.reuse.software/info/github.com/pypsa/atlite)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/PyPSA/atlite/master.svg)](https://results.pre-commit.ci/latest/github/PyPSA/atlite/master)
[![JOSS](https://joss.theoj.org/papers/10.21105/joss.03294/status.svg)](https://doi.org/10.21105/joss.03294)
[![Discord](https://img.shields.io/discord/911692131440148490?logo=discord)](https://discord.gg/AnuJBk23FU)

atlite is a [free software](https://www.gnu.org/philosophy/free-sw.en.html),
[xarray](https://docs.xarray.dev/en/stable/)-based Python library for
converting weather data (like wind speeds and solar influx) into energy system
data. It is lightweight and keeps computing resource requirements (CPU, RAM)
low. It is therefore well suited for big weather datasets.

atlite converts the following weather data fields into time series relevant
for energy systems, for any subset of a full weather database.

![Weather data fields that atlite processes and the time series it creates](docs/assets/images/workflow_chart.png)

atlite was initially developed by the Renewable Energy Group at
[FIAS](https://fias.institute/) to carry out simulations for the
[CoNDyNet project (archived)](https://web.archive.org/web/20210420004051/http://condynet.de/),
financed by the [German Federal Ministry of Education and Research (BMBF)](https://www.bmftr.bund.de/EN/)
as part of the
[Stromnetze Research Initiative (archived)](https://web.archive.org/web/20190127192900/http://forschung-stromnetze.info/projekte/grundlagen-und-konzepte-fuer-effiziente-dezentrale-stromnetze/).

With atlite, we want to build a bridge between meteorology and energy system
modelling. Traditionally, the two communities have not interacted much, and the
results of one reached the other only slowly. atlite makes it easy to use
meteorological data in energy system models: it offers standard ways to access
weather and climate datasets and to convert them into weather-dependent inputs.

For the next development goals, see the
[vision and roadmap project](https://github.com/orgs/PyPSA/projects/12/views/1)
and the list of open
[enhancements](https://github.com/PyPSA/atlite/issues/?q=is%3Aissue%20state%3Aopen%20label%3A%22type%3A%20enhancement%22).

## Installation

atlite requires Python 3.11 or newer. Install the current stable version with
`conda` from [conda-forge](https://anaconda.org/conda-forge/atlite):

``` shell
conda install -c conda-forge atlite
```

or with `pip` from [PyPI](https://pypi.org/project/atlite/):

``` shell
pip install atlite
```

To install the latest development version from GitHub:

``` shell
pip install git+https://github.com/PyPSA/atlite.git
```

## Documentation

Read the [documentation](https://atlite.readthedocs.io/en/latest/).

## Support and contributing

- To **discuss** with other PyPSA and atlite users, organise projects, share news, and
  get in touch with the community, join the [Discord server](https://discord.gg/AnuJBk23FU).
- For **bugs and feature requests**, use the [issue tracker](https://github.com/PyPSA/atlite/issues).
- We welcome anyone interested in **contributing** to this project. If you have ideas or
  suggestions, or encounter problems, please open an issue or a pull request.
  See the [contributing guidelines](https://atlite.readthedocs.io/en/latest/contributing/contributing/).

## Authors and copyright

Copyright (C) Contributors to atlite <https://github.com/pypsa/atlite>

See the [contributors](https://atlite.readthedocs.io/en/latest/contributing/contributors/) for details.

## License

This work is licensed under multiple licenses:

- All original source code is licensed under [MIT](LICENSES/MIT.txt).
- The documentation is licensed under [CC-BY-4.0](LICENSES/CC-BY-4.0.txt).
- Configuration and data files are mostly licensed under [CC0-1.0](LICENSES/CC0-1.0.txt).

See the individual files for license details.
