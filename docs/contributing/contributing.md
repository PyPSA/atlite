<!--
SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>

SPDX-License-Identifier: CC-BY-4.0
-->

# Contributing

We welcome anyone interested in contributing to atlite, be it with new ideas,
suggestions, bug reports or code.

The [Discord server](https://discord.gg/AnuJBk23FU) of the PyPSA ecosystem has
public voice and text channels to organise projects, ask questions, share news,
and chat with the community.

Please submit issues and pull requests to the
[GitHub repository](https://github.com/PyPSA/atlite).
For the next development goals, see the
[vision and roadmap project](https://github.com/orgs/PyPSA/projects/12/views/1)
and the list of open
[enhancements](https://github.com/PyPSA/atlite/issues/?q=is%3Aissue%20state%3Aopen%20label%3A%22type%3A%20enhancement%22).

## Development setup

Clone the repository and install atlite in editable mode with the development
and documentation dependencies:

``` bash
git clone https://github.com/PyPSA/atlite.git
cd atlite
uv sync --extra dev --extra docs
```

Run the tests with:

``` bash
uv run pytest
```

Tests that download weather data need [ERA5 credentials](../home/installation.md#era5-credentials). Without them, pytest skips these tests.

## Code style

We use [pre-commit](https://pre-commit.com/) to lint and format code with
[ruff](https://docs.astral.sh/ruff/), check types with [mypy](https://mypy-lang.org/),
and check [REUSE compliance](https://reuse.software/).

- To run the hooks on every `git commit`: `pre-commit install`
- To run the hooks manually: `pre-commit run --all`

Every pull request that changes functionality should add a note to the
[release notes](../release-notes.md) under "Upcoming Release".

## Documentation

The documentation uses [MkDocs](https://www.mkdocs.org/) with the
[Material](https://squidfunk.github.io/mkdocs-material/) theme. The pages are
Markdown files in `docs/`; the navigation is defined in `mkdocs.yml`.
The [API reference](../api/cutout.md) is generated from the NumPy-style
docstrings with [mkdocstrings](https://mkdocstrings.github.io/).
To preview the documentation locally with live reload, run:

``` bash
uv run mkdocs serve
```

To check for broken links and other warnings, run:

``` bash
uv run mkdocs build --strict
```

## Contributing examples

Examples are always welcome. You can submit a
[Jupyter notebook](https://docs.jupyter.org/en/latest/) (`.ipynb`) directly
as an example:

1. Write the notebook, e.g. `foo.ipynb`, and place it in `docs/examples/`.
2. Decide for each cell whether its output is relevant for the example.
   The documentation does not run the notebooks, but shows their stored output.
   Keep relevant output, but keep the number of figures small.
   Clear all other output, e.g. with *Edit → Clear Outputs of All Cells* in JupyterLab.
3. Add the notebook to the `nav` section in `mkdocs.yml`, or tell us in the pull
   request where it should appear.
4. Add a short description to the [examples overview](../examples/examples.md).
5. Open a pull request.

For the reasoning behind this workflow, see
[atlite#38](https://github.com/PyPSA/atlite/issues/38).
