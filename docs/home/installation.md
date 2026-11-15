<!--
SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>

SPDX-License-Identifier: CC-BY-4.0
-->

# Installation

atlite requires Python 3.11 or newer. We recommend a dedicated environment, for
example with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html),
[uv](https://docs.astral.sh/uv/) or
[venv](https://docs.python.org/3/library/venv.html).

=== "conda"

    ``` bash
    conda install -c conda-forge atlite
    ```

=== "pip"

    ``` bash
    pip install atlite
    ```

=== "uv"

    ``` bash
    uv add atlite
    ```

To install the latest development version from the `master` branch on GitHub:

``` bash
pip install git+https://github.com/PyPSA/atlite.git
```

## ERA5 credentials

To download weather data, you need credentials for a source of ERA5 data.
atlite supports two sources. See [Datasets](datasets.md) for how they differ.

### Earth Data Hub (recommended)

1. Create an account at <https://earthdatahub.destine.eu>.
2. Go to the [personal access tokens](https://earthdatahub.destine.eu/account-settings#my-personal-access-tokens) in your account settings.
3. Click on the "i" icon next to a token in the section "My personal access tokens".
4. Click on "Download .netrc". Save the file in your project folder or in your home
   folder, or append its content to an existing `.netrc` file.
   As an alternative, store the API key in the environment variable `EARTHDATAHUB_API_KEY`.
5. On Linux and macOS, if the `.netrc` file is in your home folder, it must have the
   permissions `0600` (read and write only for the owner):

    ``` bash
    chmod 600 ~/.netrc
    ```

    This does not apply to a `.netrc` file in your project folder.

### Climate Data Store (fallback)

1. Create an account at <https://cds.climate.copernicus.eu/>.
2. Go to <https://cds.climate.copernicus.eu/how-to-api>.
3. Copy the text in the code box of the section "Setup the CDS API key".
4. Save it in a file called `.cdsapirc` in your home folder.
   As an alternative, set the environment variables `CDSAPI_URL` and `CDSAPI_KEY`
   to the values in the `.cdsapirc` file.

## Computational resources

atlite keeps resource requirements low. They increase with the size of the
cutout in space and time. We run conversions on our laptops, and they usually
run in the background without slowing the computer down.

- **CPU:** atlite does some number crunching, but does not need large multicore CPUs.
- **Memory:** For ERA5, 2-4 GiB is usually sufficient. Other datasets can need more
  memory, as atlite sometimes loads them fully or partially into memory to
  create a cutout.
- **Disk space:** Depends on the size of the cutouts and datasets you use.
  Two examples for reference, both with ERA5 at hourly resolution:
    - Small cutout (Ireland, United Kingdom and parts of the Atlantic Ocean), 1 month: 60 MiB
    - Large cutout (Western Asia), 1 year: 6 GiB

Common small and medium cutouts are no problem for a normal computer.
For very large cutouts, see the example on
[large cutouts with Earth Data Hub](../examples/large_cutouts_with_era5_edh.ipynb).
