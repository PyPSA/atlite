<!--
SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>

SPDX-License-Identifier: CC-BY-4.0
-->

# Migrating Cutouts

The structure of cutout files can change between atlite releases. Each atlite
version supports one or more cutout versions. To tell them apart, every cutout
carries a **schema version**. Cutouts written before schema versions were
introduced do not have this attribute; atlite treats them as version `0`.

## Migrating a cutout

Migration writes the migrated cutout to a new file and leaves the original file
unchanged.

=== "Command line"

    ``` bash
    python -m atlite.migrate europe-2013-era5.nc --output europe-2013-era5-v1.nc
    ```

=== "Python"

    ``` python
    from atlite.migrate import migrate_cutout

    migrate_cutout("europe-2013-era5.nc", output="europe-2013-era5-v1.nc")
    ```

If the migration fails, atlite writes no output file.

To overwrite the original file instead, pass `--in-place` on the command line, or
`in_place=True` in Python, without an output file.

Migration upgrades a cutout step by step from its schema version to the current
one. Where possible, a step only updates the metadata of the file. For example,
the migration from version 0 to 1 only adds the version attribute and does not
rewrite the data. It is therefore fast, even for large cutouts.

Migration is not guaranteed to be possible for all cutouts.

!!! note

    A migrated cutout is not necessarily identical to a cutout created with the
    current atlite release. Migration only converts the structure of the file to
    the current schema version and keeps the data as it was prepared. Since the
    original cutout was created, the processing in atlite and the upstream data
    may have changed, for example through reanalysis updates or corrections by
    the data providers. If you need data that is consistent with the current
    atlite release and upstream data, create the cutout again instead of
    migrating it.

## Schema versions

| Version | Changes |
| --- | --- |
| 0 | Cutouts written before schema versions were introduced. |
| 1 | Adds the `atlite_cutout_schema_version` attribute. |

## Checking the schema version

[`read_schema_version`][atlite.schema.read_schema_version] reads the schema
version of a cutout file without loading its data:

``` python
>>> from atlite.schema import read_schema_version
>>> read_schema_version("europe-2013-era5.nc")
0
```

For a loaded cutout, use the attribute
[`Cutout.schema_version`][atlite.Cutout.schema_version].

## Loading a cutout

atlite never migrates cutouts implicitly. When you load a cutout with
[`atlite.Cutout`][atlite.Cutout], the result depends on its schema version:

- **Current version:** atlite loads the cutout.
- **Older, still supported version:** atlite loads the cutout and shows an
  [`OutdatedCutoutWarning`][atlite.schema.OutdatedCutoutWarning] with the
  commands to migrate it. atlite refuses to modify the cutout, for example with
  [`Cutout.prepare`][atlite.Cutout.prepare], until you migrate it.
- **Unsupported version:** atlite raises an
  [`IncompatibleCutoutError`][atlite.schema.IncompatibleCutoutError].
  Cutouts that are too new need a newer atlite release. Cutouts that are too old
  must be migrated with an older atlite release or created again.
