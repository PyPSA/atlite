..
  SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>

  SPDX-License-Identifier: CC-BY-4.0

##################
Migrating cutouts
##################

The structure of cutout files can change between atlite releases.
Each version of ``atlite`` supports one or multiple cutout versions and to tell these
apart, every cutout carries a **schema version**.
Cutouts written the before schema versions were introduced do not have this attribute and
are treated as version ``0``.

Migrating a cutout
==================

Migration writes the migrated cutout to a new file and leaves the original file
unchanged. From the command line:

.. code-block:: bash

    python -m atlite.migrate europe-2013-era5.nc --output europe-2013-era5-v1.nc

Or from Python:

.. code-block:: python

    from atlite.migrate import migrate_cutout

    migrate_cutout("europe-2013-era5.nc", output="europe-2013-era5-v1.nc")

If the migration fails, no output file is written.

To overwrite the original file instead, pass ``--in-place`` on the command line
or ``in_place=True`` in Python without an output file.

Migration upgrades a cutout step by step from its schema version to the
current one. Where possible, a step only updates the file's metadata. Migrating
from version 0 to 1, for example, only adds the version attribute and does not
rewrite the data, so it is fast even for large cutouts.

There is no guarantee that migration of cutouts is always possible.

.. note::

    A migrated cutout is not necessarily identical to a cutout freshly created
    with the current atlite release. Migration only converts the structure of
    the cutout file to the current schema version; it keeps the data as it was
    originally prepared. Between the creation of the original cutout and today,
    the processing in atlite and the upstream data sources (for example
    reanalysis updates or corrections published by data providers) may have
    changed. If you need data that is consistent with the current atlite
    release and upstream data, create the cutout again instead of migrating it.


Existing schema versions and changes between them
=================================================

========  =====================================================
Version   Changes
========  =====================================================
0         Cutouts written before schema versions were introduced.
1         Adds the ``atlite_cutout_schema_version`` attribute.
========  =====================================================

Checking the schema version
===========================

The schema version of a cutout file can be read without loading its data:

>>> from atlite.schema import read_schema_version
>>> read_schema_version("europe-2013-era5.nc")
0

For a loaded cutout, use the attribute `atlite.Cutout.schema_version`.

What happens when a cutout is loaded
====================================

atlite never migrates cutouts implicitly.
Depending on the schema version of the cutout, loading it with :py:class:`atlite.Cutout` behaves as follows:

* **Current version:** the cutout is loaded.
* **Older, still supported version:** the cutout is loaded, and an
  :py:class:`atlite.schema.OutdatedCutoutWarning` is shown that includes the
  commands to migrate it. Modifications to the cutout, e.g. using :py:meth:`atlite.Cutout.prepare` 
  are refused and need the cutout to be migrated first.
* **Unsupported version:** an :py:class:`atlite.schema.IncompatibleCutoutError`
  is raised. Cutouts that are too new require a newer atlite release. Cutouts
  that are too old have to be migrated with an older atlite release or created
  again.