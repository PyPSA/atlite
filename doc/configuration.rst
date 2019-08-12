#############
Configuration
#############

Different settings of Atlite are set with defaults that
suit people in their first steps.

Further down the line you might want to change some of these
settings, like e.g. storage locations for cutouts and datasets.

There are different ways to change the settings in Atlite, depending
on your use case:

1. Settings can be changed live inside the programme.
2. You can create custom files with different configurations and load
   them into atlite as needed.
3. Default settings which are loaded on ``import atlite`` can be overwritten.


Accessing and changing the configuration
========================================

The configuration of atlite can be accessed via the ``atlite.config`` module.
All currently supported configuration options are listed in the variable ::

    atlite.config.ATTRS

To set a different value for a selected option ::

    atlite.config.<option_name> = <new_value>

These settings do not persist, e.g. after reloading Atlite they would have to
be set again.


.. _note-on-file-paths:

A note on file paths
--------------------

For Atlite we designed the configuration system to interpret relative file
paths always relative to the location of the most recently read configuration
file.
This means that if e.g.

* the default configuration is read, all relative paths
  are relative to the install directory of the Atlite package
  (which you can determine via ``atlite.__file__``).
* the configuration is read from your users home directory
  (``~/`` or ``C:\\Users\\<Your username>\\``), then all relative paths
  are considered relative to this location.

There are two exceptions:

* Absolute paths are not affected by this.
* Paths starting with ``<ATLITE>`` are always considered relative to 
  the package installation directory

If you have troubles with the relative path system, a quick solution
is to use absolute paths instead of relative ones.


Saving and loading custom configurations from file
==================================================

You can save the current configuration of Atlite to file using ::

    atlite.config.save(<path>)

Configuration files saved by Atlite can also be edited manually.

The configuration from a file can be read and loaded into Atlite using ::

    atlite.config.read(<path>)

This is helpful when you have multiple projects each having their
own project directory, into which you can put distinct Atlite configuration files.

.. note:: 
    When the configuration from ``path``, this path is then set as the base for
    all relative paths in the configuration.
    See  :ref:`note-on-file-paths` for details.


Default configuration
=====================

If you want to change the default configuration loaded into Atlite upon
``import atlite``, then you can do so by following these steps:

1. Head into the folder of your local Atlite installation
   (can be determined by looking into ``atlite.__file__``).
2. Copy the file ``config.example.yaml`` into your user's home directory,
   i.e. ``~`` (Linux) or ``C:\\Users\\<your username>`` (Windows).
3. Rename the file to ``.atlite.config.yaml`` (note the leading ``.``).
4. Change the configuration in the file according to your needs.

Alternatively you can also download the file from our `GitHub site <https://github.com/PyPsa/atlite>`_.

**Important:**
When the configuration is read from a different directory, here the 
home directory, then relative paths are considered relative to this
directory.
See  :ref:`note-on-file-paths` for details.
