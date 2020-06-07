## Copyright 2016-2017 Gorm Andresen (Aarhus University), Jonas Hoersch (FIAS), Tom Brown (FIAS)

## This program is free software; you can redistribute it and/or
## modify it under the terms of the GNU General Public License as
## published by the Free Software Foundation; either version 3 of the
## License, or (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.


"""
Renewable Energy Atlas Lite (Atlite)

Light-weight version of Aarhus RE Atlas for converting weather data to power systems data
"""


from __future__ import absolute_import

from .cutout import Cutout
from .gis import compute_indicatormatrix, regrid

from .version import version as __version__

__author__ = "Gorm Andresen (Aarhus University), Jonas Hoersch (FIAS), Tom Brown (FIAS), Markus Schlott (FIAS), David Schlachtberger (FIAS)"
__copyright__ = "Copyright 2016-2017 Gorm Andresen (Aarhus University), Jonas Hoersch (FIAS), Tom Brown (FIAS), Markus Schlott (FIAS), David Schlachtberger (FIAS), GNU GPL 3"
