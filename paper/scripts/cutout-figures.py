#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:47:13 2021

@author: fabulous
"""

import atlite
import matplotlib.pyplot as plt
import geopandas as gpd
import cartopy.crs as ccrs
from cartopy.crs import PlateCarree as plate
import cartopy.io.shapereader as shpreader
from pathlib import Path 


plt.rc('figure', figsize=(10, 6))

figpath = Path('../figures/')


        


cutout = atlite.Cutout('cutout', x=slice(10, 30), y=slice(40,50), time='2013', module='era5')


# fig, ax = plt.subplots()



# shpfilename = shpreader.natural_earth(resolution='10m', category='cultural',
                                      # name='admin_0_countries')
# reader = shpreader.Reader(shpfilename)
# UkIr = gpd.GeoSeries({r.attributes['NAME_EN']: r.geometry for r in reader.records()},
                     # crs={'init': 'epsg:4326'}).reindex(['United Kingdom', 'Ireland'])