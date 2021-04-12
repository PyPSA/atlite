#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 22:13:21 2021

@author: fabian
"""
import atlite

fine = atlite.Cutout('test-fine', x=slice(10, 12), y=slice(30, 32), time='2015-01-01',
                     module='era5')

fine.prepare()

coarse = atlite.Cutout('test-fine', dx=0.5, dy=0.5,
                       x=slice(10, 12), y=slice(30, 32), time='2015-01-01',
                       module='era5')
coarse.prepare()
