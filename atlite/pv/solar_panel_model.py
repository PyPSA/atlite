from __future__ import absolute_import

from operator import itemgetter

import numpy as np
import pandas as pd
import xarray as xr

def SolarPanelModel(ds, irradiation, pc):
    A, B, C, D = itemgetter('A', 'B', 'C', 'D')(pc)
    fraction = (pc['NOCT'] - pc['Tamb']) / pc['Intc']

    with np.errstate(divide='ignore', invalid='ignore'):
        eta_ref = (A + B*irradiation + C*np.log(irradiation))
        eta = (eta_ref *
               (1. + D * (fraction * irradiation + (ds['temperature'] - pc['Tstd']))) /
               (1. + D * fraction / pc['ta'] * eta_ref * irradiation))

    power = irradiation * eta * pc['inverter_efficiency']
    power.values[irradiation.transpose(*irradiation.dims).values < pc['threshold']] = 0.

    return power.rename('AC power')
