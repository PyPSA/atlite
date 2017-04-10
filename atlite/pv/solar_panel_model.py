from __future__ import absolute_import

from operator import itemgetter

import numpy as np
import pandas as pd
import xarray as xr

def SolarPanelModel(ds, irradiation, pc):
    irrad_t = irradiation['total tilted']
    A, B, C, D = itemgetter('A', 'B', 'C', 'D')(pc)

    eta_ref = (A + B*irrad_t + C*np.log(irrad_t))
    fraction = (pc['NOCT'] - pc['Tamb']) / pc['Intc']

    eta = (eta_ref *
           (1. + D * (fraction * irrad_t + (ds['temperature'] - pc['Tstd']))) /
           (1. + D * fraction / pc['ta'] * eta_ref * irrad_t))

    power = irrad_t * eta * pc['inverter_efficiency']
    power.values[irrad_t.transpose(*irrad_t.dims).values < pc['threshold']] = 0.

    return xr.Dataset({'AC power': power})
