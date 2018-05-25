# -*- coding: utf-8 -*-
from __future__ import absolute_import

from operator import itemgetter

import numpy as np
import pandas as pd
import xarray as xr

# Huld model was copied from gsee -- global solar energy estimator
# by Stefan Pfenninger
# https://github.com/renewables-ninja/gsee/blob/master/gsee/pv.py

def _power_huld(irradiance, t_amb, pc):
    """
    AC power per capacity predicted by Huld model, based on W/m2 irradiance.

    Maximum power point tracking is assumed.

    [1] Huld, T. et al., 2010. Mapping the performance of PV modules,
        effects of module type and data averaging. Solar Energy, 84(2),
        p.324-338. DOI: 10.1016/j.solener.2009.12.002
    """

    # normalized module temperature
    T_ = (pc['c_temp_amb'] * t_amb + pc['c_temp_irrad'] * irradiance) - pc['r_tmod']

    # normalized irradiance
    G_ = irradiance / pc['r_irradiance']

    # Suppress divide-by-zero and invalid-value warnings in log for G_ = 0
    with np.errstate(invalid='ignore', divide='ignore'):
        # NB: np.log without base implies base e or ln
        eff = (1 + pc['k_1'] * np.log(G_) + pc['k_2'] * (np.log(G_)) ** 2 +
               T_ * (pc['k_3'] + pc['k_4'] * np.log(G_) +
                     pc['k_5'] * (np.log(G_)) ** 2) +
               pc['k_6'] * (T_ ** 2))

        eff = eff.fillna(0.)
        eff.values[eff.values < 0] = 0.  # Also make sure efficiency can't be negative

    return G_ * eff * pc.get('inverter_efficiency', 1.)

def _power_bofinger(irradiance, t_amb, pc):
    """
    AC power per capacity predicted by bofinger model, based on W/m2 irradiance.

    Maximum power point tracking is assumed.

    [2] Hans Beyer, Gerd Heilscher and Stefan Bofinger, 2004. A robust model
    for the MPP performance of different types of PV-modules applied for the
    performance check of grid connected systems.
    """

    fraction = (pc['NOCT'] - pc['Tamb']) / pc['Intc']

    with np.errstate(divide='ignore', invalid='ignore'):
        eta_ref = (pc['A'] + pc['B']*irradiance + pc['C']*np.log(irradiance))
        eta = (eta_ref *
               (1. + pc['D'] * (fraction * irradiance + (t_amb - pc['Tstd']))) /
               (1. + pc['D'] * fraction / pc['ta'] * eta_ref * irradiance))

    capacity = (pc['A'] + pc['B'] * 1000. + pc['C'] * np.log(1000.))*1e3
    power = irradiance * eta * (pc.get('inverter_efficiency', 1.) / capacity)
    power.values[irradiance.transpose(*irradiance.dims).values < pc['threshold']] = 0.

    return power.rename('AC power')

def SolarPanelModel(ds, irradiance, pc):
    model = pc.get('model', 'huld')

    if model == 'huld':
        return _power_huld(irradiance, ds['temperature'], pc)
    elif model == 'bofinger':
        return _power_bofinger(irradiance, ds['temperature'], pc)
    else:
        AssertionError("Unknown panel model: {}".format(model))
