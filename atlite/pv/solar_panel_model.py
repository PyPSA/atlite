import numpy as np
import pandas as pd
import xarray as xr

def ReferenceEfficiency(irradiation, panelconfig):
    irrad_t = irradiation['total tilted']

    eta_ref = panelconfig['A'] + panelconfig['B']*irrad_t + panelconfig['C']*np.log(irrad_t)
    #neglect last two terms for no incoming radiation
    eta_ref = eta_ref.where(eta_ref>=panelconfig['A'])
    eta_ref = eta_ref.fillna(panelconfig['A'])
    return eta_ref.rename('reference efficiency')

def CellTemperature(ds, irradiation, eta_ref, panelconfig):
    temp = ds['temperature']
    irrad_t = irradiation['total tilted']

    T = panelconfig['NOCT'] - panelconfig['Tamb']
    I = irrad_t / panelconfig['Intc']
    frac = eta_ref / panelconfig['ta']
    temp_cell = ( (T * I) / (1.0 + frac * panelconfig['D']) ) \
                * ( 1.0 - frac * ( 1.0 - panelconfig['D'] * panelconfig['Tstd'] ) ) + temp
    return temp_cell.rename('cell temperature')

def ElectricModelDC(irradiation, eta_ref, temp_cell, panelconfig):
    irrad_t = irradiation['total tilted']

    irrad_t = irrad_t.where(irrad_t >= panelconfig['threshold'])
    irrad_t = irrad_t.fillna(0.0)
    eta = eta_ref * (1 + panelconfig['D'] * (temp_cell - panelconfig['Tstd']) )
    pdc = irrad_t * eta
    return pdc.rename('DC power')

def ElectricModelAC(pdc, panelconfig):
    pac = pdc * panelconfig['inverter_efficiency']
    return pac.rename('AC power')

def SolarPanelModel(ds, irradiation, panelconfig):

    eta_ref = ReferenceEfficiency(irradiation, panelconfig)
    temp_cell = CellTemperature(ds, irradiation, eta_ref, panelconfig)
    pdc = ElectricModelDC(irradiation, eta_ref, temp_cell, panelconfig)
    pac = ElectricModelAC(pdc, panelconfig)

    solar_panel = xr.Dataset({da.name: da
                              for da in [eta_ref, temp_cell, pdc, pac]})
    return solar_panel
