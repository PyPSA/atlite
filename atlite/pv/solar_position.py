import numpy as np
import pandas as pd
import xarray as xr

def DayNumber(ds):
    time = ds['time']

    day_number = pd.to_datetime(time.values).dayofyear - 1
    return xr.DataArray(day_number, coords=[time], dims=['time'], name='day number')

#Equation of Time [min]
def EquationOfTime(day_number):
    B = 2*np.pi*(day_number-81.0)/365.0
    ET = 9.87*np.sin(2*B)-7.53*np.cos(B)-1.5*np.sin(B)
    return ET.rename('equation of time')

#Apparent Solar Time [h]
def ApparentSolarTime(ds, equation_of_time, standard_lon=0.0, daylight_saving=0.0):
    local_standard_time = ds['time']
    local_lon = ds['lon']

    local_standard_time = pd.to_datetime(local_standard_time.values)
    local_standard_time = local_standard_time.hour*60. + local_standard_time.minute
    AST = local_standard_time + equation_of_time + 4.0*(local_lon - standard_lon) - daylight_saving
    AST = AST/60.0
    return AST.rename('apparent solar time')

#Hour Angle [rad]
def HourAngle(apparent_solar_time):
    h = (apparent_solar_time - 12.0)*15.0
    h = np.deg2rad(h)
    return h.rename('hour angle')

#Declination [rad]
def Declination(day_number):
    gamma = 2*np.pi*day_number/365.0
    delta = 0.006918-0.399912*np.cos(gamma)+0.070257*np.sin(gamma) \
            -0.006758*np.cos(2*gamma) + 0.000907*np.sin(2*gamma) \
            -0.002697*np.cos(3*gamma) + 0.00148*np.sin(3*gamma)
    return delta.rename('declination')

#Solar Altitude [rad]
def SolarAltitudeAngle(ds, declination, hour_angle):
    lat = ds['lat']
    lat = np.deg2rad(lat)

    alpha = np.arcsin( np.sin(lat)*np.sin(declination) + np.cos(lat)*np.cos(declination)*np.cos(hour_angle) )
    return alpha.rename('altitude')

#Solar Zenith Angle [rad]
def SolarZenithAngle(ds, declination, hour_angle):
    lat = ds['lat']
    lat = np.deg2rad(lat)

    phi = np.arccos( np.sin(lat)*np.sin(declination) + np.cos(lat)*np.cos(declination)*np.cos(hour_angle) )
    return phi.rename('zenith')

#Solar Azimuth Angle [rad]
#To be checked (against REatlas)
def SolarAzimuthAngle(ds, apparent_solar_time, declination, hour_angle, altitude):
    lat = ds['lat']
    lat = np.deg2rad(lat)

    z = np.arcsin( np.cos(declination)*np.sin(hour_angle) / np.cos(altitude) )

    #Sun might be behind E-W line. If so, the above formula must be corrected:
    before_EW_line = ( np.cos(declination) > np.tan(declination)/np.tan(lat) )
    behind_EW_line = ( np.cos(declination) <= np.tan(declination)/np.tan(lat) )

    morning = ( apparent_solar_time < 12 )
    afternoon = ( apparent_solar_time > 12)
    noon = ( apparent_solar_time == 12 )

    z_before = z.where( before_EW_line )
    z_before = z_before.fillna(0.0)

    z_temp = z.where( behind_EW_line )

    z_morning = -np.pi + np.abs( z_temp ) * morning
    z_afternoon = np.pi - z_temp * afternoon
    z_noon = 0.0 * z_temp * noon

    z_morning = z_morning.fillna(0.0)
    z_afternoon = z_afternoon.fillna(0.0)
    z_noon = z_noon.fillna(0.0)

    z_behind = z_morning + z_afternoon + z_noon

    z = z_before + z_behind
    return z.rename('azimuth')

#Incidence Angle [rad]
def SolarIncidenceAngle(ds, declination, hour_angle, altitude, settings):
    lat = ds['lat']
    lat = np.deg2rad(lat)
    surface_slope = np.deg2rad(settings['surface slope'])
    surface_azimuth = np.deg2rad(settings['surface azimuth'])

    theta = np.arccos( np.sin(lat)*np.sin(declination)*np.cos(surface_slope) \
                        - np.cos(lat)*np.sin(declination)*np.sin(surface_slope)*np.cos(surface_azimuth) \
                        + np.cos(lat)*np.cos(declination)*np.cos(hour_angle)*np.cos(surface_slope) \
                        + np.sin(lat)*np.cos(declination)*np.cos(hour_angle)*np.sin(surface_slope)*np.cos(surface_azimuth) \
                        + np.cos(declination)*np.sin(hour_angle)*np.sin(surface_slope)*np.sin(surface_azimuth) )
    return theta.rename('incidence')

def ExtraterrestrialRadiation(day_number, solar_constant=1366.1):
    gamma = 2*np.pi*day_number/365.0
    extra = solar_constant*(1+0.033*np.cos(gamma))
    return extra.rename('extra')

def SolarPosition(ds, settings):

    day_number = DayNumber(ds)
    EoT = EquationOfTime(day_number)
    AST = ApparentSolarTime(ds, EoT)
    h = HourAngle(AST)
    dec = Declination(day_number)

    altitude = SolarAltitudeAngle(ds, dec, h)
    zenith = SolarZenithAngle(ds, dec, h)
    azimuth = SolarAzimuthAngle(ds, AST, dec, h, altitude)
    incidence = SolarIncidenceAngle(ds, dec, h, altitude, settings)
    extra = ExtraterrestrialRadiation(day_number)

    solar_position = xr.Dataset({da.name: da
                                 for da in [altitude, zenith, azimuth, incidence, extra]})
    return solar_position
