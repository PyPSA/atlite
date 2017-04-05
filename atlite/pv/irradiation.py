import numpy as np
import pandas as pd
import xarray as xr

def DiffuseHorizontalIrrad(ds, solar_position, clearsky_model):
    influx = ds['influx']
    sinaltitude = np.sin(solar_position['altitude'])
    extra = solar_position['extra']

    if clearsky_model is None:
        clearsky_model = ('enhanced'
                          if 'temperature' in ds and 'humidity' in ds
                          else 'simple')

    # Reindl 1990 clearsky model

    k = influx / (extra*sinaltitude) # clearsky index
    # k = k.rename('clearsky index')

    if clearsky_model == 'simple':
        # Simple Reindl model without ambient air temperature and
        # relative humidity
        fraction = (((k > 0.0) & (k <= 0.3)) *
                    np.fmin( 1.0, 1.020-0.254*k+0.0123*sinaltitude )
                    +
                    ((k > 0.3) & (k < 0.78)) *
                    np.fmin( 0.97, np.fmax( 0.1, 1.400-1.749*k+0.177*sinaltitude ) )
                    +
                    (k >= 0.78) *
                    np.fmax( 0.1, 0.486*k-0.182*sinaltitude ))
    elif clearsky_model == 'enhanced':
        # Enhanced Reindl model with ambient air temperature and relative humidity
        T = ds['temperature']
        rh = ds['humidity']

        fraction = (((k > 0.0) & (k <= 0.3)) *
                    np.fmin( 1.0, 1.000-0.232*k+0.0239*sinaltitude-0.000682*T+0.0195*rh )
                    +
                    ((k > 0.3) & (k < 0.78)) *
                    np.fmin( 0.97, np.fmax( 0.1, 1.329-1.716*k+0.267*sinaltitude-0.00357*T+0.106*rh ) )
                    +
                    (k >= 0.78) *
                    np.fmax( 0.1, 0.426*k-0.256*sinaltitude+0.00349*T+0.0734*rh ))
    elif clearsky_model == 'reatlas':
        # Is supposed to implement the simple Reindl model, but
        # deviates in two locations (see below).

        fraction = (((k > 0.0) & (k <= 0.3)) *
                    np.fmin( 1.0, 1.020-0.254*k+0.0123*sinaltitude )
                    +
                    ((k > 0.3) & (k <= 0.78)) * # XXX in Reindl et al. the = is in condition three
                    np.fmin( 0.97, np.fmax( 0.1, 0.1400-1.749*k+0.177*sinaltitude ) ) # XXX in Reindl et al. it is 1.400 not 0.1400
                    +
                    (k > 0.78) *
                    np.fmax( 0.1, 0.486*k-0.182*sinaltitude ))
    else:
        raise ArgumentError("`clearsky model` must be chosen from 'simple', 'enhanced' and 'reatlas'")

    # Set diffuse fraction to one when the sun isn't up
    # fraction = fraction.where(sinaltitude >= np.sin(np.deg2rad(threshold))).fillna(1.0)
    # fraction = fraction.rename('fraction index')

    return (influx * fraction).rename('diffuse horizontal')

def DirectHorizontalIrrad(ds, solar_position, diffuse):
    influx = ds['influx']
    sinaltitude = np.sin(solar_position['altitude'])

    return (influx - diffuse).rename('direct horizontal')

def TiltedDiffuseIrrad(ds, solar_position, surface_orientation, diffuse, beam):
    influx = ds['influx']
    sinaltitude = np.sin(solar_position['altitude'])
    extra = solar_position['extra']
    cosincidence = np.cos(surface_orientation['incidence'])
    surface_slope = surface_orientation['slope']

    # Hay-Davies Model

    f = np.sqrt(beam/influx).fillna(0.0)
    # f = f.rename('brightening factor')

    A = (beam / (extra * sinaltitude)).fillna(0.0)
    # A = A.rename('anisotropy factor')

    R_b = (cosincidence/sinaltitude).fillna(0.0)
    # R_b = R_b.rename('geometric factor diffuse')

    diffuse_t = ((1.0 - A) * ((1 + np.cos(surface_slope)) / 2.0) *
                 (1.0 + f * np.sin(surface_slope/2.0)**3)
                 + A * R_b ) * diffuse

    return diffuse_t.rename('diffuse tilted')

def TiltedDirectIrrad(solar_position, surface_orientation, beam):
    sinaltitude = np.sin(solar_position['altitude'])
    cosincidence = np.cos(surface_orientation['incidence'])

    R_b = (cosincidence/sinaltitude).fillna(0.0)
    # R_b = R_b.rename('geometric factor beam')

    return (R_b * beam).rename('direct tilted')

def TiltedGroundIrrad(ds, solar_position, surface_orientation):
    influx = ds['influx']
    outflux = ds['outflux']
    surface_slope = surface_orientation['slope']

    albedo = (outflux / influx)
    # fixup albedo: - only positive fluxes and - cap at 1.0
    # note: REatlas does not do the fixup
    albedo = albedo.where((influx > 0.0) & (outflux > 0.0)).fillna(0.)
    albedo.values[albedo.values > 1.0] = 1.0
    # albedo = albedo.rename('albedo')

    ground_t = influx * albedo * (1.0 - np.cos(surface_slope)) / 2.0
    return ground_t.rename('ground tilted')

def TiltedIrradiation(ds, solar_position, surface_orientation, clearsky_model, altitude_threshold=5.):
    diffuse = DiffuseHorizontalIrrad(ds, solar_position, clearsky_model)
    beam = DirectHorizontalIrrad(ds, solar_position, diffuse)

    diffuse_t = TiltedDiffuseIrrad(ds, solar_position, surface_orientation, diffuse, beam)
    beam_t = TiltedDirectIrrad(solar_position, surface_orientation, beam)
    ground_t = TiltedGroundIrrad(ds, solar_position, surface_orientation)

    total_t = (diffuse_t + beam_t + ground_t).rename('total tilted')

    cap_alt = np.sin(solar_position['altitude']) >= np.sin(np.deg2rad(altitude_threshold))
    irradiation = xr.Dataset({
        da.name: da.where(cap_alt).fillna(0.)
        for da in [diffuse, diffuse_t, beam, beam_t, ground_t, total_t]
    })
    return irradiation
