import numpy as np
import pandas as pd
import xarray as xr

def DiffuseHorizontalIrrad(ds, solar_position, settings):
    influx = ds['influx']
    altitude = solar_position['altitude']
    extra = solar_position['extra']
    REatlas = settings['simulate REatlas']
    model = settings['clearsky model']

    # Reindl 1990 clearsky model

    sinaltitude = np.sin( altitude )
    k = influx / (extra*sinaltitude)
    k = k.rename('clearsky index')

    # Simulates the REatlas routine
    if REatlas is True:
        condlist = [( k > 0.0 ) & ( k <= 0.3 ),
                    ( k > 0.3 ) & ( k <= 0.78 ), # XXX in Reindl et al. the = is in condition three
                    ( k > 0.78 )]

        temp1 = np.fmin( 1.0, 1.020-0.254*k+0.0123*np.sin(altitude) )
        temp2 = np.fmin( 0.97, np.fmax( 0.1, 0.1400-1.749*k+0.177*np.sin(altitude) ) ) # XXX in Reindl et al. it is 1.400 not 0.1400
        temp3 = np.fmax( 0.1, 0.486*k-0.182*np.sin(altitude) )

        fraction1 = temp1*condlist[0]
        fraction2 = temp2*condlist[1]
        fraction3 = temp3*condlist[2]

    if REatlas is False:
        # Simple Reindl model without ambient air temperature and relative humidity
        if model is 'Simple':
            condlist = [( k > 0.0 ) & ( k <= 0.3 ),
                        ( k > 0.3 ) & ( k < 0.78 ),
                        ( k >= 0.78 )]

            temp1 = np.fmin( 1.0, 1.020-0.254*k+0.0123*np.sin(altitude) )
            temp2 = np.fmin( 0.97, np.fmax( 0.1, 1.400-1.749*k+0.177*np.sin(altitude) ) )
            temp3 = np.fmax( 0.1, 0.486*k-0.182*np.sin(altitude) )

            fraction1 = temp1*condlist[0]
            fraction2 = temp2*condlist[1]
            fraction3 = temp3*condlist[2]

        # Enhanced Reindl model with ambient air temperature and relative humidity
        if model is 'Enhanced':
            t = ds['temperature']
            rh = ds['relative humidity']

            condlist = [( k > 0.0 ) & ( k <= 0.3 ),
                        ( k > 0.3 ) & ( k < 0.78 ),
                        ( k >= 0.78 )]

            temp1 = np.fmin( 1.0, 1.000-0.232*k+0.0239*np.sin(altitude)-0.000682*T+0.0195*rh )
            temp2 = np.fmin( 0.97, np.fmax( 0.1, 1.329-1.716*k+0.267*np.sin(altitude)-0.00357*T+0.106*rh ) )
            temp3 = np.fmax( 0.1, 0.426*k-0.256*np.sin(altitude)+0.00349*T+0.0734*rh )

            fraction1 = temp1*condlist[0]
            fraction2 = temp2*condlist[1]
            fraction3 = temp3*condlist[2]

    # Set diffuse fraction to one when the sun isn't up
    threshold = 5.0
    fraction = (fraction1 + fraction2 + fraction3).where(np.sin(altitude) >= np.sin(np.deg2rad(threshold))).fillna(1.0)
    fraction = fraction.rename('fraction index')

    diffuse = influx*fraction

    # This avoids division by zero and other bad stuff when the sun isn't up
    capalt = ( np.sin(altitude) >= np.sin(np.deg2rad(threshold)) )
    diffuse = diffuse.where(capalt).fillna(0.0)
    return diffuse.rename('diffuse horizontal')

def DirectHorizontalIrrad(ds, solar_position, diffuse):
    influx = ds['influx']
    altitude = solar_position['altitude']

    beam = (influx - diffuse)

    # This avoids division by zero and other bad stuff when the sun isn't up
    threshold = 5.0
    capalt = ( np.sin(altitude) >= np.sin(np.deg2rad(threshold)) )
    beam = beam.where(capalt).fillna(0.0)
    return beam.rename('direct horizontal')

def TiltedDiffuseIrrad(ds, solar_position, diffuse, beam, settings):
    influx = ds['influx']
    altitude = solar_position['altitude']
    incidence = solar_position['incidence']
    extra = solar_position['extra']
    surface_slope = np.deg2rad(settings['surface slope'])

    # Hay-Davies Model

    f = np.sqrt( beam/influx )
    f = f.fillna(0.0)
    f = f.rename('brightening factor')

    A = beam / (extra * np.sin(altitude))
    A = A.fillna(0.0)
    A = A.rename('anisotropy factor')

    sinaltitude = np.sin(altitude)
    cosincidence = np.cos(incidence)

    R_b = cosincidence/sinaltitude

    R_b = R_b.fillna(0.0)
    R_b = R_b.rename('geometric factor diffuse')

    diffuse_t = ( ( 1.0 - A ) * ( (1 + np.cos(surface_slope)) / 2.0 ) \
            * ( 1.0 + f * np.power(np.sin(surface_slope/2.0), 3) ) \
            + A * R_b ) \
            * diffuse

    # This avoids division by zero and other bad stuff when the sun isn't up
    threshold = 5.0
    capalt = ( np.sin(altitude) >= np.sin(np.deg2rad(threshold)) )
    diffuse_t = diffuse_t.where(capalt).fillna(0.0)
    return diffuse_t.rename('diffuse tilted')

def TiltedDirectIrrad(solar_position, beam):
    altitude = solar_position['altitude']
    incidence = solar_position['incidence']

    sinaltitude = np.sin(altitude)
    cosincidence = np.cos(incidence)

    R_b = cosincidence/sinaltitude

    R_b = R_b.fillna(0.0)
    R_b = R_b.rename('geometric factor beam')

    beam_t = R_b * beam

    # This avoids division by zero and other bad stuff when the sun isn't up
    threshold = 5.0
    capalt = ( np.sin(altitude) >= np.sin(np.deg2rad(threshold)) )
    beam_t = beam_t.where(capalt).fillna(0.0)
    return beam_t.rename('direct tilted')

# To be checked (against REatlas)
def TiltedGroundIrrad(ds, solar_position, settings):
    influx = ds['influx']
    outflux = ds['outflux']
    altitude = solar_position['altitude']
    surface_slope = np.deg2rad(settings['surface slope'])

    #some inf and nan handling
    cond1, cond2 = influx > 0.0, outflux > 0.0
    albedo = (outflux / influx) * cond1 * cond2
    albedo = albedo.fillna(0.0)

    #set cap at 1.0
    albedo = albedo.where(albedo <= 1.0)
    albedo = albedo.fillna(1.0)
    albedo = albedo.rename('albedo')

    ground_t = influx * albedo * ( 1.0 - np.cos(surface_slope)) / 2.0

    # This avoids division by zero and other bad stuff when the sun isn't up
    threshold = 5.0
    capalt = ( np.sin(altitude) >= np.sin(np.deg2rad(threshold)) )
    ground_t = ground_t.where(capalt).fillna(0.0)
    return ground_t.rename('ground tilted')

def TiltedTotalIrrad(diffuse_t, beam_t, ground_t):
    irradiation_t = diffuse_t + beam_t + ground_t
    return irradiation_t.rename('total tilted')

def TiltedTotalIrradiation(ds, solar_position, settings):

    diffuse = DiffuseHorizontalIrrad(ds, solar_position, settings)
    beam = DirectHorizontalIrrad(ds, solar_position, diffuse)

    diffuse_t = TiltedDiffuseIrrad(ds, solar_position, diffuse, beam, settings)
    beam_t = TiltedDirectIrrad(solar_position, beam)
    ground_t = TiltedGroundIrrad(ds, solar_position, settings)

    total_t = TiltedTotalIrrad(diffuse_t, beam_t, ground_t)

    irradiation = xr.Dataset({da.name: da
                              for da in [diffuse, diffuse_t,
                                         beam, beam_t,
                                         ground_t,
                                         total_t]})
    return irradiation
