#Dataset Settings
gebco_path = '/home/vres-climate/data/GEBCO_2014_2D.nc'
cutout_dir = '/home/vres/data/cutouts'
ncep_dir = '/home/vres-climate/data/rda_ucar'
cordex_dir = '/home/vres-climate/data/cordex/RCP8.5'
sarah_dir = '/home/vres-climate/data/sarah_v2'

features = {
    'wind': ['wnd10m', 'wnd50m', 'wnd100m', 'roughness'],
    'influx': ['influx_toa', 'influx_direct', 'influx_diffuse', 'influx', 'albedo'],
    'temperature': ['temperature', 'soil_temperature'],
    'runoff': ['runoff']
}
