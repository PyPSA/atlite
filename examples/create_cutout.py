
# In this example we assume you have set in config.py

# ncep_dir = '/path/to/weather_data/'

# where the files have format e.g.

# 'ncep_dir/{year}{month:0>2}/tmp2m.*.grb2'



import atlite

cutout = atlite.Cutout(name="my_new_europe_cutout",
                       weather_dataset="ncep",
                       lons=slice(-12.18798349, 41.56244222),
                       lats=slice(71.65648314, 33.56459975),
                       years=slice(2012, 2013),
                       cutout_dir='/home/vres/data/cutouts')
