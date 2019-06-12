try:
    import cdsapi
    has_cdsapi = True
except ImportError:
    has_cdsapi = False

def _noisy_unlink(path):
    logger.info(f"Deleting file {path}")
    os.unlink(path)

def _retrieve_data(product, chunks=None, **updates):
    """Download data like ERA5 from the Climate Data Store (CDS)"""

    if not has_cdsapi:
        raise RuntimeError(
            "Need installed and configured cdsapi python package available from "
            "https://cds.climate.copernicus.eu/api-how-to"
        )

    # Default request
    request = {
        'product_type':'reanalysis',
        'format':'netcdf',
        'day': list(range(1, 31+1)),
        'time':[
            '00:00','01:00','02:00','03:00','04:00','05:00',
            '06:00','07:00','08:00','09:00','10:00','11:00',
            '12:00','13:00','14:00','15:00','16:00','17:00',
            '18:00','19:00','20:00','21:00','22:00','23:00'
        ],
        'month': list(range(1, 12+1)),
        # 'area': [50, -1, 49, 1], # North, West, South, East. Default: global
        # 'grid': [0.25, 0.25], # Latitude/longitude grid: east-west (longitude) and north-south resolution (latitude). Default: 0.25 x 0.25
    }
    request.update(updates)

    assert {'year', 'month', 'variable'}.issubset(request), "Need to specify at least 'variable', 'year' and 'month'"

    result = cdsapi.Client().retrieve(
        product,
        request
    )

    fd, target = mkstemp(suffix='.nc')
    os.close(fd)

    logger.info("Downloading request for {} variables to {}".format(len(request['variable']), target))

    result.download(target)

    ds = xr.open_dataset(target, chunks=chunks or {})
    weakref.finalize(ds._file_obj, _noisy_unlink, target)

    return ds

def get_data_gebco_height(xs, ys, gebco_fn=None):
    # gebco bathymetry heights for underwater
    tmpdir = tempfile.mkdtemp()
    cornersc = np.array(((xs[0], ys[0]), (xs[-1], ys[-1])))
    minc = np.minimum(*cornersc)
    maxc = np.maximum(*cornersc)
    span = (maxc - minc)/(np.asarray((len(xs), len(ys)))-1)
    minx, miny = minc - span/2.
    maxx, maxy = maxc + span/2.

    fd, target = mkstemp(suffix='.nc')
    os.close(fd)

    delete_target = True

    try:
        ret = subprocess.call(['gdalwarp', '-of', 'NETCDF',
                               '-ts', str(len(xs)), str(len(ys)),
                               '-te', str(minx), str(miny), str(maxx), str(maxy),
                               '-r', 'average',
                               gebco_fn, target])
        assert ret == 0, "gdalwarp was not able to resample gebco"
    except OSError:
        logger.warning("gdalwarp was not found for resampling gebco. "
                       "Next-neighbour interpolation will be used instead!")
        target = gebco_fn
        delete_target = False

    with xr.open_dataset(target) as ds_gebco:
        height = (ds_gebco.rename({'lon': 'x', 'lat': 'y', 'Band1': 'height'})
                          .reindex(x=xs, y=ys, method='nearest')
                          .load()['height'])
    shutil.rmtree(tmpdir)
    return height
