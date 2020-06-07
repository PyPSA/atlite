from __future__ import absolute_import

from setuptools import setup, find_packages

with open('README.rst', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='atlite',
    author='Jonas Hoersch (FIAS), Tom Brown (FIAS), Gorm Andresen (Aarhus University)',
    author_email='jonas.hoersch@posteo.de',
    description='Library for fetching and converting weather data to power systems data',
    long_description=long_description,
    url='https://github.com/PyPSA/atlite',
    license='GPLv3',
    packages=find_packages(exclude=['doc', 'test']),
    include_package_data=True,
    use_scm_version={'write_to': 'atlite/version.py'},
    setup_requires=['setuptools_scm'],
    install_requires=['numpy',
                      'scipy',
                      'pandas>=0.22',
                      'bottleneck',
                      'numexpr',
                      'xarray>=0.11.2',
                      'dask>=0.18.0',
                      'rasterio',
                      'shapely',
                      'progressbar2',
                      'geopandas'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
    ])
