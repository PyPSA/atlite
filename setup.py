from __future__ import absolute_import

from setuptools import setup, find_packages
from codecs import open
import six

with open('README.rst', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='atlite',
    version='0.0.2',
    author='Gorm Andresen (Aarhus University), Jonas Hoersch (FIAS), Tom Brown (FIAS)',
    author_email='hoersch@fias.uni-frankfurt.de',
    description='Light-weight version of Aarhus RE Atlas for converting weather data to power systems data',
    long_description=long_description,
    url='https://github.com/FRESNA/atlite',
    license='GPLv3',
    packages=find_packages(exclude=['doc', 'test']),
    include_package_data=True,
    install_requires=['numpy',
                      'scipy',
                      'pandas>=0.22',
                      'bottleneck',
                      'numexpr',
                      'xarray>=0.11.2',
                      'dask>=0.18.0',
                      'rasterio',
                      'shapely',
                      'progressbar2'
                      'geopandas'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
    ])
