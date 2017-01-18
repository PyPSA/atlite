
# This script uses the FIAS vresutils to built heat demand for each
# European country based on its population distribution (at NUTS 3
# resolution)


import sys, os

from vresutils import shapes as vshapes, mapping as vmapping, transfer as vtransfer
from load import europe

import pandas as pd
import numpy as np

import scipy as sp

import atlite

cutout = atlite.Cutout('europe-2011-2014')

#list of grid cells
grid_cells = cutout.grid_cells()
print(len(grid_cells))


#pd.Series nuts3 code -> 2-letter country codes
mapping = vmapping.countries_to_nuts3()

countries = mapping.value_counts().index.sort_values()


#pd.Series
gdp,pop = europe.gdppop_nuts3()

#pd.Series nuts3 code -> polygon
nuts3 = pd.Series(vshapes.nuts3(tolerance=None, minarea=0.))

#2014 populations

pop["CH040"] = pop["CH04"]
pop["CH070"] = pop["CH07"]
#Montenegro
pop["ME000"] = 650
pop["AL1"] = 2893
pop["BA1"] = 3871
pop["RS1"] = 7210

pop[nuts3.index][pd.isnull(pop[nuts3.index])]

pop.groupby(mapping).sum()

#takes 10 minutes
pop_map = pd.DataFrame()

for country in countries:
    print(country)
    country_nuts = mapping.index[mapping == country]
    trans_matrix = cutout.indicatormatrix(np.asarray(nuts3[country_nuts]))
    #CH has missing bits
    country_pop = pop[country_nuts].fillna(0.)
    pop_map[country] = np.array(trans_matrix.multiply(np.asarray(country_pop)).sum(axis=1))[:,0]

print(pop_map.sum())

#%%time
pop_matrix = sp.sparse.csr_matrix(pop_map.T)
index = pd.Index(countries,name="countries")

#%%time
hd = cutout.heat_demand(matrix=pop_matrix, index=index)

hd.T.to_pandas().to_pickle("heat_demand.pkl")
