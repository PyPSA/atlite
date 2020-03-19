
# This script creates the heat demand for a city

# It is assumed the city is half on grid cell 43 and half on grid cell
# 44

# It assumes you have already created a cutout for Europe called
# "europe-2011-01.nc" in the local directory



from scipy import sparse


import atlite


cutout = atlite.Cutout(path='europe-2011-01.nc')

# A sparse matrix describing the city relative to the grid coordinates
matrix = sparse.csr_matrix(([0.5,0.5], ([0,1],[43,44])), shape=(2, len(cutout.grid_coordinates())))

print(matrix)

hd = cutout.heat_demand(matrix)

print(hd)
