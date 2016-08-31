


# Set up config


# Philosophy

Atlite assumes you have a global dataset over many years, but are only interested in a
portion of the globe (e.g. Europe) for a shorter time.

Atlite therefore first makes a "cutout" - a geographical rectangle and
a selection of times, e.g. all hours in 2011 and 2012.


Once this cutout has been created, you can use a sparse matrix to take
linear combinations of time series based on the weather data,
e.g. wind generation time series or heat demand.
