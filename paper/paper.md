---
title: 'Atlite: A Lightweight Python Package for Calculating Renewable Power Potentials and Time-Series'
tags:
  - Python
  - energy systems
authors:
  - name: Fabian Hofmann
    orcid: 0000-0002-6604-5450
    affiliation: "1" # (Multiple affiliations must be quoted)
    # add your names here
affiliations:
 - name: Frankfurt Institute for Advanced Studies
   index: 1
 - name: Institution Name
   index: 2
 - name: Institution Name
   index: 3
date: 01 March 2021
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: xxxx <- The name of the AAS journal.
---

<!-- See https://joss.readthedocs.io/en/latest/submitting.html for all details -->
<!-- compile with 

pandoc --citeproc -s paper.md -o paper.pdf  

 -->

# Summary

Renewable energy sources build the backbone of future energy systems. The urgent need to reduce green house gas emissions and the improving cost-efficiency of renewable technologies, pave the way for the society to invest into a fast and ongoing decarbonization. Yet, a renewable energy system is highly dependent on the weather, on the production side with technologies like solar PV or wind turbines, as well as on the demand side like heating or cooling. Atlite is an open python software package for retrieving reanalysis weather data and converting it to time series and potentials for energy systems. Based on detailed mathematical models, it simulates the power output of wind turbines, solar photo-voltaic panels, solar-thermal collectors, run-of-river power plants and hydro-electrical dams as well as heating demand degree days and heat pump coefficients of performance.


# Statement of need

<!-- In energy system modelling, the research field that simulates feasible and cost-efficient energy systems, precalculated time-series for different technologies are of increasing importance.  -->

<!-- context of atlite -->
Atlite was initially build as a light-weight alternative to the Danish REatlas [@andresen_validation_2015]. Downsides of REatlas.  What other packages exist?  What's their up/donwside? Why is atlite necessary? 

<!-- software/packages and implementation -->
Atlite highly relies on the python packages *xarray* and  *dask* which allows for parallelized and memory-efficient processing of large data sets. 


# Basic Concept

<!-- Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% } -->

## The Cutout Class

## Conversion Functions

## Land Use Availability


# Related Research 

pypsa-eur, pypsa-eur-sec, others? 




# Availability

The Atlite package is available via the open source package management systems *Python Package Index* and *Conda* [@anaconda], available and tested for Linux, Mac and Windows systems. Atlite is released and licensed under the GPLv3. Source code at [Github](https://github.com/PyPSA/atlite). CI, docs ...


# Acknowledgements


# References