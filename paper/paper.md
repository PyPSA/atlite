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

# Summary

The increasing importance of renewable power sources...


# Statement of need


Atlite is an open python software package for retrieving reanalysis weather data and converting it to time series and potentials for energy systems. Based on detailed mathematical models, it simulates the power output of wind turbines, solar photo-voltaic panels, solar-thermal collectors, run-of-river power plants and hydro-electrical dams. These may be distributed over large areas  Atlite highly relies on the python packages *xarray* and  *dask* which allows for parallelized and memory-efficient processing of large data sets. In this paper we introduce into the basic concepts of atlite and 

* for whom is created? 
* where is it already used?


# Basic Concept

## The Cutout class

## The Conversion function



# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References