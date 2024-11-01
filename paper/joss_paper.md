---
title: 'stream-membership: A Python package for empirical density modeling of stellar streams'
tags:
  - Python
  - astronomy
  - galactic dynamics
  - milky way
  - stellar streams
authors:
  - name: Adrian M. Price-Whelan
    orcid: 0000-0003-0872-7098
    affiliation: "1"
    equal-contrib: true
  - name: Kiyan Tavangar
    orcid: 0000-0001-6584-6144
    affiliation: "2"
    equal-contrib: true
    corresponding: true # is this ok?
affiliations:
  - index: 1
    name: Center for Computational Astrophysics, Flatiron Institute, USA
    ror: 00sekdz59
  - index: 2
    name: Columbia University, USA
    ror: 00hj8s172
date: 31 October 2024
bibliography: joss_paper.bib
aas-doi:
aas-journal:
---


# Statement of Need

Stellar streams are one of the most powerful tools for understanding the formation and evolution of galaxies. They form when a bound group of stars (either a globular cluster or a dwarf galaxy) gets stripped of its members as it falls into a larger host galaxy (e.g. the Milky Way). This creates a thin stream of stars along the sky which approximately traces the orbits of its member stars. Precise orbits are critical for constraining the shape of the Milky Way's gravitational potential and understanding the structure of our galaxy's dark matter halo. Furthermore, inhomogeneities in a stream's density along its track are sensitive probes of small-scale structure in the Milky Way halo, and one of the only probes of low mass dark matter subhalos. Stellar streams are therefore key structures in our ongoing search for the nature of dark matter. However, to use their considerable constraining power to maximum effect, we need empirical models of as many streams as possible. We present `stream-membership`, a Python package that provides a flexible framework for modeling the density of a stellar stream as a function of position along the stream track. `stream-membership` is designed to be accessible by researchers of all levels, especially those with a grasp of probability distributions

# Summary

The past decade has seen a number of astronomy papers presenting density models of stellar streams `[@Erkal:17; @Koposov:19; @Li:21; @Ferguson:22; @Tavangar:22; @Patrick:22; @Starkman:24]`.
`stream-membership` has three major improvements over most existing stream modeling techniques. First, it is built on top of `jax` `[@jax:18]` and `numpyro` `[@Bingham:19; @Phan:19]`. `Jax` is a numerical computing Python library which easily implements transformation of programs in Python and NumPy `[@numpy:20]`. `Numpyro` is a Python library built atop `jax` for creating probabilistic programs with an easy-to-use NumPy interface. These two libraries significantly simplify and accelerate the model creation and fitting process. 

Second, `stream-membership` is the first package which includes the ability to model offtrack and non-Gaussian features of streams. Such features have been observed in a few streams thus far `[@Shipp:18; @Price-Whelan:18; @Li:21; @Ferguson:22]`. They provide the strongest constraints on past interactions with other Milky Way structure, including dark matter subhalos `[@Bonaca:19]`. While `stream-membership` is primarily designed for stream characterization as opposed to stream discovery, it is capable of recovering these previously unidentified offtrack and non-Gaussian features in known streams. 

Lastly, `stream-membership` is written with no specific stream in mind and is designed to be broadly applicable to many stellar streams. It will enable the rapid generation of dozens of stream density models. Amassing a census of such models will allow statistical analyses of stellar stream structure, which is the most promising way to constrain the Milky Way's structure and dark matter models.

# References

