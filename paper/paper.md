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
  - name: Kiyan Tavangar
    orcid: 0000-0001-6584-6144
    affiliation: "2"
    corresponding: true # is this ok?
affiliations:
  - index: 1
    name: Center for Computational Astrophysics, Flatiron Institute, USA
    ror: 00sekdz59
  - index: 2
    name: Columbia University, USA
    ror: 00hj8s172
date: 31 October 2024
bibliography: paper.bib
# aas-doi:
# aas-journal:
---

# Summary

`stream-membership` is a Python package that provides a flexible framework for creating and fitting probabilistic models of stellar stream properties. It is built on top of `jax` [@jax:18] and `numpyro` [@Bingham:19; @Phan:19] both of which significantly simplify and accelerate the model creation and fitting process compared to previous codes. 

The overarching purpose of `stream-membership` is to serve as an easy-to-use tool to characterize a large number of streams in the Milky Way. Specifically, `stream-membership` is built to characterize known stellar streams, as opposed to disovering new streams. However, it is able to recover new extensions or features of existing streams. The main properties that `stream-membership` is designed to model are: 1) astrometric properties (positions and velocities) and 2) density of stars along the stream. It is written with no specific stream in mind and can be applied to a diverse set of stellar streams. This should allow statistical population-level analyses which will constrain the structure of the Milky Way and the nature of dark matter. 

`stream-membership` is designed to be accessible by researchers of all levels, especially those with a grasp of probability distributions. Additionally, a slight modification of this framework could lead to applications in other scientific fields where a density model is sought or required.

# Statement of Need

Stellar streams are one of the most powerful tools for understanding the structure of galaxies. They form when a bound group of stars (either a globular cluster or a dwarf galaxy) gets stripped of its members as it falls into a larger host galaxy (e.g. the Milky Way). This creates a thin stream of stars along the sky which approximately traces the orbits of its member stars. Precise orbits are critical for constraining the shape of the Milky Way's gravitational potential and understanding the structure of our galaxy's dark matter halo. Furthermore, inhomogeneities in a stream's density along its length are sensitive probes of small-scale structure and are one of the only probes of low-mass dark matter subhalos. Stellar streams are therefore key structures in our ongoing search for dark matter.

The past decade has seen a number of astronomy papers presenting density models of stellar streams [@Erkal:17; @Koposov:19; @Li:21; @Ferguson:22; @Tavangar:22; @Patrick:22; @Starkman:23]. These studies tend to apply their method to one or two streams at a time (see @Patrick:22 for an exception). This has been useful to uncover the complex morphology of multiple Milky Way streams and has led to analysis of individual features in some streams [@Bonaca:19]. However, it is difficultor impossible to constrain the nature of dark matter or global properties of the Milky Way halo with just one or two streams. The real constraining power comes from statistical analyses of dozens of stellar streams, but such studies are extremely rare [@Ibata:24]. In fact, a population-level analysis of inhomogeneities in streams has never been attempted. For many years, this was partly because we did not know of enough Milky Way streams to make such an analysis possible. Now, with more than 140 discovered streams [@Mateu:23], we have the inverse problem: we have too few streams with characterized inhomogeneities.

`stream-membership` is designed to solve this problem by providing a framework with which to characterize streams quickly and easily. It has three major improvements over previous stream modeling techniques. First, it uses `jax` and `numpyro` to simplify and accelerate the model creation and fitting process. `Jax` is a numerical Python library for easy implementation of program transformations in Python and NumPy [@Harris:20]. `Numpyro` is a Python library built atop `jax` for creating probabilistic programs with an easy-to-use NumPy interface.

Second, `stream-membership` is the first package which includes the ability to model offtrack and non-Gaussian features of streams. Streams in a smooth gravitational potential are expected to lie along a single track. However, offtrack features have been observed in a few streams thus far [@Shipp:18; @Price-Whelan:18; @Li:21; @Ferguson:22] and they provide the strongest constraints on past interactions with small-scale Milky Way structure, including dark matter subhalos [@Bonaca:19]. While `stream-membership` is primarily designed for stream characterization as opposed to stream discovery, it is capable of recovering these previously unidentified offtrack and non-Gaussian features in known streams.

Lastly, `stream-membership` is written with no specific stream in mind and is designed to be broadly applicable to many stellar streams. It will enable the rapid generation of dozens of stream density models. We believe the outputs of these models have the potential to create the tightest constraints thus far of Milky Way structure and even the nature of dark matter.

# Acknowledgements
The authors thank Cecilia Mateu, Ana Bonaca, and the wider Community Atlas of Tidal Streams collaboration for helpful discussions.

# References