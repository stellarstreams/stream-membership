Overview
============

Welcome to ``stream-membership``, an empirical modeling package for stellar streams!

``stream-membership`` is a Python package that provides a flexible framework for modeling the density of a stellar stream as a function of position along the stream track.
Its two main innovations over existing stream modeling techniques are the backend code it uses (jax and numpyro) 
and the ability to model the offtrack and non-Gaussian features we are most interested in.
It is designed for stream characterization as opposed to stream discovery, although it is capable of recovering new features in known streams.
In principle it is flexible enough to be used for any stream, but it is most useful for streams with sky position and proper motion information (i.e. streams in Gaia).

``stream-membership`` is most useful for:

#. Creating a density model of a stellar stream
#. Obtaining posterior distributions for stream properties (e.g. density, track, and width in position and proper motion space)
#. Generating stream membership probabilities
#. Discovering and characterizing offtrack or non-Gaussian features in a stream

Stellar streams are one of the most powerful tools for understanding the formation and evolution of the Milky Way. 
Stream tracks and proper motions provide constraints on the gravitational potential of the Milky Way halo, 
including the presence of large substructure such as the LMC (`Shipp et al (2021)`_).
Simultaneously, inhomogeneities in a stream's density along its track are one of the most sensitive probes of small-scale structure in the Milky Way halo, 
and may provide one of the keys to understanding the nature of dark matter 
(see `Bonaca & Price-Whelan (2024)`_ for a recent review).

Of particular interest are off-track or non-Gaussian stream features because they can probably only be formed via an interaction 
with an object such as a dark matter subhalo, while stream over- and under-densities can have a multitude of causes.
We know of a couple such features already (e.g. the spur in GD-1, possible spur in Jet, kink in ATLAS-Aliqa Uma, Jhelum's double track), 
but the real constraining power will come from statistical analyses with a much larger sample of features.
Understandably however, such features are difficult to find because by definition they lie away from the main stream track, which is where most of the stars are.
Our goal in developing ``stream-membership`` is to create stream density models which incorporate offtrack features.

.. _Shipp et al (2021): https://ui.adsabs.harvard.edu/abs/2021ApJ...923..149S/abstract
.. _Bonaca & Price-Whelan (2024): https://ui.adsabs.harvard.edu/abs/2024arXiv240519410B/abstract