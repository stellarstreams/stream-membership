Distributions
=============

If you would like to add a new distribution, please open a pull request.

List of distributions
---------------------

:class:`NormalSpline`:
Represents a Normal distribution where the loc and (log)scale parameters are
controlled by splines that are evaluated at some other parameter values x. In
other words, this distribution is conditional on x.

:class:`TruncatedNormalSpline`:
Equivalent to :class:`NormalSpline`, but for a truncated Normal distribution

:class:`Normal1DSplineMixture`:
Represents a mixture of Normal distributions where the loc and (log)scale parameters are
controlled by splines that are evaluated at some other parameter values x. Takes a
``mixing_distribution`` parameter which specifies the relative weights of the mixture components.

:class:`TruncatedNormal1DSplineMixture`:
Equivalent to :class:`Normal1DSplineMixture`, but for multiple truncated Normal distributions

:class:`IndependentGMM`:
Represents a Gaussian Mixture Model where the components are fixed to their input locations
and there are no covariances (but each dimension can have different scales / standard deviations).

:class:`DirichletSpline`:
Represents a Dirichlet distribution where the concentration parameters are
controlled by splines that are evaluated at some other parameter values x.

:class:`ConcatenatedDistributions`:
Represents a multi-dimensional distribution that is the concatenation of multiple distributions.
