import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

from stream_membership.distributions import (
    Normal1DSplineMixture,
    TruncatedNormal1DSplineMixture,
)


@pytest.mark.parametrize(
    ("SplineClass", "extra_kwargs"),
    [(Normal1DSplineMixture, {}), (TruncatedNormal1DSplineMixture, {"high": 16.0})],
)
def test_mixture(SplineClass, extra_kwargs):
    knots = np.linspace(0, 10, 16)
    loc_vals = np.stack((0.5 * knots + 5.0, -0.1 * knots + 15.5), axis=0)
    ln_scale_vals = np.stack((1e-2 * knots, -0.2 * knots + 1.0), axis=0)
    x = np.linspace(1, 7, 128)

    _p1 = InterpolatedUnivariateSpline(knots, np.linspace(0.2, 0.6, knots.size))(x)
    probs = np.stack((_p1, 1 - _p1), axis=-1)

    mix = SplineClass(
        mixing_distribution=dist.Categorical(probs=probs),
        loc_vals=loc_vals,
        ln_scale_vals=ln_scale_vals,
        knots=knots,
        x=x,
        **extra_kwargs,
    )

    value_grid = np.repeat(np.linspace(0, 20, 111)[:, None], repeats=x.size, axis=1)

    lp = mix.log_prob(value_grid)

    if isinstance(mix, TruncatedNormal1DSplineMixture):
        assert jnp.all(jnp.isfinite(lp[value_grid < extra_kwargs["high"]]))
        assert jnp.all(~jnp.isfinite(lp[value_grid > extra_kwargs["high"]]))
