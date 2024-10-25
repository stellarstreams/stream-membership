import jax
import jax.numpy as jnp
import numpy as np
import pytest

from .. import DirichletSpline, NormalSpline, TruncatedNormalSpline


def make_dists(seed=867):
    rng = np.random.default_rng(seed)

    knots = jnp.linspace(0, 10, 8)
    x = jnp.linspace(1, 8, 32)

    dists = []

    d = NormalSpline(
        knots=knots,
        loc_vals=rng.uniform(-5, 5, knots.size),
        ln_scale_vals=rng.uniform(-1, 1, knots.size),
        x=x,
    )
    dists.append(d)

    d = TruncatedNormalSpline(
        knots=knots,
        loc_vals=rng.uniform(-5, 5, knots.size),
        ln_scale_vals=rng.uniform(-1, 1, knots.size),
        low=-4.0,
        high=4.0,
        x=x,
    )
    dists.append(d)

    d = DirichletSpline(
        knots=knots, concentration_vals=rng.uniform(0, 1, (3, knots.size)), x=x
    )
    dists.append(d)

    return dists


@pytest.mark.parametrize("dist_", make_dists())
def test_methods(dist_):
    # New x grid to be used below:
    x2 = jnp.linspace(1, 8, 64)

    s = dist_.sample(jax.random.PRNGKey(0))
    assert s.shape == (dist_.x.size, *dist_.event_shape)
    s = dist_.sample(jax.random.PRNGKey(1), sample_shape=(10,))
    assert s.shape == (10, dist_.x.size, *dist_.event_shape)

    # Test sample with a new x grid:
    s = dist_.sample(jax.random.PRNGKey(2), x=x2)
    assert s.shape == (x2.size, *dist_.event_shape)
    s = dist_.sample(jax.random.PRNGKey(3), x=x2, sample_shape=(10,))
    assert s.shape == (10, x2.size, *dist_.event_shape)

    # Test evaluating log_prob with previously provided x grid:
    shape = dist_.x.shape + dist_.event_shape
    y = np.linspace(0, 1, np.prod(shape)).reshape(shape)
    lp = dist_.log_prob(y)
    assert lp.shape == dist_.x.shape

    # Test evaluating log_prob with a different x grid:
    shape = x2.shape + dist_.event_shape
    y2 = np.linspace(0, 1, np.prod(shape)).reshape(shape)
    lp = dist_.log_prob(y2, x=x2)
    assert lp.shape == x2.shape
