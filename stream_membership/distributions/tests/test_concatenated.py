import jax
import jax.numpy as jnp
from numpyro import distributions as dist

from stream_membership.distributions.concatenated import ConcatenatedDistributions


def test_concatenated_univ():
    x1 = dist.Normal(0, 1)
    x2 = dist.Normal(2.0, 0.5)
    x = ConcatenatedDistributions([x1, x2])
    assert x.event_shape == (2,)

    values_expected_shape = [
        (jnp.array([0.0, 2.0]), 1),
        (jnp.array([[0.0, 2.0], [1.0, 3.0], [2.0, 4.0]]), 3),
    ]
    for value, expected_shape in values_expected_shape:
        assert x.log_prob(value).shape == (expected_shape,)

    shape_expected_shape = [
        ((), (2,)),
        ((3,), (3, 2)),
        ((3, 5), (3, 5, 2)),
    ]
    for sample_shape, expected_shape in shape_expected_shape:
        samples = x.sample(jax.random.PRNGKey(0), sample_shape=sample_shape)
        assert samples.shape == expected_shape


def test_concatenated_univ_multiv():
    x1 = dist.Normal(0, 1)
    x2 = dist.MultivariateNormal(
        loc=jnp.array([1.0, 2.0]),
        covariance_matrix=jnp.array([[1.0, 0.0], [0, 0.5]]) ** 2,
    )

    x = ConcatenatedDistributions([x1, x2])
    assert x.event_shape == (3,)

    values_expected_shape = [
        (jnp.array([0, 1, 2.0]), 1),
        (jnp.array([[0.0, 1, 2.0], [1.0, 2, 3.0], [2.0, 3, 4.0]]), 3),
    ]
    for value, expected_shape in values_expected_shape:
        assert x.log_prob(value).shape == (expected_shape,)

    shape_expected_shape = [
        ((), (3,)),
        ((3,), (3, 3)),
        ((3, 5), (3, 5, 3)),
    ]
    for sample_shape, expected_shape in shape_expected_shape:
        samples = x.sample(jax.random.PRNGKey(0), sample_shape=sample_shape)
        assert samples.shape == expected_shape
