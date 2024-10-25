import jax
import jax.numpy as jnp
import numpyro
import pytest
from numpyro import distributions as dist
from numpyro.infer import Predictive

from stream_membership.distributions.concatenated import ConcatenatedDistributions

values_expected_shape = [
    (jnp.array([0, 1, 2.0]), 1),
    (jnp.array([[0.0, 1, 2.0], [1.0, 2, 3.0], [2.0, 3, 4.0]]), 3),
]
sample_shapes = [(), (1,), (4,), (4, 5)]


class BaseTestConcatenated:
    def setup_dist(self):
        x1 = dist.Normal(0, 1)
        x2 = dist.Normal(2.0, 0.5)
        x3 = dist.Normal(1.0, 0.25)
        return ConcatenatedDistributions([x1, x2, x3])

    def test_univ_shape(self):
        x = self.setup_dist()
        assert x.event_shape == (3,)

    @pytest.mark.parametrize(("value", "expected_shape"), values_expected_shape)
    def test_logprob(self, value, expected_shape):
        x = self.setup_dist()
        assert x.log_prob(value).shape == (expected_shape,)

    @pytest.mark.parametrize("sample_shape", sample_shapes)
    def test_sample(self, sample_shape):
        x = self.setup_dist()

        samples = x.sample(jax.random.PRNGKey(0), sample_shape=sample_shape)
        assert samples.shape == (*sample_shape, 3)

    def test_numpyro_predictive(self):
        def model():
            x = self.setup_dist()
            numpyro.sample("x", x)

        pred = Predictive(model, num_samples=10)(jax.random.PRNGKey(42))
        assert pred["x"].shape == (10, 3)


class TestAllUnivariate(BaseTestConcatenated):
    def setup_dist(self):
        x1 = dist.Normal(0, 1)
        x2 = dist.Normal(2.0, 0.5)
        x3 = dist.Normal(1.0, 0.25)
        return ConcatenatedDistributions([x1, x2, x3])


class TestUnivariateMultivariate(BaseTestConcatenated):
    def setup_dist(self):
        x1 = dist.Normal(0, 1)
        x2 = dist.MultivariateNormal(
            loc=jnp.array([1.0, 2.0]),
            covariance_matrix=jnp.array([[1.0, 0.0], [0, 0.5]]) ** 2,
        )
        return ConcatenatedDistributions([x1, x2])
