import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive

from ..components import Normal1DComponent, Normal1DSplineComponent


def test_normal1d():
    # Using numeric values so this can be evaluated:
    c = Normal1DComponent()
    c.set_params({"mean": 1.0, "ln_std": 0.5})
    c.ln_prob(5.0)
    c.ln_prob(np.linspace(0, 10, 15))

    # Using numpyro distributions for the parameters for deferred evaluation:
    def model():
        c = Normal1DComponent()
        c.set_params(
            {
                "mean": dist.Uniform(0.0, 1.0),
                "ln_std": dist.Normal(0.5, 1.0),
            }
        )
        numpyro.sample("val", c.get_dist())

    predictive = Predictive(model, num_samples=10)
    predictive(jax.random.PRNGKey(1))

    # Test also truncated version:
    def model2():
        c = Normal1DComponent(coord_bounds=(-0.5, 0.5))
        c.set_params(
            {
                "mean": dist.Uniform(0.0, 1.0),
                "ln_std": dist.Normal(0.5, 1.0),
            }
        )
        numpyro.sample("val", c.get_dist())

    predictive2 = Predictive(model2, num_samples=10)
    predictive2(jax.random.PRNGKey(1))


def test_normal1dspline():
    # Using numeric values so this can be evaluated:
    knots = np.linspace(0, 10, 8)
    c = Normal1DSplineComponent(knots)
    c.set_params({"mean": np.ones_like(knots), "ln_std": np.full_like(knots, 0.5)})
    c.ln_prob(1.0, x=5.0)
    c.ln_prob(np.linspace(0.5, 1.5, 15), x=np.linspace(0.5, 9.5, 15))

    # Using numpyro distributions for the parameters for deferred evaluation:
    def model():
        c = Normal1DSplineComponent(knots)
        c.set_params(
            {
                "mean": dist.Uniform(jnp.zeros_like(knots), jnp.ones_like(knots)),
                "ln_std": dist.Normal(jnp.full_like(knots, 0.5), jnp.ones_like(knots)),
            }
        )
        numpyro.sample("val", c.get_dist(np.linspace(1, 4, 12)))

    predictive = Predictive(model, num_samples=10)
    predictive(jax.random.PRNGKey(1))

    # Test also truncated version:
    def model2():
        c = Normal1DSplineComponent(knots, coord_bounds=(-0.5, 0.5))
        c.set_params(
            {
                "mean": dist.Uniform(np.zeros_like(knots), np.ones_like(knots)),
                "ln_std": dist.Normal(np.full_like(knots, 0.5), np.ones_like(knots)),
            }
        )
        numpyro.sample("val", c.get_dist(np.linspace(1, 4, 12)))

    predictive2 = Predictive(model2, num_samples=10)
    predictive2(jax.random.PRNGKey(1))
