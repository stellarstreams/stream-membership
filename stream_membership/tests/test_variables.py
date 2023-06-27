import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive

from ..components import GridGMMComponent, Normal1DComponent, Normal1DSplineComponent


def test_normal1d():
    # Using numeric values so this can be evaluated:
    c = Normal1DComponent({"mean": 1.0, "ln_std": 0.5})
    c.ln_prob(5.0)
    c.ln_prob(np.linspace(0, 10, 15))

    # Using numpyro distributions for the parameters for deferred evaluation:
    def model():
        numpyro.sample("val", c.get_dist())

    predictive = Predictive(model, num_samples=10)
    predictive(jax.random.PRNGKey(1))

    # Test also truncated version:
    def model2():
        c = Normal1DComponent({"mean": 1.0, "ln_std": 0.5}, coord_bounds=(-0.5, 0.5))
        numpyro.sample("val", c.get_dist())

    predictive2 = Predictive(model2, num_samples=10)
    predictive2(jax.random.PRNGKey(1))


def test_normal1dspline():
    # Using numeric values so this can be evaluated:
    knots = np.linspace(0, 10, 8)
    params = {"mean": np.ones_like(knots), "ln_std": np.full_like(knots, 0.5)}
    c = Normal1DSplineComponent(params=params, knots=knots)
    c.ln_prob(1.0, x=5.0)
    c.ln_prob(np.linspace(0.5, 1.5, 15), x=np.linspace(0.5, 9.5, 15))

    # Using numpyro distributions for the parameters for deferred evaluation:
    def model():
        c = Normal1DSplineComponent(params=params, knots=knots)
        numpyro.sample("val", c.get_dist(np.linspace(1, 4, 12)))

    predictive = Predictive(model, num_samples=10)
    predictive(jax.random.PRNGKey(1))

    # Test also truncated version:
    def model2():
        c = Normal1DSplineComponent(
            params=params, knots=knots, coord_bounds=(-0.5, 0.5)
        )
        numpyro.sample("val", c.get_dist(np.linspace(1, 4, 12)))

    predictive2 = Predictive(model2, num_samples=10)
    predictive2(jax.random.PRNGKey(1))


def test_gridgmm1d():
    # Using numeric values so this can be evaluated:
    locs = np.array([[0], [1.0], [2.0]])
    scales = np.linspace(0.1, 0.75, locs.shape[0]).reshape(locs.shape)

    params = {"ws": np.linspace(0.1, 1, locs.shape[0])}
    c = GridGMMComponent(params=params, locs=locs, scales=scales)
    c.ln_prob(np.array([0.5]))

    grid = np.linspace(-1, 10, 128).reshape(-1, 1)
    ln_vals = c.ln_prob(grid)
    assert np.all(np.isfinite(ln_vals))


def test_gridgmm2d():
    # Using numeric values so this can be evaluated:
    locs = np.array([[0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.5, 0.75]])
    scales = np.ones_like(locs)
    scales[:, 0] = 0.25

    params = {"ws": np.ones(locs.shape[0])}
    c = GridGMMComponent(params=params, locs=locs, scales=scales)
    c.ln_prob(np.array([0.5, -0.3]))

    grid = np.stack(np.meshgrid(np.linspace(-1, 1, 128), np.linspace(-1, 3, 129))).T
    ln_vals = c.ln_prob(grid.reshape(-1, 2)).reshape(grid.shape[:-1])
    assert np.all(np.isfinite(ln_vals))

    # Do the same, but with coordinate bounds:
    params = {"ws": np.ones(locs.shape[0])}
    c = GridGMMComponent(
        params=params,
        locs=locs,
        scales=scales,
        coord_bounds=(np.array([-0.5, 0.0]), np.array([0.5, 2.5])),
    )
    c.ln_prob(np.array([0.5, -0.3]))

    grid = np.stack(np.meshgrid(np.linspace(-1, 1, 128), np.linspace(-1, 3, 129))).T
    ln_vals = c.ln_prob(grid.reshape(-1, 2)).reshape(grid.shape[:-1])
    assert not np.all(np.isfinite(ln_vals))

    # TODO: when I implement .sample() on GridGMMComponent, test it here.
    # Using numpyro distributions for the parameters for deferred evaluation:
    # def model():
    #     c = GridGMMComponent(
    #         locs=locs,
    #         scales=scales,
    #         coord_bounds=(np.array([-0.5, 0.0]), np.array([0.5, 2.5])),
    #     )
    #     c.set_params({"ws": np.ones(locs.shape[0])})
    #     numpyro.sample("val", c.get_dist())

    # predictive = Predictive(model, num_samples=10)
    # predictive(jax.random.PRNGKey(1))

    # # Test also truncated version:
    # def model2():
    #     c = Normal1DSplineComponent(knots, coord_bounds=(-0.5, 0.5))
    #     c.set_params(
    #         {
    #             "mean": dist.Uniform(np.zeros_like(knots), np.ones_like(knots)),
    #             "ln_std": dist.Normal(np.full_like(knots, 0.5), np.ones_like(knots)),
    #         }
    #     )
    #     numpyro.sample("val", c.get_dist(np.linspace(1, 4, 12)))

    # predictive2 = Predictive(model2, num_samples=10)
    # predictive2(jax.random.PRNGKey(1))
