import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive

from ..variables import GridGMMVariable, Normal1DSplineVariable, Normal1DVariable


def test_normal1d():
    # Using numeric values so this can be evaluated:
    c = Normal1DVariable({"mean": dist.Uniform(0, 10), "ln_std": dist.Uniform(-5, 5)})

    pars = {"mean": 1.0, "ln_std": 0.5}
    c.ln_prob(pars, 5.0)
    c.ln_prob(pars, np.linspace(0, 10, 15))

    # Using numpyro distributions for the parameters for deferred evaluation:
    def model():
        numpyro.sample("val", c.get_dist(c.setup_numpyro()))

    predictive = Predictive(model, num_samples=10)
    predictive(jax.random.PRNGKey(1))

    # Test also truncated version:
    def model2():
        c = Normal1DVariable(
            {"mean": dist.Uniform(0, 10), "ln_std": dist.Uniform(-5, 5)},
            coord_bounds=(-0.5, 0.5),
        )
        numpyro.sample("val", c.get_dist(c.setup_numpyro()))

    predictive2 = Predictive(model2, num_samples=10)
    predictive2(jax.random.PRNGKey(1))


def test_normal1dspline():
    # Using numeric values so this can be evaluated:
    knots = np.linspace(0, 10, 8)
    param_priors = {
        "mean": dist.Uniform(jnp.zeros_like(knots), jnp.ones_like(knots)),
        "ln_std": dist.Uniform(jnp.full_like(knots, -1), jnp.full_like(knots, 1)),
    }
    c = Normal1DSplineVariable(param_priors, knots=knots)

    params = {"mean": np.full_like(knots, 0.5), "ln_std": np.full_like(knots, 0.5)}
    c.ln_prob(params, 1.0, x=5.0)
    c.ln_prob(params, np.linspace(0.5, 1.5, 15), x=np.linspace(0.5, 9.5, 15))

    # Using numpyro distributions for the parameters for deferred evaluation:
    x_vals = np.linspace(0.5, 6.0, 12)

    def model():
        numpyro.sample("val", c.get_dist(c.setup_numpyro(), x_vals))

    predictive = Predictive(model, num_samples=10)
    predictive(jax.random.PRNGKey(1))

    # Test also truncated version:
    def model2():
        c = Normal1DSplineVariable(param_priors, knots=knots, coord_bounds=(-0.5, 0.5))
        numpyro.sample("val", c.get_dist(c.setup_numpyro(), x_vals))

    predictive2 = Predictive(model2, num_samples=10)
    predictive2(jax.random.PRNGKey(1))


def test_gridgmm1d():
    # Using numeric values so this can be evaluated:
    locs = np.array([[0], [1.0], [2.0]])
    scales = np.linspace(0.1, 0.75, locs.shape[0]).reshape(locs.shape)

    c = GridGMMVariable(
        param_priors={
            "zs": dist.Uniform(
                -jnp.ones(locs.shape[0] - 1), jnp.ones(locs.shape[0] - 1)
            )
        },
        locs=locs,
        scales=scales,
    )

    params = {"zs": np.linspace(0.1, 0.9, locs.shape[0] - 1)}
    c.ln_prob(params, np.array([0.5]))

    grid = np.linspace(-1, 10, 128).reshape(-1, 1)
    ln_vals = c.ln_prob(params, grid)
    assert np.all(np.isfinite(ln_vals))


def test_gridgmm2d():
    # Using numeric values so this can be evaluated:
    locs = np.array([[0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.5, 0.75]])
    scales = np.ones_like(locs)
    scales[:, 0] = 0.25

    c = GridGMMVariable(
        param_priors={
            "zs": dist.Uniform(
                -jnp.ones(locs.shape[0] - 1), jnp.ones(locs.shape[0] - 1)
            )
        },
        locs=locs,
        scales=scales,
    )

    params = {"zs": np.ones(locs.shape[0] - 1) / 2}
    c.ln_prob(params, np.array([0.5, -0.3]))

    grid = np.stack(np.meshgrid(np.linspace(-1, 1, 128), np.linspace(-1, 3, 129))).T
    ln_vals = c.ln_prob(params, grid.reshape(-1, 2)).reshape(grid.shape[:-1])
    assert np.all(np.isfinite(ln_vals))

    # Do the same, but with coordinate bounds:
    c = GridGMMVariable(
        param_priors={
            "zs": dist.Uniform(
                -jnp.ones(locs.shape[0] - 1), jnp.ones(locs.shape[0] - 1)
            )
        },
        locs=locs,
        scales=scales,
        coord_bounds=(np.array([-0.5, 0.0]), np.array([0.5, 2.5])),
    )

    params = {"zs": np.ones(locs.shape[0] - 1) / 2}
    c.ln_prob(params, np.array([0.5, -0.3]))

    grid = np.stack(np.meshgrid(np.linspace(-1, 1, 128), np.linspace(-1, 3, 129))).T
    ln_vals = c.ln_prob(params, grid.reshape(-1, 2)).reshape(grid.shape[:-1])
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
