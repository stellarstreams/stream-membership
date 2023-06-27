import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro import infer

from .. import ModelBase
from ..variables import Normal1DVariable


def test_subclass():
    numpyro.enable_x64()

    class TestModel(ModelBase):
        name = "test"

        ln_N_dist = dist.Uniform(-10, 15)

        variables = {
            "phi1": Normal1DVariable(
                param_priors={
                    "mean": dist.Uniform(0, 10),
                    "ln_std": dist.Uniform(-5, 5),
                },
                coord_bounds=(0, 5),
            ),
            "phi2": Normal1DVariable(
                param_priors={
                    "mean": dist.Uniform(-5, 5),
                    "ln_std": dist.Uniform(-5, 5),
                },
            ),
        }

    # Make fake data:
    rng = np.random.default_rng(seed=42)

    N = 10_070
    true_pars = {
        "ln_N": np.log(N),
        "phi1": {
            "mean": 2.0,
            "ln_std": 0.423,
        },
        "phi2": {"mean": 0.0, "ln_std": 0.55},
    }

    data = {
        "phi1": dist.TruncatedNormal(
            true_pars["phi1"]["mean"],
            np.exp(true_pars["phi1"]["ln_std"]),
            low=0.0,
            high=5.0,
        ).sample(jax.random.PRNGKey(42), (N,)),
        "phi2": rng.normal(
            true_pars["phi2"]["mean"], np.exp(true_pars["phi2"]["ln_std"]), size=N
        ),
    }

    params0 = {
        "ln_N": np.log(500.0),
        "phi1": {
            "mean": 3.0,
            "ln_std": 0.1,
        },
        "phi2": {"mean": -1.0, "ln_std": 0.1},
    }
    obj_val = TestModel.objective(params0, data)
    assert np.isfinite(obj_val)

    # Try optimizing without bounds:
    optimizer = jaxopt.ScipyMinimize(
        "bfgs",
        fun=TestModel.objective,
        maxiter=1_000,
    )
    opt_res = optimizer.run(
        init_params=params0,
        data=data,
    )
    opt_pars = opt_res.params
    assert opt_res.state.success

    for k, truth in true_pars.items():
        if isinstance(truth, dict):
            for par_name in truth:
                assert np.allclose(opt_pars[k][par_name], truth[par_name], atol=0.1)
        else:
            assert np.allclose(opt_pars[k], truth, rtol=1e-2)

    # Try optimizing with bounds:
    optimizer = jaxopt.ScipyBoundedMinimize(
        "l-bfgs-b",
        fun=TestModel.objective,
        # options=dict(maxls=10000),
        maxiter=1_000,
    )
    opt_res = optimizer.run(
        init_params=params0, data=data, bounds=TestModel._get_jaxopt_bounds()
    )
    opt_pars = opt_res.params
    assert opt_res.state.success

    for k, truth in true_pars.items():
        if isinstance(truth, dict):
            for par_name in truth:
                assert np.allclose(opt_pars[k][par_name], truth[par_name], atol=0.1)
        else:
            assert np.allclose(opt_pars[k], truth, rtol=1e-2)

    # Now try sampling:
    nchains = 2

    rng = np.random.default_rng(seed=42)

    leaves, tree_def = jax.tree_util.tree_flatten(params0)
    chain_leaves = []
    for leaf in leaves:
        leaf = np.atleast_1d(leaf)
        arr = jnp.reshape(leaf, (1,) + leaf.shape)
        arr = arr + rng.normal(0, 1e-3, size=(nchains,) + leaf.shape)
        chain_leaves.append(arr)
    chain_params0 = jax.tree_util.tree_unflatten(tree_def, chain_leaves)

    sampler = infer.MCMC(
        infer.NUTS(TestModel.setup_numpyro),
        num_warmup=1000,
        num_samples=1000,
        num_chains=nchains,
        progress_bar=False,
    )
    sampler.run(jax.random.PRNGKey(0), data=data, init_params=chain_params0)
    print(sampler.get_samples())
