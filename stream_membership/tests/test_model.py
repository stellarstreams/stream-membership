import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from scipy.interpolate import InterpolatedUnivariateSpline

from ..gmm import IndependentGMM
from ..model import ModelComponent
from ..numpyro_dist import TruncatedNormalSpline


def test_subclass():
    numpyro.enable_x64()

    phi1_lim = (-100, 20)
    phi2_lim = (-8, 4)
    pm1_lim = (None, -2.0)

    # Simulate data:
    rng_key = jax.random.PRNGKey(416)
    N = 1024

    keys = jax.random.split(rng_key, num=4)
    data = {
        "phi1": dist.TruncatedNormal(
            loc=-70, scale=30, low=phi1_lim[0], high=phi1_lim[1]
        ).sample(keys[1], (N,)),
        "phi2": dist.Uniform(*phi2_lim).sample(keys[2], (N,)),
    }

    _loc_spl = InterpolatedUnivariateSpline(
        [-100, -20, 0.0, 20], [-5.0, -1.0, 2.0, 5.0]
    )
    data["pm1"] = dist.TruncatedNormal(
        loc=_loc_spl(data["phi1"]), scale=2, high=pm1_lim[1]
    ).sample(keys[3])

    # Set up the model:
    phi12_locs = jnp.stack(
        jnp.meshgrid(
            jnp.arange(phi1_lim[0] - 20, phi1_lim[1] + 20 + 1e-3, 20),
            jnp.arange(phi2_lim[0] - 4, phi2_lim[1] + 4 + 1e-3, 4),
        ),
        axis=-1,
    ).reshape(-1, 2)

    pm1_knots = jnp.arange(-100, 20 + 1e-3, 30)
    bkg_model = ModelComponent(
        name="background",
        coord_distributions={
            ("phi1", "phi2"): IndependentGMM,
            "pm1": TruncatedNormalSpline,
        },
        coord_parameters={
            ("phi1", "phi2"): {
                "mixing_distribution": (
                    dist.Categorical,
                    dist.Dirichlet(jnp.ones(phi12_locs.shape[0])),
                ),
                "locs": phi12_locs.T,
                "scales": dist.HalfNormal(2.0).expand((2, 1)),
                "low": jnp.array([phi1_lim[0], phi2_lim[0]])[:, None],
                "high": jnp.array([phi1_lim[1], phi2_lim[1]])[:, None],
            },
            "pm1": {
                "loc_vals": dist.Uniform(-8, 20).expand([pm1_knots.shape[0]]),
                "ln_scale_vals": dist.Uniform(-5, 5).expand([pm1_knots.shape[0]]),
                "knots": pm1_knots,
                "x": data["phi1"],
                "low": pm1_lim[0],
                "high": pm1_lim[1],
                "spline_k": 3,
            },
        },
        log_prob_extra_data={"pm1": {"x": "phi1"}},
    )

    # Try running SVI to get MAP parameters:
    optimizer = numpyro.optim.Adam(1e-2)
    # guide = AutoNormal(background_model)
    guide = AutoDelta(bkg_model, init_loc_fn=numpyro.infer.init_to_sample())
    svi = SVI(bkg_model, guide, optimizer, Trace_ELBO())
    svi_results = svi.run(jax.random.PRNGKey(0), 2_000, data=data)

    thing = Predictive(guide, params=svi_results.params, num_samples=1)
    MAP_p = thing(rng_key, data=data)
    MAP_p = {k: v[0] for k, v in MAP_p.items()}
    MAP_p_unpacked = bkg_model.expand_numpyro_params(MAP_p)

    grid_1ds = {
        "phi1": jnp.linspace(*phi1_lim, 101),
        "phi2": jnp.linspace(*phi2_lim, 102),
        "pm1": jnp.linspace(-15.0, pm1_lim[1], 103),
    }
    grids, ln_probs = bkg_model.evaluate_num_on_2d_grids(
        MAP_p_unpacked, grids=grid_1ds, x_coord_name="phi1"
    )
