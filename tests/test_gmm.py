import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from stream_membership.distributions.gmm import IndependentGMM


@pytest.mark.parametrize(
    "kwargs",
    [
        {  # Two 1D distributions, truncated low/high:
            "probs": [1.0, 0.2],
            "locs": np.array([[1.0], [2.0]]).T,
            "scales": np.array([[0.25], [0.1]]).T,
            "low": 0.5,
            "high": 2.1,
        },
        {  # Three 2D distributions, truncated low:
            "probs": [1.0, 0.5, 0.2],
            "locs": np.array([[1.0, 2.0], [1.5, 0.5], [0.5, 1.0]]).T,
            "scales": 0.2,
            "low": np.array([0.2, 0.0])[:, None],
        },
        {  # Three 2D distributions, truncated low but only one dim:
            "probs": [1.0, 0.5, 0.2],
            "locs": np.array([[1.0, 2.0], [1.5, 0.5], [0.5, 1.0]]).T,
            "scales": 0.2,
            "low": np.array([0.2, -np.inf])[:, None],
        },
        {  # Two 3D distributions, truncated high:
            "probs": [1.0, 0.2],
            "locs": np.array([[1.0, 2.0, 0.0], [1.5, 0.5, -1]]).T,
            "scales": np.array([[1.0, 1.0, 2.0], [2, 1, 1]]).T,
            "high": np.array([2.2, 1.5, 3.5])[:, None],
        },
    ],
)
def test_gridgmm(kwargs):
    """
    A mixture of two 1D distributions
    """
    mix = dist.Categorical(probs=jnp.array(kwargs.pop("probs")))
    gmm = IndependentGMM(mix, **kwargs)
    gmm_notrunc = IndependentGMM(
        mix, **{k: v for k, v in kwargs.items() if k not in ["low", "high"]}
    )
    D = gmm._D

    N_samples = 10

    rng = np.random.default_rng(seed=42)
    vals = rng.uniform(-10, 10, size=(N_samples, D))
    check = np.all(gmm.support.check(vals), axis=1)

    logprob_vals = gmm.log_prob(vals)
    assert np.all(np.isfinite(logprob_vals[check]))
    assert np.all(~np.isfinite(logprob_vals[~check]))
    assert np.all(logprob_vals[check] >= gmm_notrunc.log_prob(vals[check]))
