import warnings

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from ..truncatedgridgmm import TruncatedGridGMM


@pytest.mark.parametrize(
    "kwargs",
    [
        {  # Two 1D distributions, truncated low/high:
            "probs": [1.0, 0.2],
            "locs": [[1.0], [2.0]],
            "scales": [[0.25], [0.1]],
            "low": 0.5,
            "high": 2.1,
        },
        {  # Three 2D distributions, truncated low:
            "probs": [1.0, 0.5, 0.2],
            "locs": [[1.0, 2.0], [1.5, 0.5], [0.5, 1.0]],
            "scales": 0.2,
            "low": [0.2, 0.0],
        },
        {  # Three 2D distributions, truncated low but only one dim:
            "probs": [1.0, 0.5, 0.2],
            "locs": [[1.0, 2.0], [1.5, 0.5], [0.5, 1.0]],
            "scales": 0.2,
            "low": [0.2, -np.inf],
        },
        {  # Two 3D distributions, truncated high:
            "probs": [1.0, 0.2],
            "locs": [[1.0, 2.0, 0.0], [1.5, 0.5, -1]],
            "scales": [[1.0, 1.0, 2.0], [2, 1, 1]],
            "high": [2.2, 1.5, 3.5],
        },
    ],
)
def test_gridgmm(kwargs):
    """
    A mixture of two 1D distributions
    """
    mix = dist.Categorical(probs=jnp.array(kwargs.pop("probs")))
    gmm = TruncatedGridGMM(mix, **kwargs)
    gmm_notrunc = TruncatedGridGMM(
        mix, **{k: v for k, v in kwargs.items() if k not in ["low", "high"]}
    )
    D = gmm._D

    # low and high could be inf, so to generate random numbers we need to be safe:
    val_high = np.broadcast_to(kwargs.get("high", np.inf), (D,))
    val_low = np.broadcast_to(kwargs.get("low", -np.inf), (D,))
    BIGNUM = int(1e6)
    N_samples = 10

    rng = np.random.default_rng(seed=42)
    good_vals = rng.uniform(
        np.max([val_low, np.full(D, -1e4)], axis=0),
        np.min([val_high, np.full(D, 1e4)], axis=0),
        (N_samples, D),
    )
    logprob_vals = gmm.log_prob(good_vals)
    assert np.all(np.isfinite(logprob_vals))
    assert np.all(logprob_vals >= gmm_notrunc.log_prob(good_vals))

    # outside of truncated region
    # only have 1 if val_low isn't -inf, same for 2
    args = []
    if np.any(np.isfinite(val_low)):
        tmp = val_low.copy()
        tmp[~np.isfinite(val_low)] = -0.5 * BIGNUM  # ???
        args.append(rng.uniform(-BIGNUM, tmp, (N_samples, D)))

    if np.any(np.isfinite(val_high)):
        tmp = val_high.copy()
        tmp[~np.isfinite(val_high)] = 0.5 * BIGNUM  # ???
        args.append(rng.uniform(tmp, BIGNUM, (N_samples, D)))

    bad_vals = np.concatenate(args)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logprob_vals = gmm.log_prob(bad_vals)
    assert np.all(~np.isfinite(logprob_vals))

    # TODO: remove this after implementing sample()
    with pytest.raises(NotImplementedError):
        gmm.sample()
