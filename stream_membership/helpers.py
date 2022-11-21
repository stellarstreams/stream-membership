import jax
import jax.numpy as jnp
import jax.scipy as jsci
import numpyro.distributions as dist

from .truncnorm import CustomTruncatedNormal

__all__ = ["ln_simpson"]


@jax.jit
def ln_simpson(ln_y, x):
    """
    Evaluate the log of the definite integral of a function evaluated on a grid using
    Simpson's rule
    """

    dx = jnp.diff(x)[0]
    num_points = len(x)
    if num_points // 2 == num_points / 2:
        raise ValueError("Because of laziness, the input size must be odd")

    weights_first = jnp.asarray([1.0])
    weights_mid = jnp.tile(jnp.asarray([4.0, 2.0]), [(num_points - 3) // 2])
    weights_last = jnp.asarray([4.0, 1.0])
    weights = jnp.concatenate([weights_first, weights_mid, weights_last], axis=0)

    return jsci.special.logsumexp(ln_y + jnp.log(weights), axis=-1) + jnp.log(dx / 3)


def two_truncated_normal_mixture(w, mean1, ln_std1, mean2, ln_std2, low, high, yerr):
    mix = dist.Categorical(probs=jnp.array([w, 1.0 - w]).T)

    var1 = jnp.exp(2 * ln_std1)
    var2 = var1 + jnp.exp(2 * ln_std2)

    # dists = [
    #     CustomTruncatedNormal(mean1, jnp.sqrt(var1 + yerr**2), low=low, high=high),
    #     CustomTruncatedNormal(mean2, jnp.sqrt(var2 + yerr**2), low=low, high=high),
    # ]
    dists = [
        dist.TruncatedNormal(
            loc=mean1, scale=jnp.sqrt(var1 + yerr**2), low=low, high=high
        ),
        dist.TruncatedNormal(
            loc=mean2, scale=jnp.sqrt(var2 + yerr**2), low=low, high=high
        ),
    ]
    return dist.MixtureGeneral(mix, dists)


def two_normal_mixture(w, mean1, ln_std1, mean2, ln_std2, yerr):
    mix = dist.Categorical(probs=jnp.array([w, 1.0 - w]).T)

    var1 = jnp.exp(2 * ln_std1)
    var2 = var1 + jnp.exp(2 * ln_std2)

    dists = [
        dist.Normal(mean1, jnp.sqrt(var1 + yerr**2)),
        dist.Normal(mean2, jnp.sqrt(var2 + yerr**2)),
    ]
    return dist.MixtureGeneral(mix, dists)
