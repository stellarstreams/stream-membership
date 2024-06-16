import jax.numpy as jnp
import numpyro.distributions as dist

__all__ = ["two_truncated_normal_mixture", "two_normal_mixture"]


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
