"""
Jax and numpyro support for truncated Normal distributions.
"""

import numpyro.distributions as dist
import scipy.stats as osp_stats
from jax import lax
from jax._src.numpy import lax_numpy as jnp
from jax._src.numpy.util import _wraps
from jax._src.random import uniform
from jax._src.scipy import special
from jax._src.scipy.stats.truncnorm import _log_gauss_mass
from jax._src.scipy.stats.truncnorm import logpdf as truncnorm_logpdf
from numpyro.distributions.util import is_prng_key, promote_shapes

__all__ = ["APWTruncatedNormal"]


@_wraps(osp_stats.truncnorm.ppf, update_doc=False)
def ppf(q, a, b):
    q, a, b = jnp.broadcast_arrays(q, a, b)

    case_left = a < 0
    case_right = ~case_left

    def ppf_left(q, a, b):
        log_Phi_x = jnp.logaddexp(
            special.log_ndtr(a), jnp.log(q) + _log_gauss_mass(a, b)
        )
        # TODO: should use ndtri_exp(log_Phi_x), but that's not in jax.scipy.special
        return special.ndtri(jnp.exp(log_Phi_x))

    def ppf_right(q, a, b):
        log_Phi_x = jnp.logaddexp(
            special.log_ndtr(-b), jnp.log1p(-q) + _log_gauss_mass(a, b)
        )
        # TODO: should use ndtri_exp(log_Phi_x), but that's not in jax.scipy.special
        return -special.ndtri(jnp.exp(log_Phi_x))

    out = jnp.empty_like(q)
    out = jnp.select(
        [case_left, case_right],
        [ppf_left(q, a, b), ppf_right(q, a, b)],
    )

    return out


def rvs(key, a, b, loc=0.0, scale=1.0, shape=()):
    dtype = jnp.result_type(float)
    finfo = jnp.finfo(dtype)
    minval = finfo.tiny
    U = uniform(key, shape=shape, minval=minval)
    Y = ppf(U, a, b, loc=loc, scale=scale)
    return Y * scale + loc


class APWTruncatedNormal(dist.Distribution):
    arg_constraints = {
        "loc": dist.constraints.real,
        "scale": dist.constraints.positive,
        "low": dist.constraints.dependent,
        "high": dist.constraints.dependent,
    }
    reparametrized_params = ["low", "high"]

    def __init__(self, loc=0.0, scale=1.0, low=-jnp.inf, high=jnp.inf):
        self.loc, self.scale, self.low, self.high = promote_shapes(
            loc, scale, low, high
        )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(loc), jnp.shape(scale), jnp.shape(low), jnp.shape(high)
        )

        self._a = (self.low - self.loc) / self.scale
        self._b = (self.high - self.loc) / self.scale
        self._support = dist.constraints.interval(self.low, self.high)

        super().__init__(batch_shape=batch_shape, event_shape=())

    @dist.constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return rvs(
            key,
            self._a,
            self._b,
            loc=self.loc,
            scale=self.scale,
            shape=sample_shape + self.batch_shape,
        )

    def log_prob(self, value):
        return truncnorm_logpdf(
            value, a=self._a, b=self._b, loc=self.loc, scale=self.scale
        )
