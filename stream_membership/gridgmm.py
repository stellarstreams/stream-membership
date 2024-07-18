import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import lax
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike
from numpyro.distributions import constraints
from numpyro.distributions.constraints import _IndependentConstraint, _Interval
from numpyro.distributions.util import promote_shapes

__all__ = ["IsotropicGridGMM"]


class _IntervalVector(_IndependentConstraint):
    def __init__(self, lower_bound, upper_bound):
        super().__init__(_Interval(lower_bound, upper_bound), 1)


interval_vector = _IntervalVector


# TODO: test against mixturesamefamily with multivariate normals
class IsotropicGridGMM(dist.Distribution):
    """
    A Gaussian Mixture Model where the components are fixed to their input locations and
    there are no covariances (but each dimension can have different scales / standard
    deviations).
    """

    arg_constraints = {  # noqa: RUF012
        "locs": constraints.real,
        "scales": constraints.positive,
        "low": constraints.less_than(float("inf")),
        "high": constraints.greater_than(-float("inf")),
    }
    reparametrized_params = ["locs", "scales"]  # noqa: RUF012

    def __init__(
        self,
        mixing_distribution: dist.CategoricalLogits | dist.CategoricalProbs,
        locs: ArrayLike = 0.0,
        scales: ArrayLike = 1.0,
        low: ArrayLike = -jnp.inf,
        high: ArrayLike = jnp.inf,
        *,
        validate_args=True,
    ):
        dist.mixtures._check_mixing_distribution(mixing_distribution)
        self.mixing_distribution = mixing_distribution

        # K = mixture components, D = dimensions
        # - event_shape is the dimensionality of the data - number of dependent
        #   coordinates, i.e., "D" in the below
        # - batch_shape is the number of independent dimensions - here "K"
        combined_shape = lax.broadcast_shapes(jnp.shape(locs), jnp.shape(scales))
        if len(combined_shape) > 2:
            msg = (
                "locs and scales must have at most 2 dimensions, but got "
                f"{len(combined_shape)} dims"
            )
            raise ValueError(msg)
        self._K, self._D = combined_shape
        batch_shape = (self._K,)
        event_shape = (self._D,)

        self.locs, self.scales = promote_shapes(
            jnp.array(locs),
            jnp.array(scales),
            shape=combined_shape,
        )
        self._locs = jnp.broadcast_to(self.locs, combined_shape)
        self._scales = jnp.broadcast_to(self.scales, combined_shape)

        # low and high should be the same shape as event_shape (D)
        self.low, self.high = promote_shapes(
            jnp.array(low), jnp.array(high), shape=event_shape
        )
        self._low = jnp.broadcast_to(self.low, event_shape)
        self._high = jnp.broadcast_to(self.high, event_shape)

        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

        self._support = interval_vector(low, high)

        if jnp.all(jnp.isinf(self._low)) and jnp.all(jnp.isinf(self._high)):
            self._component_dist = dist.Normal(loc=self._locs, scale=self._scales)
        else:
            self._component_dist = dist.TruncatedNormal(
                loc=self._locs, scale=self._scales, low=self._low, high=self._high
            )

    @constraints.dependent_property
    def support(self) -> constraints.Constraint:
        return self._support

    def component_log_probs(self, value: ArrayLike) -> jax.Array:
        value = jnp.expand_dims(value, -1)
        component_log_probs = self._component_dist.log_prob(value)
        return jax.nn.log_softmax(self.mixing_distribution.logits) + component_log_probs

    def log_prob(self, value: ArrayLike) -> jax.Array:
        comp_lp = self.component_log_probs(value)
        return logsumexp(comp_lp.sum(axis=-2), axis=-1)

    def sample(self, *_, **__):
        msg = "Sampling not implemented for IsotropicGridGMM"
        raise NotImplementedError(msg)
