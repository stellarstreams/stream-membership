__all__ = ["Normal1DSplineMixture", "TruncatedNormal1DSplineMixture"]

from typing import Any

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike

from stream_membership.distributions import NormalSpline, TruncatedNormalSpline


class Normal1DSplineMixture(dist.MixtureGeneral):
    def __init__(
        self,
        mixing_distribution: dist.CategoricalProbs | dist.CategoricalLogits,
        loc_vals: ArrayLike,
        ln_scale_vals: ArrayLike,
        knots: ArrayLike,
        x: ArrayLike,
        spline_k: int = 3,
        *,
        validate_args=None,
    ):
        """
        Represents a mixture of Normal distributions where the parameters are controlled
        by splines that are evaluated at some other parameter values x.

        Parameters
        ----------

        x
            Array of x values at which to evaluate the splines.
        spline_k
            Degree of the spline.
        """
        # Should have shape (n_knots, )
        self.knots = jnp.array(knots)
        self._n_knots = len(self.knots)

        # The pre-specified grid to evaluate on
        self.x = jnp.array(x)

        # Spline order:
        self.spline_k = int(spline_k)

        # Should have shape (n_components, n_knots)
        combined_shape = jax.lax.broadcast_shapes(
            jnp.shape(loc_vals), jnp.shape(ln_scale_vals)
        )
        if validate_args and (
            len(combined_shape) != 2 or combined_shape[-1] != self._n_knots
        ):
            msg = (
                "locs, scales, and concentrations must have 2 axes, but got "
                f"{len(combined_shape)}. The shape must be broadcastable to: "
                "(n_components, n_knots), where n_components is the number of mixture "
                "components and n_knots is the number of spline knots"
            )
            raise ValueError(msg)
        self._n_components = combined_shape[0]

        self.loc_vals = jnp.array(loc_vals)
        self.ln_scale_vals = jnp.array(ln_scale_vals)

        # Broadcasted arrays:
        self._loc_vals = jnp.broadcast_to(
            self.loc_vals, (self._n_components, self._n_knots)
        )
        self._ln_scale_vals = jnp.broadcast_to(
            self.ln_scale_vals, (self._n_components, self._n_knots)
        )
        super().__init__(
            mixing_distribution,
            self._make_components(),
            validate_args=validate_args,
        )

        expected = (self.x.size, self._n_components)
        if validate_args and self.mixing_distribution.probs.shape != expected:
            msg = (
                "The shape of the mixing distribution probabilities must be "
                "broadcastable to (len(x), n_components)"
            )
            raise ValueError(msg)

    def _make_components(self, x: ArrayLike | None = None) -> list[NormalSpline]:
        x = self.x if x is None else x
        return [
            NormalSpline(self._loc_vals[i], self._ln_scale_vals[i], self.knots, x)
            for i in range(self._n_components)
        ]

    def component_sample(
        self,
        key: jax._src.random.KeyArray,
        sample_shape: tuple = (),
        x: ArrayLike | None = None,
    ) -> jax.Array:
        components = self._make_components(x)
        keys = jax.random.split(key, self._n_components)
        samples = [
            d.sample(keys[i], sample_shape, x=x) for i, d in enumerate(components)
        ]
        return jnp.stack(samples, axis=-1)

    def component_log_probs(
        self,
        value: ArrayLike,
        x: ArrayLike | None = None,
    ) -> jax.Array:
        value = jnp.array(value)
        helper = dist.MixtureGeneral(
            self.mixing_distribution, self._make_components(x), validate_args=False
        )
        return helper.component_log_probs(value)


class TruncatedNormal1DSplineMixture(Normal1DSplineMixture):
    def __init__(
        self,
        mixing_distribution: dist.CategoricalProbs | dist.CategoricalLogits,
        loc_vals: ArrayLike,
        ln_scale_vals: ArrayLike,
        knots: ArrayLike,
        x: ArrayLike,
        low: Any | None = None,
        high: Any | None = None,
        spline_k: int = 3,
        *,
        validate_args=None,
    ):
        """
        Represents a mixture of Normal distributions where the parameters are controlled
        by splines that are evaluated at some other parameter values x.

        Parameters
        ----------

        x
            Array of x values at which to evaluate the splines.
        spline_k
            Degree of the spline.
        """
        self.low = low
        self.high = high
        if validate_args and (self.low is None and self.high is None):
            msg = "Use Normal1DSplineMixture if no truncation is needed"
            raise ValueError(msg)

        super().__init__(
            mixing_distribution=mixing_distribution,
            loc_vals=loc_vals,
            ln_scale_vals=ln_scale_vals,
            knots=knots,
            x=x,
            spline_k=spline_k,
            validate_args=validate_args,
        )

    @property
    def support(self):
        if self.low is None:
            return dist.constraints.less_than(self.high)
        if self.high is None:
            return dist.constraints.greater_than(self.low)
        return dist.constraints.interval(self.low, self.high)

    def _make_components(self, x: ArrayLike | None = None) -> list[NormalSpline]:
        x = self.x if x is None else x
        return [
            TruncatedNormalSpline(
                self._loc_vals[i],
                self._ln_scale_vals[i],
                self.knots,
                x,
                low=self.low,
                high=self.high,
            )
            for i in range(self._n_components)
        ]

    def component_log_probs(
        self, value: ArrayLike, x: ArrayLike | None = None
    ) -> jax.Array:
        x = x if x is not None else self.x
        component_log_probs = super().component_log_probs(value, x)
        value = jnp.expand_dims(value, axis=-1)
        return jnp.where(
            self.support.check(value),
            component_log_probs,
            -jnp.inf,
        )
