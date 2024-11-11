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
        scale_vals: ArrayLike,
        knots: ArrayLike,
        x: ArrayLike,
        spline_k: int | dict[str, int] = 3,
        clip_locs: tuple[float | None, float | None] = (None, None),
        clip_scales: tuple[float | None, float | None] = (None, None),
        ordered_scales: bool = True,
        validate_args=None,
    ) -> None:
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
        self.clip_locs = tuple(clip_locs)
        self.clip_scales = tuple(clip_scales)

        # The pre-specified grid to evaluate on
        self.x = jnp.array(x)

        # Spline order:
        if not isinstance(spline_k, dict):
            spline_k = {"loc": spline_k, "scale": spline_k}
        self.spline_k = spline_k

        # Should have shape (n_components, n_knots)
        combined_shape = jax.lax.broadcast_shapes(
            jnp.shape(loc_vals), jnp.shape(scale_vals)
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
        self.scale_vals = jnp.array(scale_vals)

        # Broadcasted arrays:
        self._loc_vals = jnp.broadcast_to(
            self.loc_vals, (self._n_components, self._n_knots)
        )
        self._scale_vals = jnp.broadcast_to(
            self.scale_vals, (self._n_components, self._n_knots)
        )
        if ordered_scales:
            # If specified, treat the scales as cumulatively summed variances
            self._scale_vals = jnp.stack(
                [
                    jnp.sqrt(jnp.sum(self._scale_vals[:i]**2, axis=0)) for i in range(1, self._scale_vals.shape[0] + 1)
                ],
                # [
                #     0.5
                #     * jax.scipy.special.logsumexp(2 * self._ln_scale_vals[:i], axis=0)
                #     for i in range(1, self._scale_vals.shape[0] + 1)
                # ],
                axis=0,
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
            NormalSpline(self._loc_vals[i], self._scale_vals[i], self.knots, x,
                         spline_k=self.spline_k, clip_locs=self.clip_locs, clip_scales=self.clip_scales)
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
        try:
            helper = dist.MixtureSameFamily(
            self.mixing_distribution, self._make_components(x), validate_args=False
        )
        except:
            helper = dist.MixtureGeneral(
                self.mixing_distribution, self._make_components(x), validate_args=False
            )
        return helper.component_log_probs(value)

    def log_prob(self,
                 value: ArrayLike,
                 x: ArrayLike | None = None,
            ) -> jax.Array | Any:
        log_prob = jax.scipy.special.logsumexp(self.component_log_probs(value=value, x=x), axis=-1)
        return log_prob


class TruncatedNormal1DSplineMixture(Normal1DSplineMixture):
    def __init__(
        self,
        mixing_distribution: dist.CategoricalProbs | dist.CategoricalLogits,
        loc_vals: ArrayLike,
        scale_vals: ArrayLike,
        knots: ArrayLike,
        x: ArrayLike,
        low: Any | None = None,
        high: Any | None = None,
        spline_k: int = 3,
        clip_locs: tuple[float | None, float | None] = (None, None),
        clip_scales: tuple[float | None, float | None] = (None, None),
        validate_args=None,
    ) -> None:
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
            scale_vals=scale_vals,
            knots=knots,
            x=x,
            spline_k=spline_k,
            clip_locs=clip_locs,
            clip_scales=clip_scales,
            validate_args=validate_args,
        )

    @property
    def support(self):
        if self.low is None and self.high is None:
            return dist.constraints.real
        elif self.low is None:
            return dist.constraints.less_than(self.high)
        elif self.high is None:
            return dist.constraints.greater_than(self.low)
        else:
            return dist.constraints.interval(self.low, self.high)

    def _make_components(self, x: ArrayLike | None = None) -> list[NormalSpline]:
        x = self.x if x is None else x
        return [
            TruncatedNormalSpline(
                self._loc_vals[i],
                self._scale_vals[i],
                self.knots,
                x,
                low=self.low,
                high=self.high,
                spline_k=self.spline_k,
                clip_locs=self.clip_locs,
                clip_scales=self.clip_scales,
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
