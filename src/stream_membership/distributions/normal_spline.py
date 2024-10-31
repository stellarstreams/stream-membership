__all__ = ["NormalSpline", "TruncatedNormalSpline"]

from typing import Any

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import lax
from jax.typing import ArrayLike
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline


def _clip_preserve_gradients(x, min_, max_):
    return x + lax.stop_gradient(jnp.clip(x, min_, max_) - x)


class NormalSpline(dist.Distribution):
    support = dist.constraints.real

    def __init__(
        self,
        loc_vals: ArrayLike,
        scale_vals: ArrayLike,
        knots: ArrayLike,
        x: ArrayLike,
        spline_k: int | dict[str, int] = 3,
        clip_locs: tuple[float | None, float | None] = (None, None),
        clip_scales: tuple[float | None, float | None] = (None, None),
    ) -> None:
        """
        Represents a Normal distribution where the loc and (log)scale parameters are
        controlled by splines that are evaluated at some other parameter values x. In
        other words, this distribution is conditional on x.

        Parameters
        ----------
        loc_vals
            Array of loc values at the knot locations.
        scale_vals
            Array of scale values at the knot locations.
        knots
            Array of spline knot locations.
        x
            Array of x values at which to evaluate the splines.
        spline_k (optional)
            Degree of the spline. Can be a single integer or a dictionary with keys
            "loc" and "scale" specifying the degrees of the loc and scale splines
            respectively.
        """
        x = jnp.asarray(x)
        super().__init__(batch_shape=x.shape, event_shape=())

        self.knots = jnp.array(knots)
        self.clip_locs = tuple(clip_locs)
        self.clip_scales = tuple(clip_scales)

        if not isinstance(spline_k, dict):
            spline_k = {"loc": spline_k, "scale": spline_k}
        self.spline_k = spline_k
        self.x = x
        self.loc_vals = jnp.array(loc_vals)
        self.scale_vals = jnp.array(scale_vals)

        # TODO: probably a way to vmap here instead...
        if self.loc_vals.ndim == 0:
            self._loc_spl = lambda _: self.loc_vals
        else:
            self._loc_spl = InterpolatedUnivariateSpline(
                self.knots,
                self.loc_vals,
                k=self.spline_k["loc"],
                endpoints="not-a-knot",  # TODO: make this customizable?
            )

        if self.scale_vals.ndim == 0:
            self._scale_spl = lambda _: self.scale_vals
        else:
            self._scale_spl = InterpolatedUnivariateSpline(
                self.knots,
                self.scale_vals,
                k=self.spline_k["scale"],
                endpoints="not-a-knot",  # TODO: make this customizable?
            )

    def _make_helper_dist(self, x: ArrayLike | None = None) -> dist.Normal:
        x = self.x if x is None else x
        return dist.Normal(
            loc=_clip_preserve_gradients(self._loc_spl(x), *self.clip_locs),
            scale=_clip_preserve_gradients(self._scale_spl(x), *self.clip_scales),
        )

    def sample(
        self,
        key: jax._src.random.KeyArray,
        sample_shape: Any = (),
        x: ArrayLike | None = None,
    ) -> jax.Array | Any:
        """
        Draws samples from the distribution.

        Parameters
        ----------
        key
            JAX random number generator key.
        sample_shape
            Shape of the sample.
        x
            Array of x values at which to evaluate the splines. If not provided, the
            x values provided at initialization will be used.
        """
        helper = self._make_helper_dist(x)
        return helper.sample(key=key, sample_shape=sample_shape)

    def log_prob(self, value: ArrayLike, x: ArrayLike | None = None) -> jax.Array | Any:
        """
        Evaluates the log probability density for a batch of samples given by value.

        Parameters
        ----------
        value
            Array of samples to evaluate the log probability for.
        x
            Array of x values at which to evaluate the splines. If not provided, the
            x values provided at initialization will be used.
        """
        helper = self._make_helper_dist(x)
        return helper.log_prob(value)


class TruncatedNormalSpline(NormalSpline):
    def __init__(
        self,
        loc_vals: ArrayLike,
        scale_vals: ArrayLike,
        knots: ArrayLike,
        x: ArrayLike,
        low: Any | None = None,
        high: Any | None = None,
        spline_k: int | dict[str, int] = 3,
        clip_locs: tuple[float | None, float | None] = (None, None),
        clip_scales: tuple[float | None, float | None] = (None, None),
    ) -> None:
        """
        Represents a truncated Normal distribution where the loc and (log)scale
        parameters are controlled by splines that are evaluated at some other parameter
        values x. In other words, this distribution is conditional on x.

        Parameters
        ----------
        loc_vals
            Array of loc values at the knot locations.
        scale_vals
            Array of log scale values at the knot locations.
        knots
            Array of spline knot locations.
        x
            Array of x values at which to evaluate the splines.
        low (optional)
            Lower bound of the distribution.
        high (optional)
            Upper bound of the distribution.
        spline_k (optional)
            Degree of the spline. Can be a single integer or a dictionary with keys
            "loc" and "scale" specifying the degrees of the loc and scale splines
            respectively.
        """
        super().__init__(
            loc_vals=loc_vals,
            scale_vals=scale_vals,
            knots=knots,
            x=x,
            spline_k=spline_k,
        )

        self.low = low
        self.high = high

        self.clip_locs = tuple(clip_locs)
        self.clip_scales = tuple(clip_scales)

    def _make_helper_dist(self, x: ArrayLike | None = None) -> dist.Normal:
        x = self.x if x is None else x
        return dist.TruncatedNormal(
            loc=_clip_preserve_gradients(self._loc_spl(x), *self.clip_locs),
            scale=_clip_preserve_gradients(self._scale_spl(x), *self.clip_scales),
            low=self.low,
            high=self.high,
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
