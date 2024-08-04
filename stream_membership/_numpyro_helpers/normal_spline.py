__all__ = [
    "NormalSpline",
    "TruncatedNormalSpline",
    "DirichletSpline",
]

from typing import Any

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline


class NormalSpline(dist.Distribution):
    support = dist.constraints.real

    def __init__(
        self,
        loc_vals: ArrayLike,
        ln_scale_vals: ArrayLike,
        knots: ArrayLike,
        x: ArrayLike,
        spline_k: int | dict[str, int] = 3,
    ) -> None:
        """
        Represents a Normal distribution where the loc and (log)scale parameters are
        controlled by splines that are evaluated at some other parameter values x. In
        other words, this distribution is conditional on x.

        Parameters
        ----------
        loc_vals
            Array of loc values at the knot locations.
        ln_scale_vals
            Array of log scale values at the knot locations.
        knots
            Array of spline knot locations.
        x
            Array of x values at which to evaluate the splines.
        spline_k (optional)
            Degree of the spline. Can be a single integer or a dictionary with keys
            "loc" and "ln_scale" specifying the degrees of the loc and ln_scale splines
            respectively.
        """

        super().__init__(batch_shape=(), event_shape=())

        self.knots = jnp.array(knots)

        if not isinstance(spline_k, dict):
            spline_k = {"loc": spline_k, "ln_scale": spline_k}
        self.spline_k = spline_k
        self.x = jnp.array(x)
        self.loc_vals = jnp.array(loc_vals)
        self.ln_scale_vals = jnp.array(ln_scale_vals)

        if self.loc_vals.ndim == 0:
            self._loc_spl = lambda _: self.loc_vals
        else:
            self._loc_spl = InterpolatedUnivariateSpline(
                self.knots,
                self.loc_vals,
                k=self.spline_k["loc"],
                endpoints="not-a-knot",  # TODO: make this customizable?
            )

        if self.ln_scale_vals.ndim == 0:
            self._ln_scale_spl = lambda _: self.ln_scale_vals
        else:
            self._ln_scale_spl = InterpolatedUnivariateSpline(
                self.knots,
                self.ln_scale_vals,
                k=self.spline_k["ln_scale"],
                endpoints="not-a-knot",  # TODO: make this customizable?
            )

    def _make_helper_dist(self, x: ArrayLike | None = None) -> dist.Normal:
        x = self.x if x is None else x
        return dist.Normal(loc=self._loc_spl(x), scale=jnp.exp(self._ln_scale_spl(x)))

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
        ln_scale_vals: ArrayLike,
        knots: ArrayLike,
        x: ArrayLike,
        low: Any | None = None,
        high: Any | None = None,
        spline_k: int | dict[str, int] = 3,
    ) -> None:
        """
        Represents a truncated Normal distribution where the loc and (log)scale
        parameters are controlled by splines that are evaluated at some other parameter
        values x. In other words, this distribution is conditional on x.

        Parameters
        ----------
        loc_vals
            Array of loc values at the knot locations.
        ln_scale_vals
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
            "loc" and "ln_scale" specifying the degrees of the loc and ln_scale splines
            respectively.
        """
        super().__init__(
            loc_vals=loc_vals,
            ln_scale_vals=ln_scale_vals,
            knots=knots,
            x=x,
            spline_k=spline_k,
        )

        self.low = low
        self.high = high

    def _make_helper_dist(self, x: ArrayLike | None = None) -> dist.Normal:
        x = self.x if x is None else x
        return dist.TruncatedNormal(
            loc=self._loc_spl(x),
            scale=jnp.exp(self._ln_scale_spl(x)),
            low=self.low,
            high=self.high,
        )
