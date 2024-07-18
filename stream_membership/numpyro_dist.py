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

        self._loc_spl = InterpolatedUnivariateSpline(
            self.knots,
            self.loc_vals,
            k=self.spline_k["loc"],
            endpoints="not-a-knot",  # TODO: make this customizable?
        )

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
        self, key: Any, sample_shape: Any = (), x: ArrayLike | None = None
    ) -> jax.Array | Any:
        helper = self._make_helper_dist(x)
        return helper.sample(key=key, sample_shape=sample_shape)

    def log_prob(self, value: ArrayLike, x: ArrayLike | None = None) -> jax.Array | Any:
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


def eval_spline(knots: jax.Array, vals: jax.Array, x: jax.Array, k: int):
    spl = InterpolatedUnivariateSpline(knots, vals, k=k, endpoints="not-a-knot")
    return spl(x)


class DirichletSpline(dist.Dirichlet):
    def __init__(
        self,
        concentration_vals: ArrayLike,
        knots: ArrayLike,
        x: ArrayLike,
        spline_k: int = 3,
        *,
        validate_args=None,
    ):
        """
        Represents a Dirichlet distribution where the concentration parameters are
        controlled by splines that are evaluated at some other parameter values x.

        Parameters
        ----------
        concentration_vals
            Array of concentration values at the knot locations.
        knots
            Array of spline knot locations.
        x
            Array of x values at which to evaluate the splines.
        spline_k
            Degree of the spline.
        """
        # Should have shape (n_knots, )
        self.knots = jnp.array(knots)

        # Can have any shape
        self.x = jnp.array(x)

        # Should have shape (n_components, n_knots)
        self.concentration_vals = jnp.array(concentration_vals)
        if jnp.ndim(concentration_vals) != 2:
            msg = "`concentration_vals` must be two dimensional."
            raise ValueError(msg)

        spl = jax.vmap(eval_spline, in_axes=(None, 0, None, None))
        conc = spl(self.knots, self.concentration_vals, self.x, spline_k)

        # TODO: need to make sure conc values are > 0 -- either do this by passing
        # values through a transform, like sigmoid? Or like here, where we just
        # threshold:
        fl = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
        conc = jnp.where(conc > 0, conc, jnp.finfo(fl).tiny).T

        super().__init__(
            concentration=conc,
            validate_args=validate_args,
        )
