__all__ = ["DirichletSpline"]

from typing import Any

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline


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

        self.spline_k = int(spline_k)

        # Should have shape (n_components, n_knots)
        self.concentration_vals = jnp.array(concentration_vals)
        if jnp.ndim(concentration_vals) != 2:
            msg = "`concentration_vals` must be two dimensional."
            raise ValueError(msg)

        self._spl = jax.vmap(eval_spline, in_axes=(None, 0, None, None))
        conc = self._spl(self.knots, self.concentration_vals, self.x, self.spline_k)

        super().__init__(
            concentration=self._clean_conc(conc),
            validate_args=validate_args,
        )

    @staticmethod
    def _clean_conc(conc):
        # TODO: need to make sure conc values are > 0 -- either do this by passing
        # values through a transform, like sigmoid? Or like here, where we just
        # threshold:
        fl = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
        return jnp.where(conc > 0, conc, jnp.finfo(fl).tiny).T

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
        if x is None:
            return super().sample(key, sample_shape)

        conc = self._spl(self.knots, self.concentration_vals, x, self.spline_k)
        obj = dist.Dirichlet(concentration=self._clean_conc(conc))
        return obj.sample(key, sample_shape)

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
        if x is None:
            return super().log_prob(value)

        conc = self._spl(self.knots, self.concentration_vals, x, self.spline_k)
        obj = dist.Dirichlet(concentration=self._clean_conc(conc))
        return obj.log_prob(value)
