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

from .model import ModelComponent


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
        self, key: Any, sample_shape: Any = (), x: ArrayLike | None = None
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
        self, key: Any, sample_shape: Any = (), x: ArrayLike | None = None
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


class _StackedModelComponent(dist.Distribution):
    def __init__(self, model_component: ModelComponent, pars: dict | None = None):
        """
        TODO: docstring
        """
        self.model_component = model_component
        super().__init__(
            batch_shape=(),
            event_shape=(len(self.model_component.coord_names),),
        )
        self._model_component_dists = self.model_component.make_dists(pars)

    def component_log_probs(self, value: ArrayLike):
        value = jnp.atleast_2d(value)

        data = {
            coord_name: value[:, i]
            for i, coord_name in enumerate(self.model_component.coord_names)
        }
        extra_data = self.model_component._make_conditional_data(data)

        lps = []
        i = 0
        for coord_name, dist_ in self._model_component_dists.items():
            n = dist_.event_shape[0] if len(dist_.event_shape) > 0 else 1
            lps.append(
                dist_.log_prob(
                    jnp.squeeze(value[:, i : i + n]), **extra_data[coord_name]
                )
            )
            i += n

        return jnp.stack(lps, axis=-1)

    def log_prob(self, value: ArrayLike):
        return jnp.sum(self.component_log_probs(value), axis=-1)

    def sample(self, key: jax.random.PRNGKey, sample_shape: tuple = ()) -> jax.Array:
        samples = self.model_component.sample(key, sample_shape)
        return jnp.concatenate(
            [jnp.atleast_2d(s.T).T for s in samples.values()], axis=-1
        )
