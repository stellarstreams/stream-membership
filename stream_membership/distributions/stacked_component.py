__all__ = ["_StackedModelComponent"]

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike

from .._typing import CoordinateName
from ..model import ModelComponent


class vector_interval(dist.constraints.Constraint):
    def __init__(self, lower_bounds: ArrayLike, upper_bounds: ArrayLike):
        self.lower_bounds = jnp.array(lower_bounds)
        self.upper_bounds = jnp.array(upper_bounds)
        if self.lower_bounds.shape != self.upper_bounds.shape:
            msg = "Lower and upper bounds must have the same shape."
            raise ValueError(msg)

    def __call__(self, x):
        return jnp.all((x > self.lower_bounds) & (x < self.upper_bounds), axis=-1)

    def tree_flatten(self):
        return (self.lower_bounds, self.upper_bounds), (
            ("lower_bounds", "upper_bounds"),
            {},
        )

    def feasible_like(self, prototype):
        values = (self.lower_bounds + self.upper_bounds) / 2
        values[jnp.isinf(self.lower_bounds)] = self.upper_bounds[
            jnp.isinf(self.lower_bounds)
        ] + 2 * jnp.abs(self.upper_bounds[jnp.isinf(self.lower_bounds)])
        values[jnp.isinf(self.upper_bounds)] = self.lower_bounds[
            jnp.isinf(self.lower_bounds)
        ] + 2 * jnp.abs(self.lower_bounds[jnp.isinf(self.lower_bounds)])
        return jnp.broadcast_to(values, jax.numpy.shape(prototype))


class _StackedModelComponent(dist.Distribution):
    # TODO: set event shape to correct value??

    def __init__(
        self,
        model_component: ModelComponent,
        pars: dict | None = None,
        dists: dict[CoordinateName, dist.Distribution] | None = None,
    ):
        """
        NOTE: for internal use only.

        TODO: docstring
        """
        self.model_component = model_component
        super().__init__(
            batch_shape=(),
            event_shape=(len(self.model_component.coord_names),),
        )
        self.pars = pars
        self._inputted_dists = dists
        self._model_component_dists = self.model_component.make_dists(
            pars, self._inputted_dists
        )

        # Set up the support
        lows = []
        highs = []
        for dist_ in self._model_component_dists.values():
            low = jnp.atleast_1d(
                jnp.squeeze(getattr(dist_.support, "lower_bound", -jnp.inf))
            )
            high = jnp.atleast_1d(
                jnp.squeeze(getattr(dist_.support, "upper_bound", jnp.inf))
            )
            low, high = jnp.broadcast_arrays(low, high)
            lows.append(low)
            highs.append(high)
        lower = jnp.concatenate(lows)
        upper = jnp.concatenate(highs)
        self._support = vector_interval(lower, upper)

    @property
    def support(self):
        return self._support

    def component_log_probs(self, value: ArrayLike):
        value = jnp.atleast_2d(value)

        data: dict[str, jax.Array] = {
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

    def sample(
        self,
        key: jax._src.random.KeyArray,
        sample_shape: tuple = (),
    ) -> jax.Array:
        samples = self.model_component.sample(
            key,
            sample_shape,
            dists=self._model_component_dists,
        )
        if not sample_shape:
            return jnp.concatenate([jnp.atleast_1d(s) for s in samples.values()])
        else:
            # The shape to use for non-joint samples:
            samples_list = [
                s.reshape((*sample_shape, 1))
                if isinstance(name, str)
                else s.reshape((*sample_shape, 2))
                for name, s in samples.items()
            ]

            return jnp.concatenate(
                [jnp.atleast_2d(s.T) for s in samples_list], axis=0
            ).T
