__all__ = ["_StackedModelComponent"]

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike

from .._typing import CoordinateName
from ..model import ModelComponent


class _StackedModelComponent(dist.Distribution):
    def __init__(
        self,
        model_component: ModelComponent,
        pars: dict | None = None,
        overrides: dict[CoordinateName, dist.Distribution] | None = None,
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
        self.overrides = overrides
        self._model_component_dists = self.model_component.make_dists(
            pars, self.overrides
        )

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
        self, key: jax._src.random.KeyArray, sample_shape: tuple = ()
    ) -> jax.Array:
        samples = self.model_component.sample(
            key, sample_shape, overrides=self.overrides
        )
        return jnp.concatenate(
            [jnp.atleast_2d(s.T).T for s in samples.values()], axis=-1
        )
