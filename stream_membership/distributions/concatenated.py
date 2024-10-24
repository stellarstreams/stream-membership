__all__ = ["ConcatenatedDistributions"]

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike

from ..utils import atleast_2d
from .numpyro_helpers import ConcatenatedConstraints


class ConcatenatedDistributions(dist.Distribution):
    def __init__(self, dists: list[dist.Distribution]) -> None:
        """Make a single multi-dimensional distribution from a list of distributions."""

        self._dists = dists

        # Set up the distribution support
        self._sizes = [
            dist_.event_shape[0] if dist_.event_shape else 1 for dist_ in dists
        ]
        self._support = ConcatenatedConstraints(
            [dist_.support for dist_ in dists],
            sizes=self._sizes,
        )

        # TODO: need to specify event_dim = -1?
        super().__init__(
            batch_shape=(),
            event_shape=(sum(self._sizes),),
        )

    @property
    def support(self):
        return self._support

    def component_log_probs(self, value: ArrayLike) -> jax.Array:
        """Compute the log probability of each component distribution.

        TODO: this currently ignores the conditional data needs. Do we need it, though?
        The conditional data for the spline distributions should be set when the
        distributions are defined, I think...
        """
        value = jnp.asarray(value)
        assert value.shape[-1] == self.event_shape[0]

        lps = []

        i = 0
        for dist_, size in zip(self._dists, self._sizes, strict=True):
            lps.append(atleast_2d(dist_.log_prob(value[..., i : i + size]), axis=-1))
            i += size

        return jnp.concatenate(lps, axis=-1)

    def log_prob(self, value: ArrayLike) -> jax.Array:
        """Compute the log probability of the concatenated distribution."""
        return jnp.sum(self.component_log_probs(value), axis=-1)

    def sample(
        self,
        key: jax._src.random.KeyArray,
        sample_shape: tuple = (),
    ) -> jax.Array:
        """Sample from the concatenated distribution."""
        keys = jax.random.split(key, len(self._dists))

        all_samples = []
        for key_, dist_ in zip(keys, self._dists, strict=True):
            all_samples.append(
                jnp.atleast_1d(dist_.sample(key_, sample_shape=sample_shape))
            )

        max_ndim = max(len(s.shape) for s in all_samples)
        all_samples = [
            jnp.expand_dims(s, -1) if len(s.shape) != max_ndim else s
            for s in all_samples
        ]
        return jnp.concatenate(all_samples, axis=-1)
