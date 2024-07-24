import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike

__all__ = ["IndependentGMM"]


class IndependentGMM(dist.MixtureSameFamily):
    def __init__(
        self,
        mixing_distribution: dist.CategoricalLogits | dist.CategoricalProbs,
        locs: ArrayLike = 0.0,
        scales: ArrayLike = 1.0,
        low: ArrayLike | None = None,
        high: ArrayLike | None = None,
        *,
        validate_args=True,
    ):
        """
        A Gaussian Mixture Model where the components are fixed to their input locations
        and there are no covariances (but each dimension can have different scales /
        standard deviations).

        Parameters
        ----------
        mixing_distribution
            Distribution over the mixture components.
        locs
            Array of means for each component. This should have shape (D, K) where D is
            the dimensionality of the data and K is the number of mixture components.
        scales
            Array of standard deviations for each component. This should have shape (D,
            K) where D is the dimensionality of the data and K is the number of mixture
            components.
        low
            Lower bounds for each dimension. This should either be a scalar or have
            shape (D,) where D is the dimensionality of the data.
        high
            Upper bounds for each dimension. This should either be a scalar or have
            shape (D,) where D is the dimensionality of the data.
        """
        # K = mixture components, D = dimensions
        # - event_shape is the dimensionality of the data - number of dependent
        #   coordinates, i.e., "D" in the below
        # - batch_shape is the number of independent dimensions - here "K"
        combined_shape = jax.lax.broadcast_shapes(jnp.shape(locs), jnp.shape(scales))
        if len(combined_shape) != 2:
            msg = (
                f"locs and scales must have 2 axes, but got {len(combined_shape)}. The "
                "shape must be: (D, K) where D is the dimensionality of the data and K "
                "is the number of mixture components."
            )
            raise ValueError(msg)
        self._D, self._K = combined_shape

        component_kwargs = {"loc": locs, "scale": scales}
        if low is not None:
            component_kwargs["low"] = low
        if high is not None:
            component_kwargs["high"] = high

        component = dist.TruncatedNormal(**component_kwargs)
        component._batch_shape = (self._D, self._K)
        super().__init__(
            mixing_distribution=mixing_distribution,
            component_distribution=component,
            validate_args=validate_args,
        )
        self._dim_dim = -2

        self._batch_shape = ()
        self._event_shape = (self._D,)

    @property
    def mixture_dim(self):
        return -1

    def component_log_probs(self, value: ArrayLike) -> jax.Array:
        if value.shape[-1] != self._D:
            msg = (
                "The input array must have the same number of coordinate dimensions "
                f"as the distribution. Expected {self._D}, got {value.shape[-2]}."
            )
            raise ValueError(msg)

        tmp = jnp.expand_dims(value, self.mixture_dim)
        component_log_probs = self.component_distribution.log_prob(tmp)

        value = jnp.expand_dims(value, axis=-1)
        return jnp.where(
            self.component_distribution.support.check(value),
            component_log_probs,
            -jnp.inf,
        )

    def log_prob(self, value: ArrayLike) -> jax.Array:
        comp_lp = self.component_log_probs(value)
        return logsumexp(
            jax.nn.log_softmax(self.mixing_distribution.logits)
            + comp_lp.sum(axis=self._dim_dim),
            axis=self.mixture_dim,
        )

    def component_sample(
        self, key: jax.random.PRNGKey, sample_shape: tuple = ()
    ) -> jax.Array:
        return self.component_distribution.sample(
            key,
            sample_shape=sample_shape,  # + self.event_shape
        )

    # def sample_with_intermediates(
    #     self, key: jax.random.PRNGKey, sample_shape: tuple = ()
    # ) -> tuple:
    #     """
    #     A version of ``sample`` that also returns the sampled component indices

    #     Parameters
    #     ----------
    #     key
    #         The rng_key key to be used for the distribution.
    #     sample_shape
    #         The sample shape for the distribution.

    #     Returns
    #     -------
    #     samples
    #         The samples from the distribution.
    #     indices
    #         The indices of the sampled components.
    #     """
    #     key_comp, key_ind = jax.random.split(key)
    #     samples = self.component_sample(key_comp, sample_shape=sample_shape)

    #     # Sample selection indices from the categorical (shape will be sample_shape)
    #     indices = self.mixing_distribution.expand(
    #         sample_shape + self.batch_shape
    #     ).sample(key_ind)
    #     indices_expanded = indices.reshape(indices.shape + (1,))

    #     # Select samples according to indices samples from categorical
    #     samples_selected = jnp.take_along_axis(
    #         samples, indices=indices_expanded, axis=-2
    #     )
    #     samples_selected = jnp.squeeze(samples_selected, axis=-1)

    #     return samples_selected, indices

    # def sample(self, key, sample_shape=()):
    #     return self.sample_with_intermediates(key=key, sample_shape=sample_shape)[0]
