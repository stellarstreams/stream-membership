__all__ = ["ConcatenatedDistributions"]

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike
from numpyro.distributions.constraints import Constraint
from numpyro.distributions.transforms import Transform, biject_to

from stream_membership.utils import atleast_2d, slice_along_axis


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
            samples = jnp.atleast_1d(dist_.sample(key_, sample_shape=sample_shape))
            if sample_shape == ():
                samples = atleast_2d(samples, axis=0)

            all_samples.append(samples)

        max_ndim = len(sample_shape) + 1
        all_samples = [
            jnp.expand_dims(s, -1) if len(s.shape) < max_ndim else s
            for s in all_samples
        ]
        return jnp.concatenate(all_samples, axis=-1).reshape((*sample_shape, -1))


class ConcatenatedConstraints(Constraint):
    def __init__(
        self,
        constraints: list[Constraint],
        sizes: list[int] | None = None,
    ):
        """Represents a concatenation of independent constraints."""
        self.constraints = constraints

        # validate event dims are the same
        # assert all(c.event_dim == constraints[0].event_dim for c in constraints)

        self.sizes = sizes
        super().__init__()

    @property
    def event_dim(self):
        # # We just grab the first constraint, because all event dims should be the same
        # return self.constraints[0].event_dim

        # TODO: can we safely assume this?
        return -1

    def __call__(self, value):
        assert value.shape[self.event_dim] == sum(self.sizes)

        i = 0
        results = []
        for constraint, size in zip(self.constraints, self.sizes, strict=True):
            results.append(
                constraint(
                    slice_along_axis(value, slc=(i, i + size), axis=self.event_dim)
                )
            )
            i += size
        results = jnp.concatenate(results, axis=self.event_dim)
        return results.all(axis=self.event_dim)

    def __repr__(self):
        return f"{self.__class__.__name__[1:]}({len(self.constraints)} constraints"

    def feasible_like(self, prototype):
        return jnp.stack(
            [constraint.feasible_like(prototype) for constraint in self.constraints], -1
        )

    def tree_flatten(self):
        return (self.constraints, self.sizes), (("constraints", "sizes"),)

    def __eq__(self, other):
        if not isinstance(other, ConcatenatedConstraints):
            return False

        return all(
            c1 == c2
            for c1, c2 in zip(self.constraints, other.constraints, strict=False)
        )


class ConcatenatedTransforms(Transform):
    def __init__(
        self,
        transforms: list[Transform],
        axis: int = 0,
        sizes: list[int] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        transforms
            A list of numpyro transform instances to concatenate.
        axis
            The axis to concatenate over.
        sizes (optional)
            A list of sizes of input values along the concatenating axis.
        """
        self.transforms = transforms
        for t in self.transforms:
            assert isinstance(t, Transform)

        self.axis = int(axis)
        self.sizes = sizes if sizes is not None else [1] * len(self.transforms)
        assert len(self.sizes) == len(self.transforms)

    @property
    def size(self) -> int:
        return sum(self.sizes)

    def _iter_transforms(self):
        i = 0
        for trans, size in zip(self.transforms, self.sizes, strict=False):
            yield i, trans, size
            i += size

    def __call__(self, x: jax.Array) -> jax.Array:
        assert x.shape[self.axis] == self.size

        vals = []
        for i, trans, size in self._iter_transforms():
            vals.append(trans(slice_along_axis(x, (i, i + size), axis=-1)))
        return jnp.concatenate(vals, axis=self.axis)

    def _inverse(self, y: jax.Array) -> jax.Array:
        assert y.shape[self.axis] == self.size

        vals = []
        for i, trans, size in self._iter_transforms():
            vals.append(trans.inv(slice_along_axis(y, (i, i + size), axis=-1)))
        return jnp.concatenate(vals, axis=self.axis)

    def tree_flatten(self):
        return (self.transforms, self.axis, self.sizes), (
            ("transforms", "axis", "sizes"),
            {},
        )

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        jacs = []
        for i, trans, size in self._iter_transforms():
            xx = slice_along_axis(x, (i, i + size))
            yy = slice_along_axis(y, (i, i + size))
            jacs.append(trans.log_abs_det_jacobian(xx, yy, intermediates=intermediates))

        return jnp.concatenate(jacs, axis=self.axis)


@biject_to.register(ConcatenatedConstraints)
def _transform_to_concatenated(constraint):
    transforms = [biject_to(c) for c in constraint.constraints]
    return ConcatenatedTransforms(transforms, sizes=constraint.sizes, axis=-1)
