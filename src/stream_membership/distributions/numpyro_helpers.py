__all__ = ["ConcatenatedConstraints", "ConcatenatedTransforms"]

import jax
import jax.numpy as jnp
from numpyro.distributions.constraints import Constraint
from numpyro.distributions.transforms import Transform, biject_to

from ..utils import slice_along_axis


class ConcatenatedConstraints(Constraint):
    """TODO"""

    def __init__(
        self,
        constraints: list[Constraint],
        sizes: list[int] | None = None,
    ):
        # TODO: validate event dims are the same?

        self.constraints = constraints
        self.sizes = sizes
        super().__init__()

    @property
    def event_dim(self):
        # We just grab the first constraint, because all event dims should be the same
        return self.constraints[0].event_dim + self.reinterpreted_batch_ndims

    def __call__(self, value):
        # NOTE: assumption that axis=-1 is the event dimension
        # TODO: will this fail with a 2D like GMM combined with a 1D like Spline?
        assert value.shape[-1] == len(self.constraints)
        results = jnp.concatenate(
            [
                jnp.atleast_2d(constraint(value[..., i]))
                for i, constraint in enumerate(self.constraints)
            ],
        )
        return results.all(0)

    def __repr__(self):
        return (
            f"{self.__class__.__name__[1:]}({len(self.constraints)} constraints, "
            f"{self.reinterpreted_batch_ndims})"
        )

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
        print("WTF", x.shape, self.size)
        assert x.shape[self.axis] == self.size

        vals = []
        for i, trans, size in self._iter_transforms():
            vals.append(trans(slice_along_axis(x, (i, i + size), axis=-1).T))
        return jnp.concatenate(vals, axis=self.axis)

    def _inverse(self, y):
        print("y WTF", y.shape, self.size)
        vals = []
        for i, trans, size in self._iter_transforms():
            vals.append(trans.inv(slice_along_axis(y, (i, i + size), axis=-1).T))
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
    return ConcatenatedTransforms(transforms, sizes=constraint.sizes, axis=0)
