__all__ = ["_ConcatenatedModelComponent"]

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike
from numpyro.distributions.constraints import Constraint
from numpyro.distributions.transforms import Transform, biject_to

from .._typing import CoordinateName
from ..model import ModelComponent


class _IndependentConstraints(Constraint):
    """
    Wraps a constraint by aggregating over ``reinterpreted_batch_ndims``-many
    dims in :meth:`check`, so that an event is valid only if all its
    independent entries are valid.
    """

    def __init__(self, constraints):
        for constraint in constraints:
            assert isinstance(constraint, Constraint)
        self.constraints = constraints
        super().__init__()

    @property
    def event_dim(self):
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
        return (self.constraints,), (("constraints",),)

    def __eq__(self, other):
        if not isinstance(other, _IndependentConstraints):
            return False

        return all(
            c1 == c2
            for c1, c2 in zip(self.constraints, other.constraints, strict=False)
        )


class ConcatenatedTransform(Transform):
    def __init__(self, transforms: list[Transform]):
        self.transforms = transforms

    def __call__(self, x):
        vals = []
        # TODO: will this fail with a 2D like GMM combined with a 1D like Spline?
        # Because then need to do, e.g., i:i+2 instead
        for i, trans in enumerate(self.transforms):
            vals.append(jnp.atleast_2d(trans(x[..., i])))
        # return jnp.stack(vals, axis=-1)
        return jnp.concatenate(vals).T

    def _inverse(self, y):
        vals = []
        for i, trans in enumerate(self.transforms):
            vals.append(jnp.atleast_2d(trans.inv(y[..., i])))
        # return jnp.stack(vals, axis=-1)
        return jnp.concatenate(vals).T

    def tree_flatten(self):
        return (self.transforms,), (("transforms",), {})

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        # TODO: will this fail with a 2D like GMM combined with a 1D like Spline?
        jacs = [
            trans.log_abs_det_jacobian(
                x[..., i : i + 1], y[..., i : i + 1], intermediates=intermediates
            )
            for i, trans in enumerate(self.transforms)
        ]
        return jnp.concatenate(jacs, axis=-1)


@biject_to.register(_IndependentConstraints)
def _transform_to_concatenated(constraint):
    transforms = [biject_to(c) for c in constraint.constraints]
    return ConcatenatedTransform(transforms)


class _ConcatenatedModelComponent(dist.Distribution):
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

        # TODO: at the end of the day, the way to have uncertainties in the model might
        # be to add a mechanism to "inject" stddev into the relevant component
        # distributions here...
        self._model_component_dists = self.model_component.make_dists(
            pars, self._inputted_dists
        )

        # Set up the distribution support
        self._support = _IndependentConstraints(
            [dist_.support for dist_ in self._model_component_dists.values()]
        )

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
