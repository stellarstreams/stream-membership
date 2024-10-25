from typing import Any

import jax
import jax.numpy as jnp

from ._typing import CoordinateName


def make_grid(low, high, step, pad_num=0, arange=True):
    if arange or not isinstance(step, int):
        return jnp.arange(low - pad_num * step, high + pad_num * step + step, step)

    return jnp.linspace(low - pad_num * step, high + pad_num * step, step)


def get_coord_from_data_dict(
    name: CoordinateName, data: dict[CoordinateName, Any]
) -> Any:
    """Retrieve a named coordinate or name pair from a dictionary of data.

    This handles a case where the data dict contains a joint coordinate, like (phi1,
    phi2), and we just want to get the phi1 data, for example.
    """
    for key in data:
        if name == key:
            return data[key]
        elif isinstance(key, tuple) and name in key:
            break
    else:
        return None

    idx = key.index(name)
    return data[key][..., idx]


def slice_along_axis(arr: jax.Array, slc: int | tuple | slice, axis=-1) -> jax.Array:
    """Dynamically slice the input array along an axis

    Parameters
    ----------
    arr
        The data array to slice.
    slc
        A specification of how to slice, either by picking an index or by specifying a
        range of indices as an iterable.
    axis
        The axis to slice over.
    """
    shape = jnp.shape(arr)
    axis = axis + len(shape) if axis < 0 else axis
    slc = slice(*slc) if isinstance(slc, tuple | list) else slice(slc)

    all_slice = len(shape) * [
        slice(None),
    ]
    all_slice[axis] = slc

    return arr[tuple(all_slice)]


def atleast_2d(arr: jax.Array, axis: int = 0) -> jax.Array:
    """Ensure that the input array has at least 2 dimensions."""
    if len(jnp.shape(arr)) == 0:
        return jnp.atleast_2d(arr)
    elif len(jnp.shape(arr)) == 1:
        return jnp.expand_dims(arr, axis)
    return arr
