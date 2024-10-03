from typing import Any

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
