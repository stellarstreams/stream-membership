import operator
from functools import reduce

import jax.numpy as jnp


def get_grid(low, high, step, pad_num=0, arange=True):
    if arange or not isinstance(step, int):
        return jnp.arange(low - pad_num * step, high + pad_num * step + step, step)
    else:
        return jnp.linspace(low - pad_num * step, high + pad_num * step, step)


def get_from_nested_dict(data, key_list):
    # https://stackoverflow.com/questions/14692690/access-nested-dictionary-items-via-a-list-of-keys
    return reduce(operator.getitem, key_list, data)


def set_in_nested_dict(data, key_list, value):
    get_from_nested_dict(data, key_list[:-1])[key_list[-1]] = value


def del_in_nested_dict(data, key_list):
    del get_from_nested_dict(data, key_list[:-1])[key_list[-1]]
