import jax.numpy as jnp


def make_grid(low, high, step, pad_num=0, arange=True):
    if arange or not isinstance(step, int):
        return jnp.arange(low - pad_num * step, high + pad_num * step + step, step)

    return jnp.linspace(low - pad_num * step, high + pad_num * step, step)
