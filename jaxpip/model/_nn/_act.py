import jax
from jax import numpy as jnp
import equinox as eqx


def isru(
    x: jax.Array,
    alpha: float = 1.0,
) -> jax.Array:
    return x * jax.lax.rsqrt(1.0 + alpha * jnp.square(x))


class ISRULayer(eqx.Module):
    alpha: float = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)

    def __init__(
        self,
        alpha: float = 1.0,
        dtype: jnp.dtype = jnp.float64,
    ) -> None:
        self.alpha = float(alpha)
        self.dtype = dtype

    def __call__(
        self,
        x: jax.Array,
    ) -> jax.Array:
        _x = x.astype(self.dtype)

        return _x * jax.lax.rsqrt(
            jnp.array(1.0, dtype=self.dtype) + self.alpha * jnp.square(_x)
        )

    def __repr__(self):
        return f"isru(alpha={self.alpha})"


class TanhLayer(eqx.Module):
    dtype: jnp.dtype = eqx.field(static=True)

    def __init__(
        self,
        dtype: jnp.dtype = jnp.float64,
    ) -> None:
        self.dtype = dtype

    def __call__(
        self,
        x: jax.Array,
    ) -> jax.Array:
        _x = x.astype(self.dtype)

        return jax.nn.tanh(_x)

    def __repr__(self):
        return "tanh"
