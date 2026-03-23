import equinox as eqx
import jax
from jax import numpy as jnp


class ISRU(eqx.Module):
    alpha: float = eqx.field(static=True)

    def __init__(
        self,
        alpha: float = 1.0,
    ) -> None:
        self.alpha = float(alpha)

    def __call__(
        self,
        x: jax.Array,
        key: jax.Array = None,
    ) -> jax.Array:
        return x * jax.lax.rsqrt(1.0 + self.alpha * jnp.square(x))

    def __repr__(self) -> str:
        return f"ISRU(alpha={self.alpha})"
