from typing import Union

import jax
from jax import numpy as jnp


def kernel_morse(
    basis_matrix: jax.Array,
    r: jax.Array,
    alpha: Union[float, jax.Array],
) -> jax.Array:
    return jnp.exp(jnp.dot(basis_matrix, r * (-1.0 / alpha)))


def kernel_reciprocal(
    basis_matrix: jax.Array,
    r: jax.Array,
    ln_alpha: Union[float, jax.Array],
) -> jax.Array:
    return jnp.exp(jnp.dot(basis_matrix, ln_alpha - jnp.log(r)))
