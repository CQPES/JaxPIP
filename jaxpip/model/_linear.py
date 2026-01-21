from typing import Tuple, Union

import equinox as eqx
import jax
from jax import numpy as jnp

from jaxpip.descriptor import PolynomialDescriptor
from jaxpip.model import AbstractModel


class PolynomialLinearModel(eqx.Module, AbstractModel):
    """Polynomial Linear Model

    Attributes:
        descriptor (PolynomialDescriptor): Polynomial based descriptor.
        coeffs (jax.Array): Linear fitting coefficients.
        dtype (jnp.dtype): Data type inherit from descriptor.
    """
    descriptor: PolynomialDescriptor = eqx.field(static=True)
    coeffs: jax.Array

    dtype: jnp.dtype = eqx.field(static=True)

    def __init__(
        self,
        descriptor: PolynomialDescriptor,
    ) -> None:
        self.descriptor = descriptor

        # inherit descriptor's dtype
        self.dtype = descriptor.dtype
        _dtype = descriptor.dtype

        self.coeffs = jnp.zeros(shape=(descriptor.num_pips,), dtype=_dtype)

    @eqx.filter_jit
    def get_energy(
        self,
        xyz: jax.Array,
    ) -> jax.Array:
        """Get potential energy of given xyz coordinates.

        Arguments:
            xyz (jax.Array): Cartesian coordinates with shape (N_atom, 3).

        Returns:
            V (jax.Array): Potential energy.
        """
        p = self.descriptor(xyz)
        V = jnp.dot(self.coeffs, p)

        return V

    @eqx.filter_jit
    def get_energy_and_forces(
        self,
        xyz: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """Get potential energy and forces of given xyz coordinates.

        Arguments:
            xyz (jax.Array): Cartesian coordinates with shape (N_atom, 3).

        Returns:
            V (jax.Array): Potential energy.
            f (jax.Array): Cartesian forces with shape (N_atom, 3).
        """
        V, g = jax.value_and_grad(self.get_energy)(xyz)

        # handle (possible) net force problems
        f = -(g - jnp.mean(g, axis=0))

        return V, f

    def __call__(
        self,
        xyz: jax.Array,
        with_force: bool = False,
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        if xyz.ndim == 3:
            if with_force:
                return jax.vmap(self.get_energy_and_forces)(xyz)

            return jax.vmap(self.get_energy)(xyz)

        if with_force:
            return self.get_energy_and_forces(xyz)

        return self.get_energy(xyz)
