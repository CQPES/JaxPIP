from typing import Tuple, Union, Optional

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
        coeffs: Optional[jax.Array] = None,
    ) -> None:
        self.descriptor = descriptor

        # inherit descriptor's dtype
        self.dtype = descriptor.dtype
        _dtype = descriptor.dtype

        if coeffs is not None:
            self.coeffs = jnp.asarray(coeffs, dtype=_dtype)
        else:
            self.coeffs = jnp.zeros(shape=(descriptor.num_pips,), dtype=_dtype)

    def update_coeffs(
        self,
        new_coeffs: jax.Array,
    ) -> "PolynomialLinearModel":
        _new_coeffs = jnp.asarray(new_coeffs, dtype=self.dtype)

        return eqx.tree_at(lambda model: model.coeffs, self, _new_coeffs)

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


if __name__ == "__main__":
    water_xyz = jnp.array([
        [+0.00000000, +0.78383672, +0.44340501],  # H
        [+0.00000000, -0.78383672, +0.44340501],  # H
        [+0.00000000, +0.00000000, -0.11085125],  # O
    ])

    print(f"HHO xyz = {water_xyz}")

    basis_set = [
        [[0, 0, 0]],
        [[0, 0, 1], [0, 1, 0]],  # r(HO) + r(HO)
        [[1, 0, 0]],             # r(HH)
        [[0, 1, 1]],             # r(HO) * r(HO)
        [[1, 0, 1], [1, 1, 0]],  # r(HH) * r(HO) + r(HH) * r(HO)
        [[0, 0, 2], [0, 2, 0]],  # r(HO)^2 + r(HO)^2
        [[2, 0, 0]],             # r(HH)^2
    ]

    pip_a2b_2 = PolynomialDescriptor(
        basis_set=basis_set,
        alpha=1.0,
        with_grad=False,
        dtype=jnp.float64,
    )

    model = PolynomialLinearModel(
        descriptor=pip_a2b_2,
    )

    print(model)
    print(model.coeffs)
