import json
import os
from typing import Optional, Tuple

import equinox as eqx
import jax
from jax import numpy as jnp

from jaxpip.descriptor import PolynomialDescriptor


class PolynomialLinearModel(eqx.Module):
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
            self.coeffs = jnp.zeros(
                shape=(descriptor.feature_dim,),
                dtype=_dtype,
            )

    def update_coeffs(
        self,
        new_coeffs: jax.Array,
    ) -> "PolynomialLinearModel":
        _new_coeffs = jnp.asarray(new_coeffs, dtype=self.dtype)

        return eqx.tree_at(lambda model: model.coeffs, self, _new_coeffs)

    def save(
        self,
        model_file: str = "jaxpip_linear.eqx",
    ) -> str:
        """Save Polynomial Linear Model to path.

        Arguments:
            model_file (str): Path to directory to store model.
                Defaults to "jaxpip_linear.eqx".

        Returns:
            model_file (str): Absolute path to saved model.
        """
        model_file = os.path.abspath(model_file)

        hyperparams = {
            "alpha": float(self.descriptor.alpha),
            "decay_kernel": self.descriptor.decay_kernel,
            "dtype": jnp.dtype(self.dtype).name,
        }

        with open(model_file, "wb") as f:
            f.write((json.dumps(hyperparams) + "\n").encode("utf-8"))
            eqx.tree_serialise_leaves(f, self)

        return model_file

    @classmethod
    def from_file(
        cls,
        basis_file: str,
        model_file: str,
    ) -> "PolynomialLinearModel":
        """Initialize Polynomial Linear Model from file.

        Arguments:
            basis_file (str): Path to basis file (json or json.gz)
            model_file (str): Path to saved model (eqx).

        Returns:
            Polynomial Linear Model
        """
        basis_file = os.path.abspath(basis_file)
        model_file = os.path.abspath(model_file)

        with open(model_file, "rb") as f:
            hyperparams = json.loads(f.readline().decode("utf-8"))

            descriptor = PolynomialDescriptor.from_file(
                basis_file=basis_file,
                alpha=hyperparams.get("alpha", 1.0),
                decay_kernel=hyperparams.get("decay_kernel", "morse"),
                dtype=jnp.dtype(hyperparams.get("dtype", "float64")),
            )

            skeleton = cls(
                descriptor=descriptor,
            )

            model = eqx.tree_deserialise_leaves(f, skeleton)

        return model

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

    def get_hessian(
        self,
        xyz: jax.Array,
    ) -> jax.Array:
        """Get hessian matrix of given xyz coordinates.

        Arguments:
            xyz (jax.Array): Cartesian coordinates with shape (N_atom, 3).

        Returns:
            hess (jax.Array): Hessian matrix with shape (3 * N_atom, 3 * N_atom).
        """
        return jax.hessian(self.get_energy)(xyz).reshape(
            (3 * xyz.shape[0],) * 2
        )


if __name__ == "__main__":
    water_xyz = jnp.array(
        [
            [+0.00000000, +0.78383672, +0.44340501],  # H
            [+0.00000000, -0.78383672, +0.44340501],  # H
            [+0.00000000, +0.00000000, -0.11085125],  # O
        ]
    )

    print(f"HHO xyz = {water_xyz}")

    basis_set = [
        [[0, 0, 0]],
        [[0, 0, 1], [0, 1, 0]],  # r(HO) + r(HO)
        [[1, 0, 0]],  # r(HH)
        [[0, 1, 1]],  # r(HO) * r(HO)
        [[1, 0, 1], [1, 1, 0]],  # r(HH) * r(HO) + r(HH) * r(HO)
        [[0, 0, 2], [0, 2, 0]],  # r(HO)^2 + r(HO)^2
        [[2, 0, 0]],  # r(HH)^2
    ]

    pip_a2b_2 = PolynomialDescriptor(
        basis_set=basis_set,
        alpha=1.0,
        dtype=jnp.float64,
    )

    model = PolynomialLinearModel(
        descriptor=pip_a2b_2,
    )

    print(model)
    print(model.coeffs)
