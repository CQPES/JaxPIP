import json
from typing import List, Tuple, Union

import jax
from jax import numpy as jnp

from jaxpip.descriptor._abc import AbstractDescriptor
from jaxpip.descriptor._libpip import (_calc_m_from_r, _calc_p_from_m,
                                       _calc_r_from_xyz)


class PIPDescriptor(AbstractDescriptor):
    """Permutation invariant polynomial descriptor.

    Attributes:
        basis_set (List[jax.Array]): Permutation invariant basis set.
        alpha (float): Range parameter of Morse-like variables.
    """

    def __init__(
        self,
        basis_set: List[jax.Array],
        alpha: float = 1.0,
    ) -> None:
        self.basis_set = basis_set
        self.alpha = alpha

    @staticmethod
    def from_json(
        basis_json: str,
        alpha: float = 1.0,
    ) -> "PIPDescriptor":
        """Load PIP basis from json file.

        Arguments:
            basis_json (str): Permutation invariant basis json file.
            alpha (float): Range parameter of Morse-like variables.
                Defaults to 1.0.
        """
        with open(basis_json) as f:
            basis_set = json.load(f)

        basis_set = [jnp.array(basis, dtype=jnp.float32)
                     for basis in basis_set]

        return PIPDescriptor(basis_set, alpha)

    def __call__(
        self,
        xyz: jax.Array,
        with_grad: bool = False
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        """Calculate permutation invariant polynomial.

        Arguments:
            xyz (jax.Array): Cartesian coordinates `xyz` with shape (n, 3),
                where `n` is the number of atoms, in Angstroms.
            with_grad (bool): Whether to calculate the gradients.
                Defaults to false.

        Returns:
            p (jax.Array): Permutation invariant polynomial.
            J_p_xyz (jax.Array): Jacobian matrix of permutation invariant
                polynomial `p` with respect to Cartesian coordinates `xyz`.
                Only be calculated when gradients are required, i.e.
                with_grad = True.
        """
        if not with_grad:
            r = _calc_r_from_xyz(xyz)
            m = _calc_m_from_r(r, alpha=self.alpha)
            p = _calc_p_from_m(m, self.basis_set)

            return p

        # calculate Jacobian matrix J_p(xyz) if required
        r, J_r_xyz = _calc_r_from_xyz(xyz, with_grad=True)
        m, J_m_r = _calc_m_from_r(r, alpha=self.alpha, with_grad=True)
        p, J_p_m = _calc_p_from_m(m, self.basis_set, with_grad=True)

        J_p_r = jnp.einsum("ij,jk->ik", J_p_m, J_m_r)
        J_p_xyz = jnp.einsum("ij,jkl->ikl", J_p_r, J_r_xyz)

        return p, J_p_xyz

    def __repr__(self) -> str:
        """Return a string representation."""
        len_morse = len(self.basis_set[0][0])
        num_atoms = int((1 + (1 + 8 * len_morse)**0.5) / 2)
        len_pip = len(self.basis_set)
        return ">>> PIPDescriptor INFO\n" \
            f">>> Number of atoms: {num_atoms}\n" \
            f">>> Length of distance and Morse-like vectors: {len_morse}\n" \
            f">>> Length of permutation invariant polynomial: {len_pip}"


if __name__ == "__main__":
    water_xyz = jnp.asarray([
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

    for (idx, basis) in enumerate(basis_set):
        basis_set[idx] = jnp.asarray(basis, dtype=jnp.float32)

    ab4_pip = PIPDescriptor(
        basis_set=basis_set,
        alpha=1.0,
    )

    p, J_p_xyz = ab4_pip(water_xyz, with_grad=True)

    print(f"p(xyz) = {p}")
    print(f"Jp(xyz) = {J_p_xyz}")
