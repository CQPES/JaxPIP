from typing import List, Tuple, Union

import jax
from jax import jit
from jax import numpy as jnp


class JaxPIP:
    """Jax implementation of Permutational Invariant Polynomials (PIP).

    Attributes:
        basis (List[List[int]]): Permutational invariant basis.
        alpha (float): Range parameter of Morse-like variable.
    """

    def __init__(
        self,
        basis: List[List[int]],
        alpha: float = 1.0,
    ) -> None:
        self.basis = basis
        self.alpha = alpha

    def calc_r_from_xyz(
        self,
        xyz: jax.Array,
        with_grad: bool = False,
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        """Calculate distance vector r from Cartesian coordinates xyz.

        Args:
            xyz (jax.Array): Cartesian coordinates xyz with shape (n, 3), where
                n is the number of atoms, in Angstroms.
            with_grad (bool): Whether to calculate the gradients.
                Defaults to false.

        Returns:
            r (jax.Array): Distance vector with length k = (n * (n - 1) / 2),
                in Angstroms.
            J_r_xyz (jax.Array): Jacobian matrix of distance vector r with
                respect to Cartesian coordinates xyz with shape (k, n, 3), in
                Angstroms. This variable will be calculate only when gradients
                are required, i.e. with_grad = True.
        """
        num_atoms = len(xyz)
        pairwise_idx = jnp.asarray([
            [i, j] for i in range(num_atoms)
            for j in range(i + 1, len(xyz))
        ])
        pairwise_xyz = jnp.asarray([
            [xyz[i], xyz[j]] for (i, j) in pairwise_idx
        ])

        @jax.jit
        def _calc_pairwise_r(atom_1, atom_2):
            return jnp.linalg.norm(atom_1 - atom_2)

        # calculate distance vector r
        r = jax.vmap(_calc_pairwise_r)(
            pairwise_xyz[:, 0, :],
            pairwise_xyz[:, 1, :],
        )

        # calculate Jacobian matrix J_r(xyz) if required
        if with_grad:
            _calc_pairwise_dr_dxyz = jax.grad(
                _calc_pairwise_r,
            )
            dr_dxyz = jax.vmap(_calc_pairwise_dr_dxyz)(
                pairwise_xyz[:, 0, :],
                pairwise_xyz[:, 1, :],
            )

            J_r_xyz = jnp.zeros((len(r), num_atoms, 3))

            for (idx, dr) in enumerate(dr_dxyz):
                J_r_xyz = J_r_xyz.at[idx, pairwise_idx[idx, 0]].set(dr)
                J_r_xyz = J_r_xyz.at[idx, pairwise_idx[idx, 1]].set(-1.0 * dr)

            return r, J_r_xyz

        return r

    def calc_morse_from_r(
        self,
        r: jax.Array,
        with_grad: bool = False,
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        """Calculate Morse-like vector m from distance vector r"""
        pass

    def calc_polynomial_from_morse(
        self,
        morse: jax.Array,
        with_grad: bool = False,
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        """Calculate Permutational Invariant Polynomials from
        Morse-like vector m.
        """
        pass


if __name__ == "__main__":
    xyz = jnp.asarray([
        [-0.37080090, -0.13842560, -0.26234460],
        [+0.52405100, -0.92893340, -0.16449230],
        [-0.75608510, +0.09429450, +0.62230030],
        [+0.06355260, +0.73483450, -0.67571110],
        [-1.14601250, -0.54180340, -1.11155510],
    ])

    ab4_pip = JaxPIP(
        basis=[[]],
        alpha=1.0,
    )

    r, J_r_xyz = ab4_pip.calc_r_from_xyz(
        xyz=xyz,
        with_grad=True,
    )

    print(f"{r = }")
    print(f"{J_r_xyz = }")
