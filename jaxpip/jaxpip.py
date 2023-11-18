import json
from typing import List, Tuple, Union

import jax
from jax import numpy as jnp


class JaxPIPDescriptor:
    """Jax Permutational Invariant Polynomials (PIP) Descriptor.

    Attributes:
        basis (List[List[int]]): Permutational invariant basis.
        alpha (float): Range parameter of Morse-like variables.
    """

    def __init__(
        self,
        basis: List[List[int]],
        alpha: float = 1.0,
    ) -> None:
        self.basis = basis
        self.alpha = alpha

    @staticmethod
    def from_json(
        basis_json: str,
        alpha: float = 1.0,
    ) -> "JaxPIPDescriptor":
        """Load PIP basis from json file.

        Args:
            basis_json (str): Json file containing PIP basis, generated using
                `BAS2json.py`.
            alpha (float): Range parameter of Morse-like variables.
        """
        with open(basis_json) as f:
            basis = json.load(f)

        return JaxPIPDescriptor(basis, alpha)

    def calc_r_from_xyz(
        self,
        xyz: jax.Array,
        with_grad: bool = False,
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        """Calculate distance vector `r` from Cartesian coordinates `xyz`.

        Args:
            xyz (jax.Array): Cartesian coordinates `xyz` with shape (n, 3),
                where `n` is the number of atoms, in Angstroms.
            with_grad (bool): Whether to calculate the gradients.
                Defaults to false.

        Returns:
            r (jax.Array): Distance vector with length k = (n * (n - 1) / 2),
                in Angstroms.
            J_r_xyz (jax.Array): Jacobian matrix of distance vector `r` with
                respect to Cartesian coordinates xyz with shape (k, n, 3), in
                Angstroms. Only be calculated when gradients are required,
                i.e. with_grad = True.
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

    def calc_m_from_r(
        self,
        r: jax.Array,
        with_grad: bool = False,
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        """Calculate Morse-like vector `m` from distance vector `r`.

        morse = exp(r / alpha)

        Args:
            r (jax.Array): Distance vector `r`.
            with_grad (bool): Whether to calculate the gradients.
                Defaults to false.

        Returns:
            m (jax.Array): Morse-like vector.
            J_m_r (jax.Array): Jacobian matrix of Morse like vector `m` with
                respect to distance vector `r`. Only be calculated when
                gradients are required, i.e. with_grad = True.
        """
        m = jnp.exp(-1.0 * r / self.alpha)

        if with_grad:
            J_m_r = jnp.diag(-1.0 * m / self.alpha)
            return m, J_m_r

        return m

    def calc_p_from_m(
        self,
        morse: jax.Array,
        with_grad: bool = False,
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        """Calculate Permutational Invariant Polynomials from
        Morse-like vector m.

        p = hat{P}(m)
        """
        pass

    def __call__(
        self,
        xyz: jax.Array,
        with_grad: bool = False
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        pass


if __name__ == "__main__":
    xyz = jnp.asarray([
        [-0.37080090, -0.13842560, -0.26234460],
        [+0.52405100, -0.92893340, -0.16449230],
        [-0.75608510, +0.09429450, +0.62230030],
        [+0.06355260, +0.73483450, -0.67571110],
        [-1.14601250, -0.54180340, -1.11155510],
    ])

    ab4_pip = JaxPIPDescriptor(
        basis=[[]],
        alpha=1.0,
    )

    r, J_r_xyz = ab4_pip.calc_r_from_xyz(
        xyz=xyz,
        with_grad=True,
    )

    print(f"{r = }")
    print(f"{J_r_xyz = }")

    m, J_m_r = ab4_pip.calc_m_from_r(
        r=r,
        with_grad=True,
    )

    print(f"{m = }")
    print(f"{J_m_r = }")
