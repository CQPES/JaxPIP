from typing import List, Tuple, Union

import jax
from jax import numpy as jnp


def _calc_r_from_xyz(
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


def _calc_m_from_r(
    r: jax.Array,
    alpha: float = 1.0,
    with_grad: bool = False,
) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
    """Calculate Morse-like vector `m` from distance vector `r`.

    morse = exp(r / alpha)

    Args:
        r (jax.Array): Distance vector `r`.
        alpha (float): Range parameter of Morse-like variables.
        with_grad (bool): Whether to calculate the gradients.
            Defaults to false.

    Returns:
        m (jax.Array): Morse-like vector.
        J_m_r (jax.Array): Jacobian matrix of Morse like vector `m` with
            respect to distance vector `r`. Only be calculated when
            gradients are required, i.e. with_grad = True.
    """
    m = jnp.exp(-1.0 * r / alpha)

    if with_grad:
        J_m_r = jnp.diag(-1.0 * m / alpha)
        return m, J_m_r

    return m


def _calc_p_from_m(
    m: jax.Array,
    basis_list: List[jax.Array],
    with_grad: bool = False,
) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
    """Calculate Permutational Invariant Polynomials from
    Morse-like vector m.

    p = hat{P}(m)

    Args:
        m (jax.Array): Morse vector `m`.
        basis_list (List[jax.Array]): Permutational invariant basis.
        with_grad (bool): Whether to calculate the gradients.
            Defaults to false.

    Returns:
        p (jax.Array): Permutation invariant polynomial.
        J_p_m (jax.Array): Jacobian matrix of permutation invariant
            polynomial `p` with respect to Morse-like vector `m`. Only be
            calculated when gradients are required, i.e. with_grad = True.
    """

    @jax.jit
    def _calc_p(
        m: jax.Array,
        basis: jax.Array,
    ) -> jax.Array:
        return (m**basis).prod(axis=1).sum(axis=0)

    p = jnp.zeros(shape=(len(basis_list)))

    if not with_grad:
        for (idx, basis) in enumerate(basis_list):
            p = p.at[idx].set(_calc_p(m, basis))

        return p

    # calculate Jacobian matrix J_p(m) if required
    _calc_p_dm = jax.grad(_calc_p)
    J_p_m = jnp.zeros(shape=(len(basis_list), len(m)))

    for (idx, basis) in enumerate(basis_list):
        p = p.at[idx].set(_calc_p(m, basis))
        J_p_m = J_p_m.at[idx].set(_calc_p_dm(m, basis))

    return p, J_p_m
