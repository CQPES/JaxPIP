import json
from typing import List, Optional, Tuple, Union

import jax
from jax import numpy as jnp

from jaxpip.descriptor import PIPDescriptor
from jaxpip.descriptor._abc import AbstractDescriptor


class FragmentPIPDescriptor(AbstractDescriptor):
    """Fragment permutation invariant polynomial descriptor.

    Attributes:
        frag (List[List[int]]): Indices of atoms in each fragment.
        basis_set_list (List[List[jax.Array]]): List of permutation invariant
            basis for each fragment.
        alpha_list (List[float]): Range parameter of Morse-like variables
            for each fragment.
        descriptor_list (List[PIPDescriptor]): LIst of permutation invariant
            polynomial descriptors for each fragment.
    """

    def __init__(
        self,
        frag: List[List[int]],
        basis_set_list: List[List[jax.Array]],
        alpha_list: Optional[List[float]],
    ) -> None:
        self.frag = frag
        self.basis_set_list = basis_set_list
        self.alpha_list = alpha_list if alpha_list else [1.0] * len(frag)

        # set up fragment descriptors
        self.descriptor_list = []

        for (basis_set, alpha) in zip(basis_set_list, alpha_list):
            descriptor = PIPDescriptor.from_json(basis_set, alpha)
            self.descriptor_list.append(descriptor)

    @staticmethod
    def from_json_list(
        frag: List[List[int]],
        basis_json_list: List[List[jax.Array]],
        alpha_list: List[float],
    ) -> "FragmentPIPDescriptor":
        """Load PIP basis from a list of json files.

        Arguments:
            frag (List[List[int]]): Indices of atoms in each fragment.
            basis_set_list (List[List[jax.Array]]): A list of permutation
                invariant basis json files.
            alpha_list (List[float]): Range parameter of Morse-like variables
                for each fragment.
        """
        basis_set_list = []
        for basis_json in basis_json_list:
            with open(basis_json) as f:
                basis_set = json.load(f)

            basis_set = [jnp.array(basis, dtype=jnp.float32)
                         for basis in basis_set]
            basis_set_list.append(basis_set)

        return FragmentPIPDescriptor(frag, basis_set_list, alpha_list)

    def __call__(
        self,
        xyz: jax.Array,
        with_grad: bool = False
    ) -> Union[List[jax.Array], List[Tuple[jax.Array, jax.Array]]]:
        """Calculate fragment permutation invariant polynomial.

        Arguments:
            xyz (jax.Array): Cartesian coordinates `xyz` with shape (n, 3),
                where `n` is the number of atoms, in Angstroms.
            with_grad (bool): Whether to calculate the gradients.
                Defaults to false.

        Returns:
            p_list (List[jax.Array]): Permutation invariant polynomial for each
                fragment.
            J_p_xyz_list (List[jax.Array]): Jacobian matrix of permutation
                invariant polynomial `p` with respect to Cartesian coordinates
                `xyz` for each fragement. Only be calculated when gradients are
                required, i.e. with_grad = True.
        """

        p_list = []

        if not with_grad:
            for (frag, descriptor) in zip(self.frag, self.descriptor_list):
                p = descriptor(xyz[frag])
                p_list.append(p)

            return p_list

        # calculate Jacobian matrix J_p(xyz) for each fragment if required
        J_p_xyz_list = []

        for (frag, descriptor) in zip(self.frag, self.descriptor_list):
            p, J_p_xyz = descriptor(xyz[frag])
            p_list.append(p)
            J_p_xyz_list.append(J_p_xyz)

        return p_list, J_p_xyz_list
