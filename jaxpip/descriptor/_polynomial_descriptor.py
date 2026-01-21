import gzip
import json
from typing import Any, List, Tuple, Union

import jax
from jax import numpy as jnp

from jaxpip.descriptor import AbstractDescriptor


class PolynomialDescriptor(AbstractDescriptor):
    def __init__(
        self,
        basis_set: List[List[List[int]]],
        alpha: float = 1.0,
        with_grad: bool = False,
        dtype: Any = jnp.float64,
    ) -> None:
        """
        Arguments:
            basis_set (List[List[List[int]]]): Permutation invariant basis set.
            alpha (float): Range parameter for Morse-like transformation.
                Defaults to 1.0.
            with_grad (bool): Whether to calculate gradients.
                Defaults to False.
            dtype (Any): Floating point precision (jnp.float64 or jnp.float32).
                Defaults jnp.float64.
        """
        # 1. precision setup
        if dtype == jnp.float64 and not jax.config.read("jax_enable_x64"):
            jax.config.update("jax_enable_x64", True)

        # 2. pre-calculate constants
        _exps = jnp.array([m for b in basis_set for m in b], dtype=dtype)
        _segs = jnp.array([i for i, b in enumerate(basis_set)
                          for _ in b], dtype=jnp.int32)
        _n_pips = len(basis_set)
        _alpha = alpha

        num_dist = _exps.shape[1]
        num_atoms = int((1 + (1 + 8 * num_dist)**0.5) / 2)
        _idx_i, _idx_j = jnp.triu_indices(num_atoms, k=1)

        # 3. store attributes
        self.flat_exponents = _exps
        self.num_flat_exponents = len(_exps)
        self.num_atoms = num_atoms
        self.num_pips = _n_pips
        self.dtype = dtype
        self.with_grad = with_grad

        # 4. forward function: xyz -> p
        def forward_fn(xyz):
            r = jnp.linalg.norm(xyz[_idx_i] - xyz[_idx_j], axis=-1)
            monomial_values = jnp.exp(jnp.dot(_exps, -r / _alpha))
            p = jax.ops.segment_sum(monomial_values, _segs, _n_pips)

            return p

        # 4. build kernel
        def val_and_jac(xyz):
            # prefer jacfwd since output dim always >> input dim
            return forward_fn(xyz), jax.jacfwd(forward_fn)(xyz)

        self._kernel = jax.jit(val_and_jac if with_grad else forward_fn)

        # 5. batch kernel
        self._batch_kernel = jax.jit(jax.vmap(self._kernel))

    def __call__(
        self,
        xyz: jax.Array,
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        """Calculate permutation invariant polynomial (PIP).

        Arguments:
            xyz (jax.Array): Cartesian coordinates xyz with shape (n, 3),
                where n is the number of atoms, in Angstroms.

        Returns:
            p (jax.Array): Permutation invariant polynomial.
            J_p_xyz (jax.Array): (Optional) Jacobian matrix dp/dxyz.
                Only be calculated when gradients are required,
                i.e. with_grad = True.
        """
        return self._batch_kernel(xyz) if xyz.ndim == 3 else self._kernel(xyz)

    @classmethod
    def from_file(
        cls,
        basis_file: str,
        alpha: float = 1.0,
        with_grad: bool = False,
        dtype: Any = jnp.float64,
    ) -> "PolynomialDescriptor":
        """Initilize PolynomialDescriptor from basis set json.

        Arguments:
            basis_file (str): Path to basis set file (json or json.gz).
            alpha (float): Range parameter for Morse-like transformation.
                Defaults to 1.0.
            with_grad (bool): Whether to calculate gradients.
                Defaults to False.
            dtype (Any): Precision, jnp.float64 or jnp.float32.
                Defaults jnp.float64.

        Returns:
            descriptor (PolynomialDescriptor): PIP descriptor.
        """
        if basis_file.endswith(".json"):
            with open(basis_file, "r") as f:
                basis_set = json.load(f)
        elif basis_file.endswith(".json.gz"):
            with gzip.open(basis_file, "rt") as f:
                basis_set = json.load(f)
        else:
            raise RuntimeError(
                f"Invalid format of basis file: {basis_file}."
            )

        descriptor = cls(basis_set, alpha, with_grad, dtype)

        return descriptor

    def __repr__(self):
        return (
            "PolynomialDescriptor(\n"
            f"  num_flat_exponents={self.num_flat_exponents},\n"
            f"  num_atoms={self.num_atoms},\n"
            f"  num_pips={self.num_pips},\n"
            f"  dtype={self.dtype},\n"
            f"  with_grad={self.with_grad}\n"
            ")"
        )


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
        with_grad=True,
    )

    p, J_p_xyz = pip_a2b_2(water_xyz)

    print(f"p(xyz) = {p}")
    print(f"Jp(xyz) = {J_p_xyz}")
