import gzip
import json
import warnings
from typing import Any

import jax
from jax import numpy as jnp

from jaxpip.types import BasisInfo, InvariantBasis


class PolynomialDescriptor:
    def __init__(
        self,
        basis_set: InvariantBasis,
        alpha: float = 1.0,
        dtype: Any = jnp.float64,
    ) -> None:
        """
        Arguments:
            basis_set (InvariantBasis): Permutation invariant basis set.
            alpha (float): Range parameter for Morse-like transformation.
                Defaults to 1.0.
            dtype (Any): Floating point precision (jnp.float64 or jnp.float32).
                Defaults jnp.float64.
        """
        # 1. precision setup
        if dtype == jnp.float64 and not jax.config.read("jax_enable_x64"):
            warnings.warn(
                "PolynomialDescriptor is initialized with float64, "
                "but JAX x64 is not enabled. "
                "Please call `jax.config.update('jax_enable_x64', True)` "
                "at the beginning of your script.",
                UserWarning,
                stacklevel=2,
            )

        self.dtype = dtype

        # 2. pre-calculate constants
        self.alpha = alpha
        self.basis_matrix = jnp.array(
            [m for b in basis_set for m in b],
            dtype=dtype,
        )
        self.poly_seg_ids = jnp.array(
            [i for i, b in enumerate(basis_set) for _ in b],
            dtype=jnp.int32,
        )

        num_dist = self.basis_matrix.shape[1]
        num_atoms = int((1 + (1 + 8 * num_dist) ** 0.5) / 2)
        self._idx_i, self._idx_j = jnp.triu_indices(num_atoms, k=1)

        # 3. store attributes
        self.basis_info = BasisInfo(
            num_atoms=num_atoms,
            num_flat_mono=len(self.basis_matrix),
            num_poly=len(basis_set),
            max_degree=int(jnp.max(jnp.sum(self.basis_matrix, axis=1))),
        )

    def __call__(
        self,
        xyz: jax.Array,
    ) -> jax.Array:
        """Calculate permutation invariant polynomial (PIP).

        Arguments:
            xyz (jax.Array): Cartesian coordinates xyz with shape (n, 3),
                where n is the number of atoms, in Angstroms.

        Returns:
            p (jax.Array): Permutation invariant polynomial.
        """
        # 1. distance vector
        r = jnp.linalg.norm(
            xyz[self._idx_i] - xyz[self._idx_j],
            axis=-1,
        )

        # 2. log-exp fusion
        flat_monos = jnp.exp(jnp.dot(self.basis_matrix, -r / self.alpha))

        # 3. segment sum
        p = jax.ops.segment_sum(
            flat_monos,
            segment_ids=self.poly_seg_ids,
            num_segments=self.basis_info.num_poly,
            indices_are_sorted=True,
            unique_indices=False,
        )

        return p

    def __repr__(self):
        return f"PolynomialDescriptor(info={self.basis_info})"

    @property
    def feature_dim(self):
        return self.basis_info.num_poly

    @classmethod
    def from_file(
        cls,
        basis_file: str,
        alpha: float = 1.0,
        dtype: Any = jnp.float64,
    ) -> "PolynomialDescriptor":
        """Initilize PolynomialDescriptor from basis set json.

        Arguments:
            basis_file (str): Path to basis set file (json or json.gz).
            alpha (float): Range parameter for Morse-like transformation.
                Defaults to 1.0.
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
            raise ValueError(f"Invalid format of basis file: {basis_file}.")

        descriptor = cls(basis_set, alpha, dtype)

        return descriptor


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
    )

    print(pip_a2b_2)

    p = pip_a2b_2(water_xyz)
    J_p_xyz = jax.jacfwd(pip_a2b_2)(water_xyz)

    print(f"p(xyz) = {p}")
    print(f"Jp(xyz) = {J_p_xyz}")
