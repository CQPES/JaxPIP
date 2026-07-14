import warnings
from typing import Any, Callable, Dict, Literal

import jax
from jax import numpy as jnp

from jaxpip.basis import flatten_basis, get_basis_info, load_basis
from jaxpip.descriptor.kernel import kernel_morse, kernel_reciprocal
from jaxpip.types import InvariantBasis

KernelFn = Callable[[jax.Array], jax.Array]


class PolynomialDescriptor:
    def __init__(
        self,
        basis_set: InvariantBasis,
        alpha: float = 1.0,
        decay_kernel: Literal["morse", "reciprocal"] = "morse",
        dtype: Any = jnp.float64,
    ) -> None:
        """
        Arguments:
            basis_set (InvariantBasis): Permutation invariant basis set.
            alpha (float): Range parameter for the distance decay kernel.
                Defaults to 1.0.
            decay_kernel (Literal["morse", "reciprocal"]): Choice of kernel
                for internuclear distance decay.
                Defaults to "morse".
            dtype (Any): Floating point precision (jnp.float64 or jnp.float32).
                Defaults to jnp.float64.
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

        # alpha range parameter
        self.alpha = alpha

        # decay kernel, morse or reciprocal
        self.decay_kernel = decay_kernel

        # pip/fi basis
        self.basis_info = get_basis_info(basis_set)

        exponents, segments = flatten_basis(basis_set)
        self.basis_matrix = jnp.array(exponents, dtype=jnp.uint8)  # 8-bit quant
        self.poly_seg_ids = jnp.array(segments, dtype=jnp.int32)

        # for segment_sum optimization
        self._poly_seg_ids_are_sorted = bool(
            jnp.all(jnp.diff(self.poly_seg_ids) >= 0),
        )

        self._idx_i, self._idx_j = jnp.triu_indices(
            self.basis_info.num_atoms,
            k=1,
        )

        _kernel_map: Dict[str, KernelFn] = {
            "morse": lambda r: kernel_morse(
                basis_matrix=self.basis_matrix.astype(self.dtype),  # cast
                r=r,
                alpha=alpha,
            ),
            "reciprocal": lambda r: kernel_reciprocal(
                basis_matrix=self.basis_matrix.astype(self.dtype),  # cast
                r=r,
                ln_alpha=jnp.log(alpha),
            ),
        }

        self._kernel = _kernel_map[self.decay_kernel]

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

        # 2. log-exp fusion kernel
        flat_monos = self._kernel(r)

        # 3. segment sum
        p = jax.ops.segment_sum(
            flat_monos,
            segment_ids=self.poly_seg_ids,
            num_segments=self.basis_info.num_poly,
            indices_are_sorted=self._poly_seg_ids_are_sorted,
            unique_indices=False,
        )

        return p

    def __repr__(self):
        return (
            f"PolynomialDescriptor(\n"
            f"    info={self.basis_info},\n"
            f"    alpha={self.alpha},\n"
            f"    decay_kernel={self.decay_kernel},\n"
            f")"
        )

    @property
    def feature_dim(self):
        return self.basis_info.num_poly

    @classmethod
    def from_file(
        cls,
        basis_file: str,
        decay_kernel: Literal["morse", "reciprocal"] = "morse",
        alpha: float = 1.0,
        dtype: Any = jnp.float64,
    ) -> "PolynomialDescriptor":
        """Initilize PolynomialDescriptor from basis set json.

        Arguments:
            basis_file (str): Path to basis set file (json or json.gz).
            alpha (float): Range parameter for Morse-like transformation.
                Defaults to 1.0.
            decay_kernel (Literal["morse", "reciprocal"]): Internuclear
                distances decay kernel. Defaults to "morse".
            dtype (Any): Precision, jnp.float64 or jnp.float32.
                Defaults jnp.float64.

        Returns:
            descriptor (PolynomialDescriptor): PIP descriptor.
        """
        try:
            basis_set = load_basis(basis_file)
        except Exception as e:
            raise RuntimeError(f"Failed to load basis file: {e}")

        descriptor = cls(
            basis_set=basis_set,
            alpha=alpha,
            decay_kernel=decay_kernel,
            dtype=dtype,
        )

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
