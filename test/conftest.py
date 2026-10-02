import os

os.environ.setdefault("JAX_ENABLE_X64", "True")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from jaxpip.descriptor import PolynomialDescriptor

# water A2B, degree 2: 3 atoms (H, H, O), 3 distances, 7 orbits / 10 monomials.
# Exponent position i refers to the triu distance order (H0-H1, H0-O, H1-O).
WATER_BASIS = [
    [[0, 0, 0]],
    [[0, 0, 1], [0, 1, 0]],
    [[1, 0, 0]],
    [[0, 1, 1]],
    [[1, 0, 1], [1, 1, 0]],
    [[0, 0, 2], [0, 2, 0]],
    [[2, 0, 0]],
]

WATER_XYZ = np.array([
    [0.0, 0.7572, 0.5866],
    [0.0, -0.7572, 0.5866],
    [0.0, 0.0, 0.0],
])  # H H O


def reference_pip(xyz, basis, alpha=1.0, decay_kernel="morse"):
    """Independent numpy oracle for the PIP vector (plain math, no jax)."""
    xyz = np.asarray(xyz)
    iu = np.triu_indices(len(xyz), k=1)
    r = np.linalg.norm(xyz[iu[0]] - xyz[iu[1]], axis=-1)
    out = []
    for orbit in basis:
        s = 0.0
        for m in orbit:
            if decay_kernel == "morse":
                s += np.exp(-np.dot(m, r) / alpha)
            else:  # reciprocal: (alpha / r)^m
                s += np.prod(np.power(alpha / r, m))
        out.append(s)
    return np.asarray(out)


@pytest.fixture
def water_desc():
    return PolynomialDescriptor(basis_set=WATER_BASIS, alpha=1.0,
                                decay_kernel="morse")


@pytest.fixture
def water_desc_recip():
    return PolynomialDescriptor(basis_set=WATER_BASIS, alpha=2.0,
                                decay_kernel="reciprocal")


@pytest.fixture
def water_batch():
    rng = np.random.default_rng(42)
    base = WATER_XYZ[None] + rng.normal(0, 0.1, (8, 3, 3))
    return jnp.asarray(base)
