import os

os.environ.setdefault("JAX_ENABLE_X64", "True")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from jaxpip.basis.msa import generate
from jaxpip.descriptor import PolynomialDescriptor

from conftest import WATER_BASIS, WATER_XYZ

# regression anchors validated against the MSA 2.0.1 binary
# (orbit count, monomial count)
ANCHORS = [
    ((2, 1), 3, 13, 20),
    ((2, 1), 5, 34, 56),
    ((4, 1), 4, 83, 1001),
    ((4, 1), 5, 208, 3003),
]


def test_water_d2_matches_handwritten_msa_basis():
    """The hand-written water A2B degree-2 basis (verified against a real
    MSA .BAS in test_basis.py) must be reproduced exactly."""
    assert generate((2, 1), 2) == WATER_BASIS


@pytest.mark.parametrize("counts,d,n_orbit,n_mono", ANCHORS)
def test_counts_match_msa_binary(counts, d, n_orbit, n_mono):
    basis = generate(counts, d)
    assert len(basis) == n_orbit
    assert sum(len(o) for o in basis) == n_mono


def test_every_orbit_is_closed_under_permutations():
    """Swapping the two equivalent atoms must map each orbit onto itself."""
    basis = generate((2, 1), 4)
    # A2B: swapping atoms 0,1 exchanges distances 1 and 2, fixes distance 0
    swap = lambda mono: [mono[0], mono[2], mono[1]]
    for orbit in basis:
        swapped = {tuple(swap(m)) for m in orbit}
        assert swapped == {tuple(m) for m in orbit}


def test_generated_basis_feeds_descriptor():
    desc = PolynomialDescriptor(basis_set=generate((2, 1), 3),
                                alpha=1.0, decay_kernel="morse")
    p = desc(jnp.asarray(WATER_XYZ))
    assert float(p[0]) == 1.0  # constant orbit
    assert bool(jnp.all(jnp.isfinite(p)))


def test_permutation_invariance_of_generated_descriptor():
    desc = PolynomialDescriptor(basis_set=generate((2, 1), 4),
                                alpha=1.0, decay_kernel="morse")
    rng = np.random.default_rng(7)
    for _ in range(5):
        xyz = WATER_XYZ + rng.normal(0, 0.1, WATER_XYZ.shape)
        p1 = np.asarray(desc(jnp.asarray(xyz)))
        p2 = np.asarray(desc(jnp.asarray(xyz[[1, 0, 2]])))
        np.testing.assert_allclose(p1, p2, atol=1e-12)
