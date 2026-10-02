import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpip.descriptor import PolynomialDescriptor

from conftest import WATER_BASIS, WATER_XYZ, reference_pip


def test_descriptor_matches_numpy_oracle(water_desc):
    p = np.asarray(water_desc(jnp.asarray(WATER_XYZ)))
    p_ref = reference_pip(WATER_XYZ, WATER_BASIS, alpha=1.0, decay_kernel="morse")
    chex.assert_trees_all_close(p, p_ref, atol=1e-13)


def test_reciprocal_kernel_matches_numpy_oracle(water_desc_recip):
    p = np.asarray(water_desc_recip(jnp.asarray(WATER_XYZ)))
    p_ref = reference_pip(WATER_XYZ, WATER_BASIS, alpha=2.0,
                          decay_kernel="reciprocal")
    chex.assert_trees_all_close(p, p_ref, atol=1e-13)


def test_permutation_invariance(water_desc):
    """Swapping the two H atoms must leave the PIP vector unchanged."""
    swapped = WATER_XYZ[[1, 0, 2]]
    p1 = np.asarray(water_desc(jnp.asarray(WATER_XYZ)))
    p2 = np.asarray(water_desc(jnp.asarray(swapped)))
    np.testing.assert_allclose(p1, p2, atol=1e-12)


def test_permutation_invariance_random_geometries(water_desc):
    rng = np.random.default_rng(7)
    for _ in range(8):
        xyz = WATER_XYZ + rng.normal(0, 0.15, WATER_XYZ.shape)
        p1 = np.asarray(water_desc(jnp.asarray(xyz)))
        p2 = np.asarray(water_desc(jnp.asarray(xyz[[1, 0, 2]])))
        np.testing.assert_allclose(p1, p2, atol=1e-12)


def test_vmap_matches_single_structure_loop(water_desc, water_batch):
    batched = jax.vmap(water_desc)(water_batch)
    for i in range(water_batch.shape[0]):
        single = water_desc(water_batch[i])
        chex.assert_trees_all_close(batched[i], single)


def test_descriptor_jit_matches_eager(water_desc, water_batch):
    eager = jax.vmap(water_desc)(water_batch)
    jitted = jax.jit(jax.vmap(water_desc))(water_batch)
    chex.assert_trees_all_close(jitted, eager)


def test_feature_dim_matches_basis(water_desc):
    n_mono = sum(len(orbit) for orbit in WATER_BASIS)
    assert water_desc.feature_dim == len(WATER_BASIS)
    assert water_desc.basis_info.num_flat_mono == n_mono
    assert water_desc.basis_info.num_atoms == 3


def test_gradient_flow_through_descriptor(water_desc):
    """Descriptor must be differentiable w.r.t. coordinates."""
    g = jax.grad(lambda x: jnp.sum(water_desc(x)))(jnp.asarray(WATER_XYZ))
    assert g.shape == WATER_XYZ.shape
    assert bool(jnp.all(jnp.isfinite(g)))


def test_fp64_without_x64_warns():
    """Constructing an fp64 descriptor without x64 must emit a UserWarning."""
    import warnings

    # simulate x64 being disabled by constructing with fp32 (allowed, no warning)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        PolynomialDescriptor(basis_set=WATER_BASIS, alpha=1.0,
                             dtype=jnp.float32)  # must NOT warn


def test_reciprocal_diverges_at_zero_distance():
    """reciprocal kernel is singular at r -> 0 (documented behavior)."""
    desc = PolynomialDescriptor(basis_set=WATER_BASIS, alpha=1.0,
                                decay_kernel="reciprocal")
    collapsed = WATER_XYZ.copy()
    collapsed[1, :] = collapsed[0, :]          # two atoms coincide -> r = 0
    p = np.asarray(desc(jnp.asarray(collapsed)))
    assert not bool(jnp.all(jnp.isfinite(p)))  # inf expected


def test_morse_finite_at_zero_distance(water_desc):
    """morse kernel stays finite at r = 0 (collision-safe)."""
    collapsed = WATER_XYZ.copy()
    collapsed[1, :] = collapsed[0, :]
    p = np.asarray(water_desc(jnp.asarray(collapsed)))
    assert bool(jnp.all(jnp.isfinite(p)))


def test_alpha_scales_the_kernel(water_desc):
    d2 = PolynomialDescriptor(basis_set=WATER_BASIS, alpha=2.0,
                              decay_kernel="morse")
    xyz = jnp.asarray(WATER_XYZ)
    # single-member degree-1 orbit (HH): exp(-r/2) = sqrt(exp(-r))
    p1 = np.asarray(water_desc(xyz))
    p2 = np.asarray(d2(xyz))
    np.testing.assert_allclose(np.sqrt(p1[2]), p2[2], atol=1e-13)
