import os

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpip.model import PolynomialLinearModel, PolynomialNeuralNetwork

from conftest import WATER_BASIS, WATER_XYZ


@pytest.fixture
def lin_model(water_desc):
    rng = np.random.default_rng(3)
    coeffs = rng.normal(0, 0.5, water_desc.feature_dim)
    return PolynomialLinearModel(descriptor=water_desc, coeffs=jnp.asarray(coeffs))


@pytest.fixture
def nn_model(water_desc):
    model = PolynomialNeuralNetwork(
        descriptor=water_desc, hidden_layers=[4],
        key=jax.random.PRNGKey(0), activation="tanh")
    # give the scaler a non-degenerate range from a few geometries
    rng = np.random.default_rng(5)
    xyzs = jnp.asarray(WATER_XYZ[None] + rng.normal(0, 0.1, (16, 3, 3)))
    p_all = jax.vmap(water_desc)(xyzs)
    v_all = jnp.linspace(-1.0, 1.0, 16)
    return model.update_scaler(p_all, v_all)


# ------------------------------------------------------------------
# forces
# ------------------------------------------------------------------
def test_forces_match_finite_differences(lin_model):
    xyz = jnp.asarray(WATER_XYZ)
    _, f = lin_model.get_energy_and_forces(xyz)
    h = 1.0e-5
    fd = np.zeros_like(np.asarray(xyz))
    for a in range(3):
        for c in range(3):
            xp = xyz.at[a, c].add(h)
            xm = xyz.at[a, c].add(-h)
            fd[a, c] = float((lin_model.get_energy(xm) - lin_model.get_energy(xp))
                             / (2 * h))
    np.testing.assert_allclose(np.asarray(f), fd, atol=1e-8)


def test_forces_match_finite_differences_nn(nn_model):
    xyz = jnp.asarray(WATER_XYZ)
    _, f = nn_model.get_energy_and_forces(xyz)
    h = 1.0e-5
    fd = np.zeros_like(np.asarray(xyz))
    for a in range(3):
        for c in range(3):
            xp = xyz.at[a, c].add(h)
            xm = xyz.at[a, c].add(-h)
            fd[a, c] = float((nn_model.get_energy(xm) - nn_model.get_energy(xp))
                             / (2 * h))
    np.testing.assert_allclose(np.asarray(f), fd, atol=1e-7)


def test_net_force_is_zero(lin_model, nn_model):
    for model in (lin_model, nn_model):
        _, f = model.get_energy_and_forces(jnp.asarray(WATER_XYZ))
        np.testing.assert_allclose(np.asarray(f).sum(axis=0), 0.0, atol=1e-14)


# ------------------------------------------------------------------
# hessian
# ------------------------------------------------------------------
def test_hessian_symmetric(lin_model):
    h = np.asarray(lin_model.get_hessian(jnp.asarray(WATER_XYZ)))
    assert h.shape == (9, 9)
    np.testing.assert_allclose(h, h.T, atol=1e-12)


def test_hessian_matches_finite_difference_of_forces(lin_model):
    xyz = jnp.asarray(WATER_XYZ)
    h = np.asarray(lin_model.get_hessian(xyz))
    hfd = np.zeros_like(h)
    eps = 1.0e-5
    for i in range(9):
        a, c = divmod(i, 3)
        fp = np.asarray(lin_model.get_energy_and_forces(
            xyz.at[a, c].add(eps))[1]).reshape(-1)
        fm = np.asarray(lin_model.get_energy_and_forces(
            xyz.at[a, c].add(-eps))[1]).reshape(-1)
        hfd[:, i] = -(fp - fm) / (2 * eps)     # dF/dx = -d2E/dx2
    np.testing.assert_allclose(h, hfd, atol=1e-6)


# ------------------------------------------------------------------
# serialization roundtrip
# ------------------------------------------------------------------
def test_linear_save_load_roundtrip(lin_model, water_desc, tmp_path):
    import json

    basis_file = tmp_path / "basis.json"
    basis_file.write_text(json.dumps(WATER_BASIS))
    model_file = lin_model.save(str(tmp_path / "lin.eqx"))
    assert os.path.isabs(model_file)

    loaded = PolynomialLinearModel.from_file(
        basis_file=str(basis_file), model_file=model_file)
    rng = np.random.default_rng(11)
    xyzs = jnp.asarray(WATER_XYZ[None] + rng.normal(0, 0.1, (4, 3, 3)))
    e1 = jax.vmap(lin_model.get_energy)(xyzs)
    e2 = jax.vmap(loaded.get_energy)(xyzs)
    chex.assert_trees_all_close(e1, e2)


def test_nn_save_load_roundtrip(nn_model, tmp_path):
    import json

    basis_file = tmp_path / "basis.json"
    basis_file.write_text(json.dumps(WATER_BASIS))
    model_file = nn_model.save(str(tmp_path / "nn.eqx"))

    loaded = PolynomialNeuralNetwork.from_file(
        basis_file=str(basis_file), model_file=model_file)
    rng = np.random.default_rng(12)
    xyzs = jnp.asarray(WATER_XYZ[None] + rng.normal(0, 0.1, (4, 3, 3)))
    e1 = jax.vmap(nn_model.get_energy)(xyzs)
    e2 = jax.vmap(loaded.get_energy)(xyzs)
    chex.assert_trees_all_close(e1, e2)


def test_update_coeffs_changes_energy_linearly(lin_model):
    xyz = jnp.asarray(WATER_XYZ)
    p = lin_model.descriptor(xyz)
    new_coeffs = jnp.zeros_like(lin_model.coeffs)
    m2 = lin_model.update_coeffs(new_coeffs)
    np.testing.assert_allclose(float(m2.get_energy(xyz)), 0.0, atol=1e-15)
    # E(new) - E(old) = dc . p
    dc = jnp.asarray(np.random.default_rng(9).normal(0, 0.3, p.shape))
    m3 = lin_model.update_coeffs(lin_model.coeffs + dc)
    expected = float(lin_model.get_energy(xyz)) + float(jnp.dot(dc, p))
    np.testing.assert_allclose(float(m3.get_energy(xyz)), expected, atol=1e-12)


# ------------------------------------------------------------------
# batching
# ------------------------------------------------------------------
def test_batched_energy_matches_loop(lin_model, water_batch):
    batched = jax.vmap(lin_model.get_energy)(water_batch)
    for i in range(water_batch.shape[0]):
        chex.assert_trees_all_close(batched[i], lin_model.get_energy(water_batch[i]))


def test_batched_forces_match_loop(lin_model, water_batch):
    batched = jax.vmap(lin_model.get_energy_and_forces)(water_batch)
    for i in range(water_batch.shape[0]):
        _, f_single = lin_model.get_energy_and_forces(water_batch[i])
        chex.assert_trees_all_close(batched[1][i], f_single)

