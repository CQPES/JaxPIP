from typing import NamedTuple, Optional, Tuple, Union

import equinox as eqx
import jax
from jax import numpy as jnp

from jaxpip.model import AbstractModel


class TrainMetrics(NamedTuple):
    """Container for training metrics.

    Attributes:
        total_loss (jax.Array): The weighted combination of losses on energy
            and forces.
        V_mae (jax.Array): Mean Absolute Error of potential energy.
        V_rmse (jax.Array): Root Mean Squared Error of potential energy.
        f_mae (jax.Array): Mean Absolute Error of forces.
        f_rmse (jax.Array): Root Mean Squared Error of forces.
        num_V (jax.Array): Number of energy samples with non-zero weights.
        num_f (jax.Array): Number of force samples with non-zero weights.
    """
    total_loss: jax.Array
    V_mae: jax.Array
    V_rmse: jax.Array
    f_mae: jax.Array
    f_rmse: jax.Array
    num_V: jax.Array
    num_f: jax.Array


@eqx.filter_jit
def compute_residuals(
    model: AbstractModel,
    xyz_batch: jax.Array,
    V_target: jax.Array,
    f_target: Optional[jax.Array] = None,
) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
    """Compute residues for a batch of configurations.

    Arguments:
        model (AbstractModel): Potential energy model.
        xyz_batch (jax.Array): Cartesian coordinates with shape
            (N_batch, N_atom, 3).
        V_target (jax.Array): Target energies with shape (N_batch,).
        f_target (Optional[jax.Array]): Target forces with shape
            (N_batch, N_atom, 3).

    Returns:
        res_V (jax.Array): Energy residues with shape (N_batch,).
        res_f (Optional[jax.Array]): Forces residues with shape
            (N_batch, N_atom, 3). Only when f_targets presented.
    """
    if f_target is not None:
        # energy + forces mode
        V_predict, f_predict = jax.vmap(model.get_energy_and_forces)(xyz_batch)

        res_V = V_predict - V_target
        res_f = f_predict - f_target

        return res_V, res_f

    V_predict = jax.vmap(model.get_energy)(xyz_batch)

    res_V = V_predict - V_target

    return res_V


@eqx.filter_jit
def compute_loss_and_metrics(
    model: AbstractModel,
    xyz_batch: jax.Array,
    V_target: jax.Array,
    f_target: Optional[jax.Array] = None,
    V_weights: Optional[jax.Array] = None,
    f_weights: Optional[jax.Array] = None,
    f_coeff: jax.Array = 0.1,
) -> Tuple[jax.Array, TrainMetrics]:
    """Computes the loss and detailed metrics for a batch of configurations.

    Arguments:
        model (AbstractModel): Potential energy model.
        xyz_batch (jax.Array): Cartesian coordinates with shape
            (N_batch, N_atom, 3).
        V_target (jax.Array): Target energies with shape (N_batch,).
        f_target (Optional[jax.Array]): Target forces with shape
            (N_batch, N_atom, 3).
        V_weights (Optional[jax.Array]): Weights for energy samples with shape
            (N_batch,).
        f_weights (Optional[jax.Array]): Weights for forces samples with shape
            (N_batch,).
        f_coeff (Optional[jax.Array]): Weighting coefficient for force loss.

    Returns:
l       loss (jax.Array): Scalar loss value.
        metrics (TrainMetrics): A TrainMetrics object containing detailed error
            statistics.
    """
    _dtype = xyz_batch.dtype

    def get_weighted_mean(values, weights):
        return jnp.sum(values * weights) / (jnp.sum(weights) + 1.0e-08)

    def masked_mae(diff, weights):
        if diff.ndim == 3:
            # force
            val = jnp.mean(jnp.abs(diff), axis=(1, 2))
        else:
            # energy
            val = jnp.abs(diff)

        mae = jnp.sum(val * (weights > 0.0)) / \
            (jnp.count_nonzero(weights) + 1.0e-08)

        return mae

    def masked_rmse(sq_diff, weights):
        rmse = jnp.sqrt(jnp.sum(sq_diff * (weights > 0)) /
                        (jnp.count_nonzero(weights) + 1.0e-08))

        return rmse

    # handle weights
    if V_weights is None:
        V_weights = jnp.ones_like(V_target, dtype=_dtype)

    # energy only
    if f_target is None:
        V_pred = jax.vmap(model.get_energy)(xyz_batch)
        diff_V = V_pred - V_target
        sq_diff_V = jnp.square(diff_V)
        # weighted_loss_V = jnp.sum(V_weights * sq_diff_V)

        loss = get_weighted_mean(sq_diff_V, V_weights)

        metrics = TrainMetrics(
            total_loss=loss,
            V_mae=masked_mae(diff_V, V_weights),
            V_rmse=masked_rmse(sq_diff_V, V_weights),
            f_mae=jnp.array(0.0, dtype=_dtype),
            f_rmse=jnp.array(0.0, dtype=_dtype),
            num_V=jnp.count_nonzero(V_weights),
            num_f=jnp.array(0, dtype=jnp.int32),
        )

        return loss, metrics

    # handle weights
    if f_weights is None:
        f_weights = jnp.ones(shape=(f_target.shape[0],), dtype=_dtype)

    # energy & forces
    V_pred, f_pred = jax.vmap(model.get_energy_and_forces)(xyz_batch)

    diff_V = V_pred - V_target
    diff_f = f_pred - f_target

    # loss on V
    sq_diff_V = jnp.square(diff_V)
    # weighted_loss_V = jnp.sum(V_weights * sq_diff_V)

    # loss on forces
    sq_diff_f = jnp.mean(jnp.square(diff_f), axis=(1, 2))
    # weighted_loss_f = jnp.sum(f_weights * sq_diff_f)

    # total loss
    loss = get_weighted_mean(sq_diff_V, V_weights) + \
        f_coeff * get_weighted_mean(sq_diff_f, f_weights)

    metrics = TrainMetrics(
        total_loss=loss,
        V_mae=masked_mae(diff_V, V_weights),
        V_rmse=masked_rmse(sq_diff_V, V_weights),
        f_mae=masked_mae(diff_f, f_weights),
        f_rmse=masked_rmse(sq_diff_f, f_weights),
        num_V=jnp.count_nonzero(V_weights),
        num_f=jnp.count_nonzero(f_weights),
    )

    return loss, metrics
