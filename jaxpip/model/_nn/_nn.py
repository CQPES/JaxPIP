from typing import Any, List, Tuple, Union

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxpip.descriptor import PolynomialDescriptor
from jaxpip.model import AbstractModel
from jaxpip.model._nn import ISRULayer, TanhLayer

_ACT_MAP = {
    "tanh": TanhLayer,
    "isru": ISRULayer,
}


class PolynomialNeuralNetwork(eqx.Module, AbstractModel):
    """Polynomial Neural Network

    Attributes:
        descriptor (PolynomialDescriptor): Descriptor that transforms Cartesian
            coordinates into polynomials.
        layers (Tuple[eqx.Module, ...]): Neural network with polynomial as
            input.
        dtype (jnp.dtype): Numerical precision inherited from descriptor.
        p_min (jax.Array): Minimum values of polynomials for neural network
            input normalization.
        p_max (jax.Array): Maximum values of polynomials for neural network
            input normalization.
        V_min (jax.Array): Minimum energy value for output rescaling.
        V_max (jax.Array): Maximum energy value for ouptut rescaling.
    """
    descriptor: PolynomialDescriptor = eqx.field(static=True)
    layers: Tuple[eqx.Module, ...]
    dtype: jnp.dtype = eqx.field(static=True)

    # scale factors
    p_min: jax.Array
    p_max: jax.Array
    V_min: jax.Array
    V_max: jax.Array

    def __init__(
        self,
        descriptor: PolynomialDescriptor,
        hidden_layers: List[int],
        key: Any,
        activation: Union[str, eqx.Module] = "tanh",
    ) -> None:
        self.descriptor = descriptor

        # inherit descriptor's dtype
        _dtype = descriptor.dtype

        # drop the first element, always 1.0
        num_inputs = descriptor.num_pips - 1

        # placeholder
        self.p_min = jnp.zeros(num_inputs, dtype=_dtype)
        self.p_max = jnp.ones(num_inputs, dtype=_dtype)
        self.V_min = jnp.array(0.0, dtype=_dtype)
        self.V_max = jnp.array(1.0, dtype=_dtype)

        # nn
        layer_sizes = [num_inputs] + hidden_layers + [1]

        layers = []
        keys = jax.random.split(key, len(layer_sizes) - 1)

        for i in range(len(layer_sizes) - 1):
            layers.append(
                eqx.nn.Linear(
                    in_features=layer_sizes[i],
                    out_features=layer_sizes[i + 1],
                    dtype=_dtype,
                    key=keys[i],
                )
            )

            if i < len(layer_sizes) - 2:
                # activation
                if isinstance(activation, eqx.Module):
                    layers.append(activation)
                else:
                    _act_cls = _ACT_MAP[activation.lower()]
                    act_layer = _act_cls(dtype=_dtype)
                    layers.append(act_layer)

        self.layers = tuple(layers)

        # store dtype
        self.dtype = _dtype

    def _update_scaling_factor(
        self,
        p_use: jax.Array,
        V_all: jax.Array,
    ) -> "PolynomialNeuralNetwork":
        p_min = jnp.min(p_use, axis=0)
        p_max = jnp.max(p_use, axis=0)

        V_min = jnp.min(V_all)
        V_max = jnp.max(V_all)

        new_model = eqx.tree_at(
            where=(lambda m: (m.p_min, m.p_max, m.V_min, m.V_max)),
            pytree=self,
            replace=(p_min, p_max, V_min, V_max),
        )

        return new_model

    def make_scale_xyz_V(
        self,
        xyz_all: jax.Array,
        V_all: jax.Array,
    ) -> "PolynomialNeuralNetwork":
        """Computes and updates scaling factors of polynomials and energy.

        Arguments:
            xyz_all (jax.Array): Reference Cartesian coordinates with shape
                (N_samples, N_atom, 3).
            V_all (jax.Array): Reference potential energies with shape
                (N_samples,).

        Returns:
            new_model (PolynomialNeuralNetwork): A new model instance with
                updated p_min, p_max, V_min, and V_max.
        """
        # calculate p of all input xyz
        p_all = self.descriptor(xyz_all)

        # drop first element
        p_use = p_all[:, 1:]

        new_model = self._update_scaling_factor(p_use, V_all)

        return new_model

    def make_scale_p_V(
        self,
        p_all: jax.Array,
        V_all: jax.Array,
    ) -> "PolynomialNeuralNetwork":
        """Updates scaling factors of polynomials and energy.

        Arguments:
            p_all (jax.Array): Reference polynomials with shape (N_p,), first
                element 1.0 kept.
            V_all (jax.Array): Reference potential energies with shape
                (N_samples,).

        Returns:
            new_model (PolynomialNeuralNetwork): A new model instance with
                updated p_min, p_max, V_min, and V_max.
        """
        return self._update_scaling_factor(p_all[:, 1:], V_all)

    @eqx.filter_jit
    def get_energy(
        self,
        xyz: jax.Array,
    ) -> jax.Array:
        """Get potential energy of given xyz coordinates.

        Arguments:
            xyz (jax.Array): Cartesian coordinates with shape (N_atom, 3).

        Returns:
            V (jax.Array): Potential energy.
        """
        # 1. xyz (in Angstrom) -> p
        p_full = self.descriptor(xyz)
        # drop first element, always 1.0
        p = p_full[1:]

        # 2. p -> X in [-1.0, 1.0]
        eps = jnp.array(1.0e-08, dtype=self.dtype)  # for numerical stability
        X = 2.0 * (p - self.p_min) / (self.p_max - self.p_min + eps) - 1.0

        # 3. predict y in [-1.0, 1.0]
        x = X

        for layer in self.layers:
            x = layer(x)

        y = x[0]

        # 4. y -> V (in eV)
        V = 0.5 * (y + 1.0) * (self.V_max - self.V_min) + self.V_min

        return V

    @eqx.filter_jit
    def get_energy_and_forces(
        self,
        xyz: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """Get potential energy and forces of given xyz coordinates.

        Arguments:
            xyz (jax.Array): Cartesian coordinates with shape (N_atom, 3).

        Returns:
            V (jax.Array): Potential energy.
            f (jax.Array): Cartesian forces with shape (N_atom, 3).
        """
        V, g = jax.value_and_grad(self.get_energy)(xyz)

        # handle (possible) net force problems
        f = -(g - jnp.mean(g, axis=0))

        return V, f

    def __call__(
        self,
        xyz: jax.Array,
        with_force: bool = False,
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        if xyz.ndim == 3:
            if with_force:
                return jax.vmap(self.get_energy_and_forces)(xyz)

            return jax.vmap(self.get_energy)(xyz)

        if with_force:
            return self.get_energy_and_forces(xyz)

        return self.get_energy(xyz)


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
        dtype=jnp.float64,
    )

    key = jax.random.PRNGKey(114514)

    model = PolynomialNeuralNetwork(
        descriptor=pip_a2b_2,
        hidden_layers=[16, 32],
        key=key,
        # activation="isru",
    )

    print(model)

    # fake data
    p_example = pip_a2b_2(water_xyz)[jnp.newaxis, :]  # (1, num_pips)
    V_example = jnp.array([0.0])

    model = model.make_scale_p_V(p_example, V_example)

    print(model)

    energy, force = model.get_energy_and_forces(water_xyz)

    print(f"Energy (eV): {energy.item():.8f}")
    print("Force (eV/A): ")
    print(force)

    # batch
    batch_xyz = jnp.repeat(water_xyz[jnp.newaxis, :], 5, axis=0)
    batch_e, batch_f = model(batch_xyz, with_force=True)

    print(f"Batch Energy Shape: {batch_e.shape}")
    print(f"Batch Force Shape:  {batch_f.shape}")
