from typing import Any, List, Tuple, Union, Callable

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxpip.descriptor import PolynomialDescriptor
from jaxpip.model.activation import ISRU


class FeatureScaler(eqx.Module):
    # (N_poly - 1,)
    p_min: jax.Array
    p_max: jax.Array
    V_min: jax.Array
    V_max: jax.Array
    esp: float = 1.0e-12  # fp64

    def __init__(
        self,
        p_min: jax.Array,
        p_max: jax.Array,
        V_min: jax.Array,
        V_max: jax.Array,
    ) -> None:
        self.p_min = p_min
        self.p_max = p_max
        self.V_min = V_min
        self.V_max = V_max

    def rescale_p(
        self,
        p: jax.Array,
    ) -> jax.Array:
        return 2.0 * (p - self.p_min) / (self.p_max - self.p_min + self.esp) - 1.0

    def unscale_V(
        self,
        V_scaled: jax.Array,
    ) -> jax.Array:
        return (
            0.5 * V_scaled * (self.V_max - self.V_min + self.esp)
            + (self.V_max + self.V_min) / 2.0
        )


class PolynomialNeuralNetwork(eqx.Module):
    """Polynomial Neural Network

    Attributes:
        descriptor (PolynomialDescriptor): Descriptor that transforms Cartesian
            coordinates into polynomials.
        layers (eqx.nn.Sequential): Neural network with polynomial as
            input.
        scaler: FeatureScaler
        dtype (jnp.dtype): Numerical precision inherited from descriptor.
    """

    descriptor: PolynomialDescriptor = eqx.field(static=True)
    layers: eqx.nn.Sequential
    scaler: FeatureScaler
    dtype: jnp.dtype = eqx.field(static=True)

    def __init__(
        self,
        descriptor: PolynomialDescriptor,
        hidden_layers: List[int],
        key: jax.Array,
        activation: Union[str, Callable] = "tanh",
    ) -> None:
        self.descriptor = descriptor

        # inherit descriptor's dtype
        _dtype = descriptor.dtype
        self.dtype = _dtype

        # drop the first element, always 1.0
        num_inputs = descriptor.feature_dim - 1

        self.scaler = FeatureScaler(
            p_min=jnp.zeros(num_inputs, dtype=_dtype),
            p_max=jnp.ones(num_inputs, dtype=_dtype),
            V_min=jnp.array(0.0),
            V_max=jnp.array(1.0),
        )

        # build network
        layer_sizes = [num_inputs] + hidden_layers + [1]
        keys = jax.random.split(key, len(layer_sizes) - 1)

        _activation_map = {
            "tanh": eqx.nn.Lambda(jax.nn.tanh),
            "isru": ISRU(),
        }

        activation_fn = (
            _activation_map[activation.lower()]
            if isinstance(activation, str)
            else activation
        )

        param_initializer = jax.nn.initializers.glorot_uniform()

        layers = []

        for i in range(len(layer_sizes) - 1):
            linear_layer = eqx.nn.Linear(
                in_features=layer_sizes[i],
                out_features=layer_sizes[i + 1],
                use_bias=True,
                dtype=_dtype,
                key=keys[i],
            )

            # Xavier

            # 1. split key
            w_key, _ = jax.random.split(keys[i])

            # 2. new weight and bias
            new_weight = param_initializer(
                w_key,
                shape=(layer_sizes[i + 1], layer_sizes[i]),
                dtype=_dtype,
            )

            new_bias = jnp.zeros(
                shape=(layer_sizes[i + 1],),
                dtype=_dtype,
            )

            # 3. update
            linear_layer = eqx.tree_at(
                where=lambda layer: (layer.weight, layer.bias),
                pytree=linear_layer,
                replace=(new_weight, new_bias),
            )

            layers.append(linear_layer)

            if i < len(layer_sizes) - 2:
                layers.append(activation_fn)

        self.layers = eqx.nn.Sequential(layers)

    def update_scaler(
        self,
        p_all: jax.Array,
        V_all: jax.Array,
    ) -> "PolynomialNeuralNetwork":
        if p_all.shape[1] == self.descriptor.feature_dim:
            p_use = p_all[:, 1:]
        else:
            p_use = p_all

        p_min = jnp.min(p_use, axis=0)
        p_max = jnp.max(p_use, axis=0)

        V_min = jnp.min(V_all)
        V_max = jnp.max(V_all)

        new_scaler = FeatureScaler(
            p_min=p_min,
            p_max=p_max,
            V_min=V_min,
            V_max=V_max,
        )

        new_model = eqx.tree_at(
            where=lambda m: m.scaler,
            pytree=self,
            replace=new_scaler,
        )

        return new_model

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
        X = self.scaler.rescale_p(p)

        # 3. p -> y in [-1.0, 1.0]
        y = self.layers(X)

        # 4. y -> V (in eV)
        V = self.scaler.unscale_V(y[0])

        return V

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

    model = model.update_scaler(p_example, V_example)

    print(model)

    energy, force = model.get_energy_and_forces(water_xyz)

    print(f"Energy (eV): {energy.item():.8f}")
    print("Force (eV/A): ")
    print(force)

    # batch
    batch_xyz = jnp.repeat(water_xyz[jnp.newaxis, :], 5, axis=0)
    batch_e, batch_f = jax.vmap(model.get_energy_and_forces)(batch_xyz)

    print(f"Batch Energy Shape: {batch_e.shape}")
    print(f"Batch Force Shape:  {batch_f.shape}")
