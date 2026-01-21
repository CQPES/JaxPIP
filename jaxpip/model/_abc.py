from abc import ABC, abstractmethod
from typing import Tuple, Union

import jax


class AbstractModel(ABC):
    """Abstract class for potential models."""

    @abstractmethod
    def get_energy(
        self,
        xyz: jax.Array,
    ) -> jax.Array:
        pass

    @abstractmethod
    def get_energy_and_forces(
        self,
        xyz: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        pass

    @abstractmethod
    def __call__(
        self,
        xyz: jax.Array,
        with_force: bool = True,
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        pass
