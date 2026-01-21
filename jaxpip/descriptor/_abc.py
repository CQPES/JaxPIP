from abc import ABC, abstractmethod
from typing import Tuple, Union

import jax


class AbstractDescriptor(ABC):
    """Abstract class for descriptors."""

    @abstractmethod
    def __call__(
        self,
        xyz: jax.Array,
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        pass
