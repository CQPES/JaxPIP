from abc import ABC, abstractmethod
from typing import Tuple, Union

import jax


class AbstractDescriptor(ABC):
    """Abstract class for polynomial based descriptors."""

    @abstractmethod
    def __call__(
        self,
        xyz: jax.Array,
        with_grad: bool = False,
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        pass
