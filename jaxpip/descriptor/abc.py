from abc import ABC, abstractmethod

import jax


class AbstractPolynomialDescriptor(ABC):
    """Abstract class for polynomial based descriptor."""

    @abstractmethod
    def __call__(
            self,
            xyz: jax.Array,
            with_grad: bool = False,
    ) -> jax.Array:
        pass
