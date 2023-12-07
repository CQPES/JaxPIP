from typing import Tuple, Union

import jax

from jaxpip.descriptor._abc import AbstractDescriptor


class ManyBodyPIPDescriptor(AbstractDescriptor):
    """Many body permutation invariant polynomial descriptor."""

    def __call__(
        self,
        xyz: jax.Array,
        with_grad: bool = False
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        pass
