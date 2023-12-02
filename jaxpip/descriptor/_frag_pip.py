from typing import Tuple, Union

import jax

from jaxpip.descriptor._abc import AbstractDescriptor


class FragmentPIPDescriptor(AbstractDescriptor):
    """Fragment permutation invariant polynomial descriptor."""

    def __call__(
        self,
        xyz: jax.Array,
        with_grad: bool = False
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        raise (
            NotImplementedError,
            "Fragment PIP has not been implemented yet."
        )
