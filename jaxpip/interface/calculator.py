from typing import List, Optional, Tuple, Union

import equinox as eqx
import jax
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from jax import numpy as jnp

from jaxpip.model import PolynomialLinearModel, PolynomialNeuralNetwork


class JaxPIPCalculator(Calculator):
    implemented_properties = [
        "energy",
        "forces",
        "hessian",
    ]

    def __init__(
        self,
        model: Union[PolynomialLinearModel, PolynomialNeuralNetwork],
        to_angstrom: float = 1.0,
        to_eV: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model

        @eqx.filter_jit
        def _energy_kernel(
            xyz: jax.Array,
        ) -> jax.Array:
            return model.get_energy(xyz / to_angstrom) * to_eV

        @eqx.filter_jit
        def _energy_forces_kernel(
            xyz: jax.Array,
        ) -> Tuple[jax.Array, jax.Array]:
            energy, forces = model.get_energy_and_forces(xyz / to_angstrom)

            energy = energy * to_eV
            forces = forces * (to_eV / to_angstrom)

            return energy, forces

        @eqx.filter_jit
        def _energy_forces_hessian_kernel(
            xyz: jax.Array,
        ) -> Tuple[jax.Array, jax.Array, jax.Array]:
            energy, forces = model.get_energy_and_forces(xyz / to_angstrom)
            hessian = model.get_hessian(xyz / to_angstrom)

            energy = energy * to_eV
            forces = forces * (to_eV / to_angstrom)
            hessian = hessian * (to_eV / to_angstrom**2)

            return energy, forces, hessian

        self._energy_kernel = _energy_kernel
        self._energy_forces_kernel = _energy_forces_kernel
        self._energy_forces_hessian_kernel = _energy_forces_hessian_kernel

    def get_hessian(
        self,
        atoms: Optional[Atoms] = None,
    ) -> np.ndarray:
        return self.get_property("hessian", atoms)

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: Optional[List[str]] = None,
        system_changes=all_changes,
    ) -> None:
        if properties is None:
            properties = self.implemented_properties

        super().calculate(
            atoms,
            properties,
            system_changes,
        )

        xyz = jnp.array(
            self.atoms.get_positions(),
            dtype=self.model.dtype,
        )

        if "hessian" in properties:
            energy, forces, hessian = self._energy_forces_hessian_kernel(xyz)

            self.results["energy"] = np.asarray(energy).item()
            self.results["forces"] = np.asarray(forces)
            self.results["hessian"] = np.asarray(hessian)

        elif "forces" in properties:
            energy, forces = self._energy_forces_kernel(xyz)

            self.results["energy"] = np.asarray(energy).item()
            self.results["forces"] = np.asarray(forces)

        elif "energy" in properties:
            energy = self._energy_kernel(xyz)

            self.results["energy"] = np.asarray(energy).item()
