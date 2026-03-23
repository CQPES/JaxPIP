# JaxPIP

Permutation Invariant Polynomials (PIPs) in JAX

Author: mizu-bai

Preprint: [Tensorization over Factorization: Rethinking Permutation Invariant Polynomials in JAX](https://doi.org/10.26434/chemrxiv.15000804/v1)

If JaxPIP helps your work, please cite correctly.

> Li, J.; Song, K.; Li, J. Tensorization over Factorization: Rethinking Permutation Invariant Polynomials in JAX. _ChemRxiv_, **2026**. https://doi.org/10.26434/chemrxiv.15000804/v1.

## Usage

### Command Line Interface (CLI)

Convert an MSA `.BAS` file to JaxPIP JSON format:

```bash
$ jaxpip bas2json /path/to/MSA.BAS [/path/to/jaxpip.json | /path/to/jaxpip.json.gz] [--gz]
```
Show basis information:

```bash
$ jaxpip show [/path/to/jaxpip.json | /path/to/jaxpip.json.gz]
```

### Python API

```python
import jax

# Remember to enable fp64
jax.config.update("jax_enable_x64", True)

import numpy as np

from jax import numpy as jnp

from jaxpip.descriptor import PolynomialDescriptor
from jaxpip.model import PolynomialLinearModel, PolynomialNeuralNetwork


# Initialize descriptor
descriptor = PolynomialDescriptor.from_file(
    "basis.json.gz",
    alpha=1.0,
    dtype=jnp.float64,
)

# Example 1. Build a linear model
coeffs = jnp.array(np.loadtxt("coeff.dat"))  # linear fitting coefficients
model = PolynomialLinearModel(
    descriptor=descriptor,
    coeffs=coeffs,
)

# Example 2: Build a Neural Network (PIP-NN)
# key = jax.random.PRNGKey(114514)
# model = PolynomialNeuralNetwork(
#     descriptor,
#     hidden_layers=[16, 32],
#     key=key,
#     activation="tanh",
# )
# ...

# Calculate energy and forces
xyz = ...  # (N_atoms, 3)

# energy shape: () scalar
energy = model.get_energy(xyz)

# energy shape: () scalar
# forces shape: (N_atoms, 3)
energy, forces = model.get_energy_and_forces(xyz)

# Batch evaluation
batch_xyz = ...  # shape: (N_batch, N_atoms, 3)

# batch_energy shape: (N_batch,)
batch_energy = jax.vmap(model.get_energy)(batch_xyz)

# batch_energy shape: (N_batch,)
# batch_forces shape: (N_batch, N_atoms, 3)
batch_energy, batch_forces = jax.vmap(model.get_energy_and_forces)(batch_xyz)
```

## License

BSD 2-Clause License

## References

- (1) Xie, Z.; Bowman, J. M. Permutationally Invariant Polynomial Basis for Molecular Energy Surface Fitting via Monomial Symmetrization. _J. Chem. Theory Comput._ **2010**, _6_ (1), 26–34. https://doi.org/10.1021/ct9004917.
- (2) Nandi, A.; Qu, C.; Bowman, J. M. Using Gradients in Permutationally Invariant Polynomial Potential Fitting: A Demonstration for CH4 Using as Few as 100 Configurations. _J. Chem. Theory Comput._ **2019**, _15_ (5), 2826–2835. https://doi.org/10.1021/acs.jctc.9b00043.
- (3) Jiang, B.; Guo, H. Permutation Invariant Polynomial Neural Network Approach to Fitting Potential Energy Surfaces. _J. Chem. Phys._ **2013**, _139_ (5). https://doi.org/10.1063/1.4817187.
- (4) Li, J.; Jiang, B.; Guo, H. Permutation Invariant Polynomial Neural Network Approach to Fitting Potential Energy Surfaces. II. Four-Atom Systems. _J. Chem. Phys._ **2013**, _139_ (20). https://doi.org/10.1063/1.4832697.
- (5) Li, J.; Song, K.; Li, J. Tensorization over Factorization: Rethinking Permutation Invariant Polynomials in JAX. _ChemRxiv_, **2026**. https://doi.org/10.26434/chemrxiv.15000804/v1.
