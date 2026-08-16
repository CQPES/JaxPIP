# JaxPIP

[![License](https://img.shields.io/pypi/l/cqpes)](https://github.com/cqpes/jaxpip/blob/main/LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/CQPES/JaxPIP)
[![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=flat&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/CQPES/JaxPIP)

Permutation Invariant Polynomials (PIPs) in JAX

Author: mizu-bai

Preprint: [Tensorization over Factorization: Rethinking Permutation Invariant Polynomials in JAX](https://doi.org/10.26434/chemrxiv.15000804/v1)

If JaxPIP helps your work, please cite correctly.

> Li, J.; Song, K.; Li, J. Tensorization over Factorization: Rethinking Permutation Invariant Polynomials in JAX. _ChemRxiv_, **2026**. https://doi.org/10.26434/chemrxiv.15000804/v1.

## Usage

### Available Basis Sets

Refer to the [`JaxPIP-Basis-Library`](https://github.com/CQPES/JaxPIP-Basis-Library) for ready-to-use basis sets (including converted MSA PIPs and collected FIs).

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

### Model Serialization

Both linear and neural-network models can be saved after fitting. `save()`
returns the absolute checkpoint path. A model checkpoint stores its
hyperparameters in the first line, followed by the serialized Equinox leaves.
The basis set is referenced by the loading path and is not embedded in the model
file, so keep the basis JSON and model checkpoint together.

```python
# Linear model
model.save("model.eqx")

model = PolynomialLinearModel.from_file(
    basis_file="basis.json.gz",
    model_file="model.eqx",
)

# Neural network
network.save("network.eqx")

network = PolynomialNeuralNetwork.from_file(
    basis_file="basis.json.gz",
    model_file="network.eqx",
)
```

Checkpoint loading reconstructs the descriptor from the saved `alpha`,
`decay_kernel`, and `dtype` values, then restores the fitted parameters.

### ASE Interface

`JaxPIPCalculator` provides energy, forces, and Hessian calculations for ASE
`Atoms` objects.

```python
from ase import Atoms

from jaxpip.interface import JaxPIPCalculator


atoms = Atoms(...)
atoms.calc = JaxPIPCalculator(model)

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
hessian = atoms.calc.get_hessian()
```

By default, the model is assumed to use Angstrom and eV. For another unit
system, set `to_angstrom` to the model length unit in Angstrom and `to_eV` to
the model energy unit in eV.

### Export to ONNX

JaxPIP models can be exported to ONNX for use in C++, Fortran, or other production environments.

```python
import jax

jax.config.update("jax_enable_x64", True)

from jax import numpy as jnp

from jaxpip.descriptor import PolynomialDescriptor
from jaxpip.model import PolynomialLinearModel, PolynomialNeuralNetwork


model = ...


# define whatever you want to export
@eqx.filter_jit
def export_fn_energy(xyz):
    return model.get_energy(xyz)


@eqx.filter_jit
def export_fn_energy_and_forces(xyz):
    return model.get_energy_and_forces(xyz)


@eqx.filter_jit
def export_fn_batch_energy(xyz):
    return jax.vmap(model.get_energy)(xyz)


@eqx.filter_jit
def export_fn_batch_energy_and_forces(xyz):
    return jax.vmap(model.get_energy_and_forces)(xyz)


import jax2onnx  # NOTE: must import jax2onnx here or jax ops will be mocked
import onnx


# single
onnx_model = jax2onnx.to_onnx(
    fn=export_fn_energy,  # or export_fn_energy_and_forces
    inputs=[
        (N_atoms, 3),
    ],
    enable_double_precision=True,
)

onnx.save(onnx_model, "model.onnx")

# batch
onnx_model_batch = jax2onnx.to_onnx(
    fn=export_fn_batch_energy,  # or export_fn_batch_energy_and_forces
    inputs=[
        ("B", N_atoms, 3),  # "B" for dynamic batch
    ],
    enable_double_precision=True,
)

onnx.save(onnx_model_batch, "model_batch.onnx")
```

After exporting, it is highly recommended to use [ONNX Simplifier](https://github.com/onnxsim/onnxsim) to optimize the computational graph:

```bash
$ onnxsim model.onnx model.sim.onnx --enable-onnxruntime-optimization
```

Also, the onnx model can be visualized in [Netron](https://netron.app).

### C/C++/Fortran API

The C API has two interchangeable backend implementations:

| Backend | Source | Runtime dependency |
| --- | --- | --- |
| ONNX Runtime | [`c_api/jaxpip_ort.cpp`](c_api/jaxpip_ort.cpp) | ONNX Runtime |
| OpenVINO | [`c_api/jaxpip_ov.cpp`](c_api/jaxpip_ov.cpp) | OpenVINO |

Both sources expose the same symbols and use [`c_api/jaxpip.h`](c_api/jaxpip.h)
as the C/C++ header. Link exactly one backend implementation into the final
executable.

For C/C++, include `c_api/jaxpip.h`, then call the runtime in this order:

```c
#include "c_api/jaxpip.h"

int main(void) {
    int n_atoms = 3;
    double xyz[9] = {/* x1, y1, z1, x2, y2, z2, x3, y3, z3 */};
    double energy;
    double forces[9];

    init_jaxpip_model("model.onnx");

    eval_jaxpip_model(xyz, &n_atoms, &energy, forces);

    finalize_jaxpip_model();
    return 0;
}
```

- Call `init_jaxpip_model` once before the MD loop.
- Call `eval_jaxpip_model` repeatedly inside the MD loop.
- Call `finalize_jaxpip_model` at the end to release backend resources.
- Coordinates and forces are flattened, row-major `double` arrays of length
  `3 * n_atoms`.
- Coordinates are in Angstroms.
- The ONNX model must have one input of shape `(N_atoms, 3)`.
- The ONNX model must have two outputs: scalar energy followed by forces of
  shape `(N_atoms, 3)`.
- Export `export_fn_energy_and_forces`; do not use an energy-only or batched
  model with this API.
- Double precision is required.
- Standard JaxPIP models produce energy in eV and forces in eV/Angstrom.

Example compiler invocations, with include and library paths adjusted for the
local runtime installation:

```bash
# ONNX Runtime
c++ -O3 -std=c++17 -I<onnxruntime-include> program.cpp c_api/jaxpip_ort.cpp \
    -L<onnxruntime-lib> -lonnxruntime

# OpenVINO
c++ -O3 -std=c++17 -I<openvino-include> program.cpp c_api/jaxpip_ov.cpp \
    -L<openvino-lib> -lopenvinoc
```

Fortran code can use the interface module in
[`c_api/jaxpip_mod.f90`](c_api/jaxpip_mod.f90) and link to either C++ backend.
The model path passed through the `ISO_C_BINDING` interface must be a
`c_char` array terminated by `c_null_char`.

## License

BSD 2-Clause License

## References

- (1) Xie, Z.; Bowman, J. M. Permutationally Invariant Polynomial Basis for Molecular Energy Surface Fitting via Monomial Symmetrization. _J. Chem. Theory Comput._ **2010**, _6_ (1), 26–34. https://doi.org/10.1021/ct9004917.
- (2) Nandi, A.; Qu, C.; Bowman, J. M. Using Gradients in Permutationally Invariant Polynomial Potential Fitting: A Demonstration for CH4 Using as Few as 100 Configurations. _J. Chem. Theory Comput._ **2019**, _15_ (5), 2826–2835. https://doi.org/10.1021/acs.jctc.9b00043.
- (3) Jiang, B.; Guo, H. Permutation Invariant Polynomial Neural Network Approach to Fitting Potential Energy Surfaces. _J. Chem. Phys._ **2013**, _139_ (5). https://doi.org/10.1063/1.4817187.
- (4) Li, J.; Jiang, B.; Guo, H. Permutation Invariant Polynomial Neural Network Approach to Fitting Potential Energy Surfaces. II. Four-Atom Systems. _J. Chem. Phys._ **2013**, _139_ (20). https://doi.org/10.1063/1.4832697.
- (5) Li, J.; Song, K.; Li, J. Tensorization over Factorization: Rethinking Permutation Invariant Polynomials in JAX. _ChemRxiv_, **2026**. https://doi.org/10.26434/chemrxiv.15000804/v1.
