import glob
import os
import shlex
import subprocess
import sys

import jax
import jax2onnx
import onnx
import pandas as pd
import polars as pl
import tqdm
from jax import numpy as jnp
from natsort import natsorted

from jaxpip.descriptor import PolynomialDescriptor
from jaxpip.model import PolynomialLinearModel


def jaxpip_linear2onnx(
    basis_file: str,
) -> None:
    descriptor = PolynomialDescriptor.from_file(
        basis_file=basis_file,
        alpha=1.0,
        with_grad=False,
        dtype=jnp.float64,
    )

    linear_model = PolynomialLinearModel(
        descriptor=descriptor,
    )

    key = jax.random.PRNGKey(114514)

    coeffs = jax.random.normal(
        key,
        shape=(descriptor.num_pips,),
        dtype=descriptor.dtype
    )

    linear_model = linear_model.update_coeffs(coeffs)

    @jax.jit
    def export_wrapper(xyz):
        return linear_model.get_energy_and_forces(xyz)

    linear_model_onnx = jax2onnx.to_onnx(
        fn=export_wrapper,
        inputs=[
            (descriptor.num_atoms, 3),
        ],
        enable_double_precision=True,
    )

    return linear_model_onnx


if __name__ == "__main__":
    basis_file = os.path.abspath("MOL_5_2_1_7.json.gz")

    basis_dir = os.path.dirname(basis_file)
    basis_name = os.path.basename(basis_file).split(".")[0]

    onnx_file = os.path.abspath(
        os.path.join(basis_dir, f"{basis_name}.onnx"))
    sim_onnx_file = os.path.abspath(
        os.path.join(basis_dir, f"{basis_name}.sim.onnx"))

    print(basis_file)
    print(onnx_file)
    print(sim_onnx_file)

    onnx_model = jaxpip_linear2onnx(basis_file)

    onnx.save(
        onnx_model,
        onnx_file,
        save_as_external_data=True,
    )

    
