import gzip
import json
import os
import sys
import time
from typing import Dict

import jax
import polars as pl
from jax import numpy as jnp

from jaxpip.descriptor import PolynomialDescriptor
from jaxpip.model import PolynomialLinearModel

_WARMUP_LOOP = 10
_CALC_LOOP = 100


def run_linear_benchmark(
    basis_file: str,
    alpha: float = 1.0,
    dtype: jnp.dtype = jnp.float64,
) -> Dict["str", float]:
    basis_dir = os.path.basename(os.path.dirname(basis_file))
    system, order = basis_dir.split("_")

    with gzip.open(basis_file, "rt") as f:
        basis_set = json.load(f)

    # bench jit
    start_jit = time.perf_counter()

    # 1. init descriptor
    desc = PolynomialDescriptor(
        basis_set=basis_set,
        alpha=alpha,
        with_grad=False,
        dtype=dtype,
    )

    # 2. linear model
    model = PolynomialLinearModel(descriptor=desc)

    # dummy data
    num_atoms = desc.num_atoms

    dummy_xyz = jnp.ones(shape=(num_atoms, 3), dtype=dtype)

    V, f = model.get_energy_and_forces(dummy_xyz)

    V.block_until_ready()
    f.block_until_ready()

    jit_time = time.perf_counter() - start_jit

    # 2. eval energy & forces
    # warmup
    for _ in range(_WARMUP_LOOP):
        model.get_energy_and_forces(dummy_xyz)

    # bench
    start_run = time.perf_counter()

    for _ in range(_CALC_LOOP):
        V, f = model.get_energy_and_forces(dummy_xyz)
        V.block_until_ready()
        f.block_until_ready()

    run_time = (time.perf_counter() - start_run) / _CALC_LOOP

    return {
        "system": system,
        "order": order,
        "num_pips": desc.num_pips,
        "num_flat_exponents": desc.num_flat_exponents,
        "jit": jit_time,
        "run": run_time,
    }


if __name__ == "__main__":
    # abspath required
    basis_file = sys.argv[1]

    res = run_linear_benchmark(basis_file=basis_file)

    pl.from_dicts(res).write_csv(f"tmp.csv")
