import glob
import gzip
import json
import os
import time

import jax
import tqdm
from jax import numpy as jnp
from natsort import natsorted

from jaxpip.descriptor import PIPDescriptor

import polars as pl

_WARMUP_LOOP = 10
_CALC_LOOP = 1_000


if __name__ == "__main__":
    # init jax prng key
    key = jax.random.PRNGKey(114514)

    # load basis
    basis_list = natsorted(glob.glob(os.path.join("*", "*.json.gz")))

    # results buffer
    res_buff = []

    for basis_file in tqdm.tqdm(basis_list):
        # parse system name and pip order
        system, order = os.path.dirname(basis_file).split("_")

        # load basis from file
        with gzip.open(basis_file, "rt") as f:
            basis_set = json.load(f)

        # bench pip
        # 1. jit
        pip_jit_time_start = time.perf_counter()

        pip = PIPDescriptor(
            basis_set=basis_set,
            alpha=1.0,
            with_grad=False,
            dtype=jnp.float64,
        )

        pip_jit_time_end = time.perf_counter()

        pip_jit_time = pip_jit_time_end - pip_jit_time_start

        # 2. calc
        xyz = jax.random.normal(key=key, shape=(pip.num_atoms, 3))

        # warmup
        for _ in range(_WARMUP_LOOP):
            _ = pip(xyz)

        pip_calc_time_start = time.perf_counter()

        for _ in range(_CALC_LOOP):
            _ = pip(xyz)

        pip_calc_time_end = time.perf_counter()

        pip_calc_time = pip_calc_time_end - pip_calc_time_start

        # bench pip with grad
        # 1. jit
        pip_with_grad_jit_time_start = time.perf_counter()

        pip_with_grad = PIPDescriptor(
            basis_set=basis_set,
            alpha=1.0,
            with_grad=True,
            dtype=jnp.float64,
        )

        pip_with_grad_jit_time_end = time.perf_counter()

        pip_with_grad_jit_time = \
            pip_with_grad_jit_time_end - pip_with_grad_jit_time_start

        # 2. calc
        # warmup
        for _ in range(_WARMUP_LOOP):
            _ = pip_with_grad(xyz)

        pip_with_grad_calc_time_start = time.perf_counter()

        for _ in range(_CALC_LOOP):
            _ = pip_with_grad(xyz)

        pip_with_grad_calc_time_end = time.perf_counter()

        pip_with_grad_calc_time = \
            (pip_with_grad_calc_time_end - pip_with_grad_calc_time_start)

        # collect
        res_buff.append({
            "system": system,
            "order": order,
            "num_pips": len(basis_set),
            "pip_jit(s)": pip_jit_time,
            "pip_calc(ms)": pip_calc_time / _CALC_LOOP * 1.0e+03,
            "pip_with_grad_jit(s)": pip_with_grad_jit_time,
            "pip_with_grad_calc_time(ms)": pip_with_grad_calc_time / _CALC_LOOP * 1.0e+03,
        })

    df = pl.from_dicts(res_buff)
    df = df.sort("system")

    df.write_csv("bench.csv")
