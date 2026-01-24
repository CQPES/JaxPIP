import glob
import os
import shlex
import subprocess
import sys
import time

import pandas as pd
import polars as pl
import tqdm
from natsort import natsorted

_BENCH_LINEAR_SINGLE = os.path.join(
    os.path.dirname(__file__),
    "bench_linear_single.py",
)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: $ python3 bench_linear.py BASIS_SET_DIR")

        exit(-1)

    basis_set_dir = sys.argv[1]

    basis_list = natsorted(glob.glob(os.path.join(
        basis_set_dir,
        "*",
        "*.json.gz",
    )))

    res_buff = []

    for basis_file in tqdm.tqdm(basis_list):
        basis_file = os.path.abspath(basis_file)

        env = os.environ.copy()

        env["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"

        subprocess.run(
            args=shlex.split(
                f"taskset -c 0 python3 -u {_BENCH_LINEAR_SINGLE} {basis_file}"
            ),
            env=env,
        )

        df_tmp = pd.read_csv("tmp.csv")

        res = {}

        for col in df_tmp.columns:
            res[col] = df_tmp[col].values.tolist()[0]

        res_buff.append(res)

        os.remove("tmp.csv")

        time.sleep(1.0)

    df = pl.from_dicts(res_buff)
    df = df.sort("system")

    df.write_csv(f"{os.path.basename(basis_set_dir)}-bench-linear.csv")
