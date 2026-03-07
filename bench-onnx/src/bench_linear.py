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

_BENCH_LINEAR_ORT = os.path.abspath("bench_linear_single.x")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: $ python3 bench_linear.py BASIS_SET_DIR")
        sys.exit(-1)

    basis_set_dir = sys.argv[1]

    all_onnx_list = natsorted(glob.glob(os.path.join(
        basis_set_dir,
        "*",
        "*.onnx",
    )))

    onnx_list = [m for m in all_onnx_list if not m.endswith(".sim.onnx")]
    sim_onnx_list = [m for m in all_onnx_list if m.endswith(".sim.onnx")]

    target_list = onnx_list + sim_onnx_list

    res_buff = []

    for model_path in tqdm.tqdm(target_list, desc="ONNX Benchmarking"):
        model_path = os.path.abspath(model_path)
        is_sim = model_path.endswith(".sim.onnx")

        cmd = f"taskset -c 0 {_BENCH_LINEAR_ORT} {model_path} > tmp.csv"

        try:
            subprocess.run(
                cmd,
                shell=True,
                check=True,
                capture_output=True,
                text=True
            )

            if os.path.exists("tmp.csv"):
                df_tmp = pd.read_csv("tmp.csv")
                if not df_tmp.empty:
                    res = df_tmp.iloc[0].to_dict()
                    res["is_sim"] = is_sim
                    res["model_file"] = os.path.basename(model_path)
                    res_buff.append(res)

                os.remove("tmp.csv")
        except subprocess.CalledProcessError as e:
            print(f"\n[Error] system {os.path.basename(model_path)}")
            print(f"{e.stderr}")

        time.sleep(1.0)

    if res_buff:
        df = pl.from_dicts(res_buff)
        df = df.sort(["system", "order", "is_sim"])

        output_name = f"{os.path.basename(os.path.normpath(basis_set_dir))}-full-onnx-bench.csv"
        df.write_csv(output_name)
