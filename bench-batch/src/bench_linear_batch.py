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

_BATCH_SIZE_LIST = [
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
]

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

    _env = os.environ.copy()
    _env["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
    _env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    _env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    _env["OMP_NUM_THREADS"] = "1"
    _env["MKL_NUM_THREADS"] = "1"

    for basis_file in tqdm.tqdm(basis_list, desc="Basis Sets"):
        basis_file = os.path.abspath(basis_file)

        for batch_size in _BATCH_SIZE_LIST:
            # 1. 运行前确保清理旧的 tmp.csv，防止误读
            if os.path.exists("tmp.csv"):
                os.remove("tmp.csv")

            print(f"\n>>> Running {os.path.basename(basis_file)} - BS: {batch_size}")

            # 2. 运行子进程
            try:
                # 移除 check=True 以便手动处理错误
                result = subprocess.run(
                    args=shlex.split(
                        f"taskset -c 0 python3 -u {_BENCH_LINEAR_SINGLE} {basis_file} {batch_size}"
                    ),
                    env=_env,
                    capture_output=True, # 捕捉错误输出方便调试
                    text=True
                )

                # 3. 检查子进程是否成功执行且生成了数据
                if result.returncode == 0 and os.path.exists("tmp.csv"):
                    df_tmp = pd.read_csv("tmp.csv")
                    res = {}
                    for col in df_tmp.columns:
                        res[col] = df_tmp[col].values.tolist()[0]
                    res_buff.append(res)
                    print(f"--- Success: BS {batch_size} finished.")
                else:
                    # 如果返回码不为0，通常是 OOM 或 Segmentation Fault
                    print(f"--- FAILED: BS {batch_size} (Return Code: {result.returncode})")
                    if result.stderr:
                        # 只打印最后一行错误，防止刷屏
                        last_err = result.stderr.strip().split('\n')[-1]
                        print(f"    Error Log: {last_err}")

            except Exception as e:
                print(f"--- Unexpected Error at BS {batch_size}: {e}")

            # 4. 清理并稍微休眠
            if os.path.exists("tmp.csv"):
                os.remove("tmp.csv")
            
            # 如果发生了错误，建议多休眠一下让系统回收显存/内存
            time.sleep(1.0)

    if res_buff:
        df = pl.from_dicts(res_buff)
        df = df.sort("system")
        out_name = f"{os.path.basename(basis_set_dir)}-bench-linear.csv"
        df.write_csv(out_name)
        print(f"\nDone! Results saved to {out_name}")
    else:
        print("\nNo data collected. All runs failed.")