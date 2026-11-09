"""Build the native API double with ASAN/UBSAN, then run CPU regressions."""

import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile

import pybind11

root = Path(__file__).resolve().parents[3]
with tempfile.TemporaryDirectory(prefix="hf3fs-batch-tests-") as directory:
    module = Path(directory) / (
        "hf3fs_py_batch_test" + sysconfig.get_config_var("EXT_SUFFIX")
    )
    command = [
        os.environ.get("CXX", "g++"),
        "-O1",
        "-g",
        "-shared",
        "-std=c++17",
        "-fPIC",
        "-DHF3FS_TEST_API",
        "-fsanitize=address,undefined",
        "-fno-omit-frame-pointer",
        "-I" + pybind11.get_include(),
        "-I" + sysconfig.get_paths()["include"],
        "-I" + str(root / "src/lib/api"),
        str(root / "src/lib/py/batch/batch_io.cc"),
        str(root / "tests/lib/py/batch_fake_api.cc"),
        "-o",
        str(module),
    ]
    print("Build:", command, flush=True)
    subprocess.run(command, check=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = directory
    env["LD_PRELOAD"] = ":".join(
        subprocess.check_output(
            [command[0], "-print-file-name=" + library], text=True
        ).strip()
        for library in ("libasan.so", "libstdc++.so")
    )
    env["ASAN_OPTIONS"] = "detect_leaks=0:halt_on_error=1"
    env["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"
    subprocess.run(
        [sys.executable, str(root / "tests/lib/py/test_batch_io.py")],
        env=env,
        check=True,
    )
