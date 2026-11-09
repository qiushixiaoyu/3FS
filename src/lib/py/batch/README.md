# Native Python batch I/O

`hf3fs_py_batch` is an optional CPU buffer-protocol interface over USRBIO.
It does not depend on PyTorch or SGLang and does not replace the existing
`hf3fs_py_usrbio` API. Each `Batch` owns one registered FD, one Iov and one IOR.
An `io()` call prepares a batch, submits it, reaps all completions and copies
successful reads to their corresponding destinations. Numeric completion IDs
preserve caller order even when completions arrive out of order.

## Build

The normal 3FS CMake build includes the `hf3fs_py_batch` target. To build only
this binding against an existing matching API installation:

```sh
cmake -S src/lib/py/batch -B build-py-batch \
  -DHF3FS_API_ROOT=/path/to/3fs-api \
  -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)" \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build build-py-batch -j
export PYTHONPATH="$PWD/build-py-batch${PYTHONPATH:+:$PYTHONPATH}"
```

The API installation must contain `include/hf3fs_usrbio.h` and
`lib/libhf3fs_api_shared.so`. The running FUSE client must be compatible with
that API library. Ensure its shared-library dependencies are discoverable by
the dynamic loader. An optional `cmake --install` installs the extension under
`<prefix>/lib`.

## Use

```python
from hf3fs_py_batch import Batch, Buffers

writer = Batch("/mnt/3fs/pages.bin", 1 << 30, 4096, 8, False)
reader = Batch("/mnt/3fs/pages.bin", 1 << 30, 4096, 32, True)
try:
    assert writer.io([0], Buffers([bytes([7]) * 4096])) == [4096]
    page = bytearray(4096)
    assert reader.io([0], Buffers([page])) == [4096]
    assert page == bytes([7]) * 4096
finally:
    reader.close()
    writer.close()
```

Buffers must be nonempty, contiguous CPU buffers; reads require writable
destinations. A retained Python buffer export owns the backing allocation.
The caller must not concurrently mutate or reuse these buffers until `io()`
returns. All bounds are checked before the first I/O is prepared. Each result
is the actual byte count or negative errno; failed/short reads leave their
destination unchanged. Existing nonempty files must have the requested size.

The GIL is released during native I/O. Calls on one ring are serialized; use
separate readers/writers for independent depths. There is still a CPU memcpy
between the user buffers and registered staging memory. This is neither
GPU-direct I/O nor a continuously replenished asynchronous queue.

Partial preparation/submission failures drain prepared operations before
raising. There is no timeout that returns buffers while I/O can still use
them: a stalled backend can block `io()` and `close()`. If the ring reports an
unrecoverable wait error, native resources remain retained until process
teardown. `close()` is idempotent after successful drain. `flush()` uses fsync;
it does not implement application metadata publication or change 3FS replica
acknowledgement semantics.

## Validation

`tests/lib/py/test_batch_io.py` exercises ordering, ownership, bounds and
failure draining using a deterministic API double; `run_batch_tests.py`
builds that double with ASAN/UBSAN. `test_batch_io_cluster.py` is an explicit
real-cluster smoke test using a caller-provided scratch file. Neither test
loads a model or allocates GPU memory.
