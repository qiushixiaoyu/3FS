"""Explicit real-cluster CPU smoke test. Use a new scratch file, not user data."""

import argparse
from pathlib import Path
import random

from hf3fs_py_batch import Batch, Buffers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--mode", required=True, choices=("write", "read"))
    args = parser.parse_args()
    page_size, count = 4096, 96
    path = Path(args.path)
    if args.mode == "write" and path.exists():
        raise FileExistsError("write test requires a new scratch file")
    if args.mode == "read" and not path.exists():
        raise FileNotFoundError(path)
    data = [bytearray(bytes([i]) * page_size) for i in range(count)]
    ring = Batch(
        str(path),
        page_size * count,
        page_size,
        8 if args.mode == "write" else 32,
        args.mode == "read",
    )
    try:
        if args.mode == "write":
            for start in range(0, count, 8):
                assert (
                    ring.io(
                        [i * page_size for i in range(start, start + 8)],
                        Buffers(data[start : start + 8]),
                    )
                    == [page_size] * 8
                )
            ring.flush()
        else:
            order = list(range(count))
            random.Random(7).shuffle(order)
            for start in range(0, count, 32):
                ids = order[start : start + 32]
                pages = [bytearray(page_size) for _ in ids]
                assert ring.io([i * page_size for i in ids], Buffers(pages)) == [
                    page_size
                ] * len(ids)
                assert pages == [data[i] for i in ids]
        print(
            f"PASS real cluster {args.mode}: {count} pages, {count * page_size} bytes",
            flush=True,
        )
    finally:
        ring.close()


if __name__ == "__main__":
    main()
