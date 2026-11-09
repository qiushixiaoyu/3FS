"""Deterministic native lifetime/failure tests; no cluster or GPU required."""

import concurrent.futures
import gc
import os
from pathlib import Path
import tempfile
import unittest

import hf3fs_py_batch_test as native


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.temp.name) / "pages.bin")
        self.reader = native.Batch(self.path, 4096, 64, 8, True)
        self.writer = native.Batch(self.path, 4096, 64, 8, False)
        self.offsets = list(range(0, 512, 64))
        self.data = [bytearray([i + 1] * 64) for i in range(8)]
        self.assertEqual(
            self.writer.io(self.offsets, native.Buffers(self.data)), [64] * 8
        )

    def tearDown(self):
        for key in list(os.environ):
            if key.startswith("TEST_"):
                os.environ.pop(key)
        self.reader.close()
        self.writer.close()
        gc.collect()
        self.assertEqual(native.live_objects(), 0)
        self.temp.cleanup()

    def test_out_of_order_partial_completions(self):
        os.environ["TEST_WAIT_PARTIAL"] = "1"
        out = [bytearray(64) for _ in range(8)]
        self.assertEqual(self.reader.io(self.offsets, native.Buffers(out)), [64] * 8)
        self.assertEqual(out, self.data)

    def test_failed_read_does_not_copy_staging_bytes(self):
        os.environ["TEST_CQE_ERROR"] = "4"
        out = [bytearray([123] * 64) for _ in range(8)]
        result = self.reader.io(self.offsets, native.Buffers(out))
        self.assertLess(result[3], 0)
        self.assertEqual(out[3], bytearray([123] * 64))
        for i in range(8):
            if i != 3:
                self.assertEqual(out[i], self.data[i])

    def test_prepare_and_submit_failures_drain(self):
        for setting in ("TEST_PREP_FAIL", "TEST_SUBMIT_FAIL"):
            with self.subTest(setting=setting):
                os.environ[setting] = "3"
                out = [bytearray(64) for _ in range(8)]
                with self.assertRaises(RuntimeError):
                    self.reader.io(self.offsets, native.Buffers(out))
                os.environ.pop(setting)
                self.assertEqual(
                    self.reader.io(self.offsets, native.Buffers(out)), [64] * 8
                )
                self.assertEqual(out, self.data)

    def test_validate_entire_batch_before_io(self):
        for offsets, buffers in (
            ([0, 4096], [bytearray(64)] * 2),
            ([0], [bytearray(65)]),
            ([0], [bytes(64)]),
            ([0, 64], [bytearray(64)]),
        ):
            with self.assertRaises(RuntimeError):
                self.reader.io(offsets, native.Buffers(buffers))
        with self.assertRaises(RuntimeError):
            native.Buffers([memoryview(bytearray(128))[::2]])
        self.assertEqual(self.reader.io([], native.Buffers([])), [])
        out = [bytearray(64) for _ in range(8)]
        self.assertEqual(self.reader.io(self.offsets, native.Buffers(out)), [64] * 8)
        self.assertEqual(out, self.data)

    def test_exports_own_and_pin_python_buffer(self):
        page = bytearray(64)
        owner = native.Buffers([page])
        with self.assertRaises(BufferError):
            page.extend(b"x")
        self.reader.io([0], owner)
        self.assertEqual(page, self.data[0])
        del owner
        gc.collect()
        page.extend(b"x")
        # The exporter remains alive even when its last Python name is gone.
        owner = native.Buffers([bytearray(64)])
        self.assertEqual(self.reader.io([64], owner), [64])

    def test_concurrent_calls_and_idempotent_close(self):
        def read():
            out = [bytearray(64) for _ in range(8)]
            self.assertEqual(
                self.reader.io(self.offsets, native.Buffers(out)), [64] * 8
            )
            self.assertEqual(out, self.data)

        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            list(pool.map(lambda _: read(), range(32)))
        self.writer.flush()
        self.reader.close()
        self.reader.close()
        with self.assertRaises(RuntimeError):
            self.reader.io([0], native.Buffers([bytearray(64)]))

    def test_existing_file_size_not_changed(self):
        with self.assertRaises(RuntimeError):
            native.Batch(self.path, 8192, 64, 8, True)
        self.assertEqual(Path(self.path).stat().st_size, 4096)


if __name__ == "__main__":
    unittest.main(verbosity=2)
