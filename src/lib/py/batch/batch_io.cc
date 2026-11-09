// Owned CPU buffers -> one native prepare/submit/reap/copy call per batch.
// No Python userdata is accessed while the GIL is released.
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <hf3fs_usrbio.h>
#include <limits>
#include <mutex>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
namespace py = pybind11;
static void require(bool ok, const char *what) {
  if (!ok) throw std::runtime_error(what);
}

// Buffer exports keep their exporters and backing Tensor/ndarray storage alive.
// The caller retains exclusive access to these pages until io() returns.
class Buffers {
 public:
  std::vector<py::buffer_info> views;
  explicit Buffers(const py::list &objects) {
    for (auto obj : objects) {
      auto b = py::cast<py::buffer>(obj).request();
      ssize_t stride = b.itemsize;
      for (ssize_t dim = b.ndim; dim-- > 0;) {
        require(b.shape[dim] >= 0 && (b.shape[dim] <= 1 || b.strides[dim] == stride), "non-contiguous CPU buffer");
        stride *= b.shape[dim];
      }
      require(b.size > 0 && b.itemsize > 0, "empty CPU buffer");
      views.push_back(std::move(b));
    }
  }
};

class Batch {
  hf3fs_iov iov{};
  hf3fs_ior ior{};
  int fd = -1, depth;
  size_t bs, file_size, payload_size, pending = 0;
  bool read_mode, have_iov = false, have_ior = false, registered = false, closed = false;
  std::mutex mutex;

  void release_resources() {
    // Never release memory/FD while a prepared operation can still touch it.
    require(pending == 0, "outstanding I/O: resources retained");
    if (have_ior) {
      hf3fs_iordestroy(&ior);
      have_ior = false;
    }
    if (have_iov) {
      hf3fs_iovdestroy(&iov);
      have_iov = false;
    }
    if (registered) {
      hf3fs_dereg_fd(fd);
      registered = false;
    }
    if (fd >= 0) {
      ::close(fd);
      fd = -1;
    }
    closed = true;
  }

 public:
  Batch(const std::string &path, size_t size, size_t page_size, int entries, bool read, size_t payload = 0)
      : depth(entries),
        bs(page_size),
        file_size(size),
        payload_size(payload),
        read_mode(read) {
    require(depth > 0 && depth <= 4096 && bs > 0 && bs <= SIZE_MAX / depth, "invalid ring dimensions");
    require(size > 0 && size <= size_t(std::numeric_limits<off_t>::max()), "invalid file size");
    require(!payload_size || (payload_size <= bs && bs % 4096 == 0 && bs - payload_size < 4096),
            "padded page must be a payload rounded up to 4096 bytes");
    try {
      fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
      require(fd >= 0, "open failed");
      struct stat st{};
      require(::fstat(fd, &st) == 0, "fstat failed");
      // Do not silently truncate an existing file belonging to another layout.
      require(st.st_size == 0 || size_t(st.st_size) == size, "existing file size mismatch");
      if (st.st_size == 0) require(::ftruncate(fd, size) == 0, "ftruncate failed");
      char mount[256]{};
      int n = hf3fs_extract_mount_point(mount, sizeof(mount), path.c_str());
      require(n > 0 && size_t(n) <= sizeof(mount), "not a 3FS path");
      require(hf3fs_reg_fd(fd, 0) <= 0, "register FD failed");
      registered = true;
      require(hf3fs_iovcreate(&iov, mount, bs * depth, 0, -1) == 0, "iovcreate failed");
      have_iov = true;
      require(hf3fs_iorcreate4(&ior, mount, depth, read, 0, 1, -1, 0) == 0, "iorcreate failed");
      have_ior = true;
    } catch (...) {
      release_resources();
      throw;
    }
  }

  std::vector<int64_t> io(const std::vector<size_t> &offsets, Buffers &buffers) {
    require(offsets.size() == buffers.views.size() && offsets.size() <= size_t(depth), "invalid batch size");
    // Validate the WHOLE batch before preparing the first operation.
    std::vector<size_t> sizes;
    for (size_t i = 0; i < offsets.size(); ++i) {
      const auto &v = buffers.views[i];
      size_t bytes = size_t(v.size) * size_t(v.itemsize);
      require(bytes <= bs && offsets[i] <= file_size && bytes <= file_size - offsets[i],
              "page outside file/buffer bounds");
      require(!read_mode || !v.readonly, "read destination is read-only");
      require(!payload_size || (bytes == payload_size && offsets[i] % 4096 == 0 &&
                                bs <= file_size - offsets[i]), "invalid padded full page");
      sizes.push_back(bytes);
    }
    std::vector<int64_t> results(offsets.size(), 0);
    std::vector<hf3fs_cqe> cqes(offsets.size());
    std::vector<bool> seen(offsets.size(), false);
    py::gil_scoped_release no_gil;
    std::lock_guard<std::mutex> hold(mutex);
    require(!closed && pending == 0, "closed or undrained client");
    if (offsets.empty()) return results;
    if (!read_mode)
      for (size_t i = 0; i < offsets.size(); ++i) {
        std::memcpy(iov.base + i * bs, buffers.views[i].ptr, sizes[i]);
        if (payload_size) std::memset(iov.base + i * bs + payload_size, 0, bs - payload_size);
      }
    bool prep_error = false, completion_error = false;
    for (size_t i = 0; i < offsets.size(); ++i) {
      int r = hf3fs_prep_io(&ior,
                            &iov,
                            read_mode,
                            iov.base + i * bs,
                            fd,
                            offsets[i],
                            payload_size ? bs : sizes[i],
                            reinterpret_cast<void *>(i + 1));
      if (r < 0) {
        prep_error = true;
        break;
      }
      ++pending;
    }
    // Prepare can become visible before submit. Drain even on partial prepare.
    bool submit_error = pending && hf3fs_submit_ios(&ior) != 0;
    while (pending) {
      int got = hf3fs_wait_for_ios(&ior, cqes.data(), pending, pending, nullptr);
      if (got == -EINTR || got == -EAGAIN || got == 0) continue;
      // An invalid ring cannot be safely reclaimed. close()/destruction retain
      // registered resources in this exceptional case until process teardown.
      require(got > 0 && size_t(got) <= pending, "unrecoverable wait: outstanding resources retained");
      pending -= got;
      for (int j = 0; j < got; ++j) {
        size_t i = reinterpret_cast<uintptr_t>(cqes[j].userdata) - 1;
        if (i >= offsets.size() || seen[i]) {
          completion_error = true;
          continue;
        }
        seen[i] = true;
        auto result = cqes[j].result;
        // A short padded completion equal to the logical payload is NOT success.
        results[i] = payload_size && result >= 0 ? (result == int64_t(bs) ? int64_t(payload_size) : -EIO) : result;
      }
    }
    require(!prep_error && !submit_error && !completion_error, "batch submission/completion error (drained)");
    if (read_mode)
      for (size_t i = 0; i < offsets.size(); ++i)
        if (results[i] == int64_t(sizes[i])) std::memcpy(buffers.views[i].ptr, iov.base + i * bs, sizes[i]);
    return results;
  }

  void flush() {
    py::gil_scoped_release no_gil;
    std::lock_guard<std::mutex> hold(mutex);
    require(!closed && pending == 0, "closed or undrained client");
    require(::fsync(fd) == 0, "fsync failed");
  }

  void close() {
    py::gil_scoped_release no_gil;
    std::lock_guard<std::mutex> hold(mutex);
    release_resources();
  }
  ~Batch() {
    // Python holds self for every active method call; destruction has no racers.
    if (!pending) release_resources();
  }
};

#ifdef HF3FS_TEST_API
extern "C" int hf3fs_test_live_objects();
PYBIND11_MODULE(hf3fs_py_batch_test, m) {
  m.def("live_objects", &hf3fs_test_live_objects);
#else
PYBIND11_MODULE(hf3fs_py_batch, m) {
#endif
  py::class_<Buffers>(m, "Buffers").def(py::init<const py::list &>());
  py::class_<Batch>(m, "Batch")
      .def(py::init<const std::string &, size_t, size_t, int, bool, size_t>(),
           py::arg("path"), py::arg("size"), py::arg("page_size"), py::arg("entries"),
           py::arg("read"), py::arg("payload_size") = 0)
      .def("io", &Batch::io, py::arg("offsets"), py::arg("buffers"))
      .def("flush", &Batch::flush)
      .def("close", &Batch::close);
}
