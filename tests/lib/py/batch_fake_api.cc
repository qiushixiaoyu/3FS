// Test-only synchronous API double: no cluster traffic, deterministic partial
// prepare, out-of-order/partial CQEs and per-object failures. Never linked into
// the production native_batch module.
#include <atomic>
#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <hf3fs_usrbio.h>
#include <unistd.h>
struct Op {
  bool read;
  void *ptr;
  int fd;
  size_t off, size;
  const void *user;
};
struct Ring {
  std::deque<Op> ops;
};
static std::atomic<int> live{0};
static int setting(const char *key) {
  const char *p = getenv(key);
  return p ? atoi(p) : 0;
}
extern "C" {
int hf3fs_test_live_objects() { return live; }
int hf3fs_extract_mount_point(char *p, int size, const char *) {
  strncpy(p, "/mock", size);
  return 6;
}
int hf3fs_reg_fd(int, uint64_t) {
  ++live;
  return 0;
}
void hf3fs_dereg_fd(int) { --live; }
int hf3fs_iovcreate(hf3fs_iov *v, const char *, size_t size, size_t, int) {
  v->base = (uint8_t *)malloc(size);
  v->size = size;
  ++live;
  return 0;
}
void hf3fs_iovdestroy(hf3fs_iov *v) {
  free(v->base);
  v->base = nullptr;
  --live;
}
int hf3fs_iorcreate4(hf3fs_ior *r, const char *, int, bool, int, int, int, uint64_t) {
  r->iorh = new Ring;
  ++live;
  return 0;
}
void hf3fs_iordestroy(hf3fs_ior *r) {
  auto *q = (Ring *)r->iorh;
  assert(q->ops.empty());
  delete q;
  --live;
}
int hf3fs_prep_io(const hf3fs_ior *r,
                  const hf3fs_iov *,
                  bool read,
                  void *ptr,
                  int fd,
                  size_t off,
                  uint64_t n,
                  const void *user) {
  auto &q = *(Ring *)r->iorh;
  if (setting("TEST_PREP_FAIL") && q.ops.size() == size_t(setting("TEST_PREP_FAIL"))) return -EAGAIN;
  q.ops.push_back({read, ptr, fd, off, size_t(n), user});
  return q.ops.size() - 1;
}
int hf3fs_submit_ios(const hf3fs_ior *) { return setting("TEST_SUBMIT_FAIL") ? -EIO : 0; }
int hf3fs_wait_for_ios(const hf3fs_ior *r, hf3fs_cqe *cq, int count, int, const timespec *) {
  auto &q = *(Ring *)r->iorh;
  int n = 0;
  if (setting("TEST_DELAY_MS")) usleep(setting("TEST_DELAY_MS") * 1000);
  while (!q.ops.empty() && n < count) {
    auto op = q.ops.back();
    q.ops.pop_back();
    auto result = uintptr_t(op.user) == uintptr_t(setting("TEST_CQE_ERROR")) ? -EIO
                  : op.read                                                  ? pread(op.fd, op.ptr, op.size, op.off)
                                                                             : pwrite(op.fd, op.ptr, op.size, op.off);
    cq[n++] = {0, 0, result, op.user};
    if (setting("TEST_WAIT_PARTIAL")) break;
  }
  return n;
}
}
