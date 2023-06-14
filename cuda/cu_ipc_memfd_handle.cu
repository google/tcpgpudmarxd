#include "cuda/cu_ipc_memfd_handle.cuh"

#include <string>

#include <absl/log/log.h>
#include "cuda/common.cuh"
#include <absl/strings/str_format.h>

namespace tcpdirect {

CuIpcMemfdHandle::CuIpcMemfdHandle(int fd, int dev_id, size_t size,
                                   size_t align) {
  LOG(INFO) << absl::StrFormat(
      "Importing CUDA IPC mem from from fd: %ld, dev_id: %ld, size: %ld, "
      "align: %ld",
      fd, dev_id, size, align);
  CU_ASSERT_SUCCESS(cuDeviceGet(&dev_, dev_id));
  CU_ASSERT_SUCCESS(cuDevicePrimaryCtxRetain(&ctx_, dev_));
  size_ = size;
  CU_ASSERT_SUCCESS(
      cuMemImportFromShareableHandle(&handle_, (void*)(long long)fd,
                                     CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
  CU_ASSERT_SUCCESS(cuMemAddressReserve(&ptr_, size_, align, 0, 0));
  CU_ASSERT_SUCCESS(cuMemMap(ptr_, size_, 0, handle_, 0));
  close(fd);
  CUmemAccessDesc desc = {};
  desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  desc.location.id = dev_id;
  desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CU_ASSERT_SUCCESS(cuMemSetAccess(ptr_, size_, &desc, 1 /*count*/));
}

CuIpcMemfdHandle::~CuIpcMemfdHandle() {
  cuMemUnmap(ptr_, size_);
  cuMemRelease(handle_);
  cuMemAddressFree(ptr_, size_);
}

}  // namespace tcpdirect
