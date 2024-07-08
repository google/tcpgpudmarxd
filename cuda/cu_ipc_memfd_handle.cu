/*
 Copyright 2024 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

#include <absl/log/log.h>
#include <absl/strings/str_format.h>

#include <string>

#include "cuda/common.cuh"
#include "cuda/cu_ipc_memfd_handle.cuh"

namespace gpudirect_tcpxd {

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

}  // namespace gpudirect_tcpxd
