#include "cuda/cuda_ipc_memhandle.cuh"

#include "base/logging.h"
#include "cuda/common.cuh"

namespace tcpdirect {

CudaIpcMemhandle::CudaIpcMemhandle(const std::string& handle) {
  memcpy(&mem_handle_, handle.data(), handle.size());
  CU_ASSERT_SUCCESS(
      cuIpcOpenMemHandle(&ptr_, mem_handle_, cudaIpcMemLazyEnablePeerAccess));
}

CudaIpcMemhandle::~CudaIpcMemhandle() { cuIpcCloseMemHandle(ptr_); }

}  // namespace tcpdirect
