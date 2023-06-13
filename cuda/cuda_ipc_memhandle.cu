#include "cuda/cuda_ipc_memhandle.cu.h"

#include "base/logging.h"
#include "cuda/common.cu.h"

namespace tcpdirect {

CudaIpcMemhandle::CudaIpcMemhandle(const std::string& handle) {
  memcpy(&mem_handle_, handle.data(), handle.size());
  CU_ASSERT_SUCCESS(
      cuIpcOpenMemHandle(&ptr_, mem_handle_, cudaIpcMemLazyEnablePeerAccess));
}

CudaIpcMemhandle::~CudaIpcMemhandle() { cuIpcCloseMemHandle(ptr_); }

}  // namespace tcpdirect
