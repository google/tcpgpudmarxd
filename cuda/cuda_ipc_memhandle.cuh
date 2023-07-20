#ifndef _EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_CUDA_IPC_MEMHANDLE_H_
#define _EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_CUDA_IPC_MEMHANDLE_H_

#include <cuda.h>

#include <string>

#include "cuda/gpu_page_handle_interface.cuh"

namespace gpudirect_tcpxd {
class CudaIpcMemhandle : public GpuPageHandleInterface {
 public:
  CudaIpcMemhandle(const std::string& handle);
  ~CudaIpcMemhandle() override;
  CUdeviceptr GetGpuMem() override { return ptr_; }

 private:
  CUipcMemHandle mem_handle_;
  CUdeviceptr ptr_;
};
}  // namespace gpudirect_tcpxd

#endif
