#ifndef _EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_CUDA_IPC_MEMHANDLE_H_
#define _EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_CUDA_IPC_MEMHANDLE_H_

#include <cuda.h>

#include <string>

#include "cuda/gpu_page_handle_interface.cuh"

namespace tcpdirect {
// Note: Users are responsibile for initializing the CUDA Primary Context of
// the device.
class CuIpcMemfdHandle : public GpuPageHandleInterface {
 public:
  CuIpcMemfdHandle(int fd, int dev_id, size_t size, size_t align);
  ~CuIpcMemfdHandle() override;
  CUdeviceptr GetGpuMem() override { return ptr_; }

 private:
  CUdevice dev_;
  CUcontext ctx_;
  CUmemGenericAllocationHandle handle_;
  CUdeviceptr ptr_;
  size_t size_;
};
}  // namespace tcpdirect

#endif
