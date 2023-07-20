#ifndef _EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_CUDA_GPU_PAGE_HANDLE_INTERFACE_H_
#define _EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_CUDA_GPU_PAGE_HANDLE_INTERFACE_H_

#include <cuda.h>

namespace gpudirect_tcpxd {
class GpuPageHandleInterface {
 public:
  virtual ~GpuPageHandleInterface() = default;
  virtual CUdeviceptr GetGpuMem() = 0;
};
}  // namespace gpudirect_tcpxd
#endif
