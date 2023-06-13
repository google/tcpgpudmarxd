#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_CUDA_CONTEXT_MANAGER_CU_H_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_CUDA_CONTEXT_MANAGER_CU_H_

#include <string>

#include "cuda/common.cuh"


namespace tcpdirect {
class CudaContextManager {
 public:
  CudaContextManager(int gpu_cuda_idx);
  CudaContextManager(std::string gpu_pci_addr);
  void PushContext();
  void PopContext();
  ~CudaContextManager();

 private:
  CUcontext ctx;
  CUdevice dev;
};
}  // namespace tcpdirect

#endif
