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

#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_CUDA_CONTEXT_MANAGER_CU_H_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_CUDA_CONTEXT_MANAGER_CU_H_

#include <cuda.h>

#include <string>

#include "cuda/common.cuh"

namespace gpudirect_tcpxd {
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
}  // namespace gpudirect_tcpxd

#endif
