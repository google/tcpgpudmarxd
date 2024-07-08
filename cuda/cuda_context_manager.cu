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

#include "cuda/cuda_context_manager.cuh"

namespace gpudirect_tcpxd {
CudaContextManager::CudaContextManager(int gpu_cuda_idx) {
  CU_ASSERT_SUCCESS(cuDeviceGet(&dev, gpu_cuda_idx));
  CU_ASSERT_SUCCESS(cuCtxCreate(&ctx, 0, dev));
}

CudaContextManager::CudaContextManager(std::string gpu_pci_addr) {
  CU_ASSERT_SUCCESS(cuDeviceGetByPCIBusId(&dev, gpu_pci_addr.c_str()));
  CU_ASSERT_SUCCESS(cuCtxCreate(&ctx, 0, dev));
}

void CudaContextManager::PushContext() {
  CU_ASSERT_SUCCESS(cuCtxPushCurrent(ctx));
}

void CudaContextManager::PopContext() {
  CUcontext old_ctx;
  CU_ASSERT_SUCCESS(cuCtxPopCurrent(&old_ctx));
}

CudaContextManager::~CudaContextManager() {
  CUcontext old_ctx;
  CU_ASSERT_SUCCESS(cuCtxPopCurrent(&old_ctx));
}
}  // namespace gpudirect_tcpxd
