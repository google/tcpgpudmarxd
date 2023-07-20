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
