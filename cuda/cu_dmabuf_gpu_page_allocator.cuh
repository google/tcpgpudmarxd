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

#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_CU_DMABUF_GPU_PAGE_ALLOCATOR_INTERFACE_H_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_CU_DMABUF_GPU_PAGE_ALLOCATOR_INTERFACE_H_

#include <cuda.h>

#include <string>
#include <unordered_map>

#include "cuda/gpu_page_allocator_interface.cuh"
#include "include/ipc_gpumem_fd_metadata.h"

namespace gpudirect_tcpxd {

struct CuGpuDmaBuf {
  int dma_buf_fd{-1};
  int gpu_mem_fd{-1};
  int ipc_gpu_mem_fd{-1};
  size_t size;
  CUdeviceptr cu_dev_ptr;
  CUmemGenericAllocationHandle cu_gen_alloc_handle;
};

// THIS IS NOT A THREAD-SAFE IMPLEMENTATION.
class CuDmabufGpuPageAllocator : public GpuPageAllocatorInterface {
 public:
  CuDmabufGpuPageAllocator(int dev_id, std::string gpu_pci_addr,
                           std::string nic_pci_addr, size_t pool_size);
  void AllocatePage(size_t size, unsigned long* id, bool* success) override;
  void FreePage(unsigned long id) override;
  CUdeviceptr GetGpuMem(unsigned long id) override;
  int GetGpuMemFd(unsigned long id) override;
  void Reset() override {}
  void Cleanup();

  ~CuDmabufGpuPageAllocator() override;
  IpcGpuMemFdMetadata GetIpcGpuMemFdMetadata(unsigned long id);

 private:
  void AllocateGpuMem(CuGpuDmaBuf* gpu_dma_buf, size_t* size);
  void InitCuMemAllocationSettings();
  size_t AlignCuMemSize(size_t size);
  bool initialized_{false};
  int dev_id_{-1};
  size_t bytes_allocated_{0};
  std::string gpu_pci_addr_;
  std::string nic_pci_addr_;
  size_t pool_size_;
  std::unordered_map<unsigned long, CuGpuDmaBuf> gpu_dma_buf_map_;
  unsigned long next_id_{0};
  CUdevice dev_;
  CUcontext ctx_;
  CUmemAllocationProp cu_mem_alloc_prop_{};
  CUmemAccessDesc cu_mem_access_desc_{};
  CUmemAllocationHandleType cu_mem_handle_type_{
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR};
  size_t cu_mem_alloc_align_{1UL << 21};
  // disallow copy and assign
  CuDmabufGpuPageAllocator(const CuDmabufGpuPageAllocator&);
  void operator=(const CuDmabufGpuPageAllocator&);
};
}  // namespace gpudirect_tcpxd
#endif
