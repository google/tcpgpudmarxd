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

#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_DMABUF_GPU_PAGE_ALLOCATOR_INTERFACE_H_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_DMABUF_GPU_PAGE_ALLOCATOR_INTERFACE_H_

#include <cuda.h>

#include <memory>
#include <string>
#include <unordered_map>

#include "cuda/gpu_page_allocator_interface.cuh"
#include "include/netdev_bridge.h"

namespace gpudirect_tcpxd {

struct GpuDmaBuf {
  int dma_buf_fd;
  int gpu_mem_fd;
  size_t size;
  CUdeviceptr gpu_mem_ptr;
};

// THIS IS NOT A THREAD-SAFE IMPLEMENTATION.
class DmabufGpuPageAllocator : public GpuPageAllocatorInterface {
 public:
  DmabufGpuPageAllocator(std::string gpu_pci_addr, std::string nic_pci_addr);
  DmabufGpuPageAllocator(std::string gpu_pci_addr, std::string nic_pci_addr,
                         bool create_page_pool, size_t pool_size);
  GpuPageAllocatorStatus AllocatePage(
      size_t size, unsigned long* id, const std::vector<int>& qids,
      const bool use_netdev_bridge, const std::string& ifname) override;
  void FreePage(unsigned long id) override;
  CUdeviceptr GetGpuMem(unsigned long id) override;
  int GetGpuMemFd(unsigned long id) override;
  void Reset() override {}
  ~DmabufGpuPageAllocator() override;

 private:
  size_t bytes_allocated_{0};
  std::string gpu_pci_addr_;
  std::string nic_pci_addr_;
  bool create_page_pool_{false};
  size_t pool_size_;
  std::unordered_map<unsigned long, GpuDmaBuf> gpu_dma_buf_map_;
  unsigned long next_id_{0};
  // disallow copy and assign
  DmabufGpuPageAllocator(const DmabufGpuPageAllocator &);
  void operator=(const DmabufGpuPageAllocator &);
};
}  // namespace gpudirect_tcpxd
#endif
