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

#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_GPU_PAGE_ALLOCATOR_INTERFACE_H_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_GPU_PAGE_ALLOCATOR_INTERFACE_H_

#include <cuda.h>

#include <vector>

#include "include/nic_configurator_interface.h"

namespace gpudirect_tcpxd {

enum class GpuPageAllocatorStatus {
  ALLOC_FAILURE,
  ALLOC_NETDEV_NETLINK_SUCCESS,
  ALLOC_DMA_BUF_IOCTL_SUCCESS
};

class GpuPageAllocatorInterface {
 public:
  virtual GpuPageAllocatorStatus AllocatePage(size_t pool_size, unsigned long* id,
                            const std::vector<int>& qids,
                            const bool use_netdev_bridge,
                            const std::string& ifname) = 0;
  virtual void FreePage(unsigned long id) = 0;
  virtual CUdeviceptr GetGpuMem(unsigned long id) = 0;
  virtual int GetGpuMemFd(unsigned long id) = 0;
  virtual void Reset() = 0;
  virtual ~GpuPageAllocatorInterface() {}
};
}  // namespace gpudirect_tcpxd

#endif
