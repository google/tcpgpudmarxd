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

#include <absl/log/log.h>
#include <absl/strings/str_format.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/types.h>
#include <sys/ioctl.h>

#include "cuda/common.cuh"
#include "cuda/cu_dmabuf_gpu_page_allocator.cuh"

namespace gpudirect_tcpxd {

namespace {
#define GPUMEM_DMA_BUF_CREATE _IOW('c', 'c', struct gpumem_dma_buf_create_info)

#define DMA_BUF_BASE 'b'
#define DMA_BUF_FRAGS_CREATE \
  _IOW(DMA_BUF_BASE, 2, struct dma_buf_frags_create_info)

#define PAGE_SHIFT (12)
#define PAGE_SIZE (1 << PAGE_SHIFT)
#define DEFAULT_POOL_SIZE 0x400000

struct gpumem_dma_buf_create_info {
  unsigned long gpu_vaddr;
  unsigned long size;
};

struct dma_buf_frags_create_info {
  __u64 pci_bdf[3];
  __s32 dma_buf_fd;
  __s32 create_page_pool;
};

}  // namespace

// This gpu page allocator always allocates a page pool and is intended to
// serve the rx buffers.
CuDmabufGpuPageAllocator::CuDmabufGpuPageAllocator(int dev_id,
                                                   std::string gpu_pci_addr,
                                                   std::string nic_pci_addr,
                                                   size_t pool_size)
    : dev_id_(dev_id),
      gpu_pci_addr_(gpu_pci_addr),
      nic_pci_addr_(nic_pci_addr),
      pool_size_(pool_size) {}

void CuDmabufGpuPageAllocator::AllocatePage(size_t size, unsigned long* id,
                                            bool* success) {
  *success = false;

  // lazy initialization
  if (!initialized_) {
    InitCuMemAllocationSettings();
    initialized_ = true;
  }

  if (size + bytes_allocated_ > pool_size_) {
    return;
  }

  *id = next_id_;
  next_id_++;

  CuGpuDmaBuf& gpu_dma_buf = gpu_dma_buf_map_[*id];
  AllocateGpuMem(&gpu_dma_buf, &size);

  CU_ASSERT_SUCCESS(cuMemGetHandleForAddressRange(
      (void*)&gpu_dma_buf.dma_buf_fd, gpu_dma_buf.cu_dev_ptr, size,
      CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0));

  LOG(INFO) << absl::StrFormat("Registered dmabuf region 0x%lx of %lu Bytes",
                               gpu_dma_buf.cu_dev_ptr, size);

  struct dma_buf_frags_create_info frags_create_info;
  frags_create_info.dma_buf_fd = gpu_dma_buf.dma_buf_fd;
  frags_create_info.create_page_pool = true;

  uint16_t pci_bdf[3];
  if (sscanf(nic_pci_addr_.c_str(), "0000:%hx:%hx.%hx", &pci_bdf[0],
             &pci_bdf[1], &pci_bdf[2]) != 3) {
    LOG(ERROR) << absl::StrFormat("Failed to parse NIC PCI bpf: %s",
                                  nic_pci_addr_.c_str());
    close(gpu_dma_buf.dma_buf_fd);
    return;
  }

  frags_create_info.pci_bdf[0] = pci_bdf[0];
  frags_create_info.pci_bdf[1] = pci_bdf[1];
  frags_create_info.pci_bdf[2] = pci_bdf[2];

  gpu_dma_buf.gpu_mem_fd =
      ioctl(gpu_dma_buf.dma_buf_fd, DMA_BUF_FRAGS_CREATE, &frags_create_info);

  if (gpu_dma_buf.gpu_mem_fd < 0) {
    PLOG(ERROR) << "Error getting dma_buf frags: ";
    close(gpu_dma_buf.dma_buf_fd);
    return;
  }

  gpu_dma_buf.size = size;
  bytes_allocated_ += size;
  *success = true;
}

void CuDmabufGpuPageAllocator::FreePage(unsigned long id) {
  if (gpu_dma_buf_map_.find(id) == gpu_dma_buf_map_.end()) return;

  auto& gpu_dma_buf = gpu_dma_buf_map_[id];

  if (gpu_dma_buf.dma_buf_fd >= 0) {
    close(gpu_dma_buf.dma_buf_fd);
  }
  if (gpu_dma_buf.gpu_mem_fd >= 0) {
    close(gpu_dma_buf.gpu_mem_fd);
  }
  if (gpu_dma_buf.ipc_gpu_mem_fd >= 0) {
    close(gpu_dma_buf.ipc_gpu_mem_fd);
  }
  if (gpu_dma_buf.cu_dev_ptr) {
    cuMemUnmap(gpu_dma_buf.cu_dev_ptr, gpu_dma_buf.size);
    cuMemRelease(gpu_dma_buf.cu_gen_alloc_handle);
    cuMemAddressFree(gpu_dma_buf.cu_dev_ptr, gpu_dma_buf.size);
  }
  bytes_allocated_ -= gpu_dma_buf.size;
}

CUdeviceptr CuDmabufGpuPageAllocator::GetGpuMem(unsigned long id) {
  if (gpu_dma_buf_map_.find(id) == gpu_dma_buf_map_.end()) return 0;
  return gpu_dma_buf_map_[id].cu_dev_ptr;
}

int CuDmabufGpuPageAllocator::GetGpuMemFd(unsigned long id) {
  if (gpu_dma_buf_map_.find(id) == gpu_dma_buf_map_.end()) return -1;
  return gpu_dma_buf_map_[id].gpu_mem_fd;
}

void CuDmabufGpuPageAllocator::AllocateGpuMem(CuGpuDmaBuf* gpu_dma_buf,
                                              size_t* size) {
  size_t aligned_size = AlignCuMemSize(*size);
  *size = aligned_size;
  CU_ASSERT_SUCCESS(cuMemCreate(&gpu_dma_buf->cu_gen_alloc_handle, *size,
                                &cu_mem_alloc_prop_, 0));
  CU_ASSERT_SUCCESS(cuMemExportToShareableHandle(
      &gpu_dma_buf->ipc_gpu_mem_fd, gpu_dma_buf->cu_gen_alloc_handle,
      cu_mem_handle_type_, 0));
  CU_ASSERT_SUCCESS(cuMemAddressReserve(&gpu_dma_buf->cu_dev_ptr, *size,
                                        cu_mem_alloc_align_, 0, 0));
  CU_ASSERT_SUCCESS(cuMemMap(gpu_dma_buf->cu_dev_ptr, *size, 0,
                             gpu_dma_buf->cu_gen_alloc_handle, 0));
  CU_ASSERT_SUCCESS(cuMemSetAccess(gpu_dma_buf->cu_dev_ptr, *size,
                                   &cu_mem_access_desc_, 1 /*count*/));
  LOG(INFO) << absl::StrFormat(
      "Allocated GPU memory with size: %ld, align: %ld", *size,
      cu_mem_alloc_align_);
}

void CuDmabufGpuPageAllocator::InitCuMemAllocationSettings() {
  CU_ASSERT_SUCCESS(cuDeviceGet(&dev_, dev_id_));
  CU_ASSERT_SUCCESS(cuDevicePrimaryCtxRetain(&ctx_, dev_));
  cu_mem_alloc_prop_ = {};
  cu_mem_alloc_prop_.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  cu_mem_alloc_prop_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  cu_mem_alloc_prop_.location.id = dev_id_;
  cu_mem_alloc_prop_.requestedHandleTypes = cu_mem_handle_type_;
  CU_ASSERT_SUCCESS(
      cuMemGetAllocationGranularity(&cu_mem_alloc_align_, &cu_mem_alloc_prop_,
                                    CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  cu_mem_access_desc_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  cu_mem_access_desc_.location.id = dev_id_;
  cu_mem_access_desc_.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
}

size_t CuDmabufGpuPageAllocator::AlignCuMemSize(size_t size) {
  if ((size % cu_mem_alloc_align_) == 0) return size;
  return size + (cu_mem_alloc_align_ - (size % cu_mem_alloc_align_));
}

IpcGpuMemFdMetadata CuDmabufGpuPageAllocator::GetIpcGpuMemFdMetadata(
    unsigned long id) {
  if (gpu_dma_buf_map_.find(id) == gpu_dma_buf_map_.end())
    return {.fd = -1, .size = 0, .align = 0};
  return {
      .fd = gpu_dma_buf_map_[id].ipc_gpu_mem_fd,
      .size = gpu_dma_buf_map_[id].size,
      .align = cu_mem_alloc_align_,
  };
}

void CuDmabufGpuPageAllocator::Cleanup() {
  for (auto& [id, _] : gpu_dma_buf_map_) {
    FreePage(id);
  }
}

CuDmabufGpuPageAllocator::~CuDmabufGpuPageAllocator() {}
}  // namespace gpudirect_tcpxd
