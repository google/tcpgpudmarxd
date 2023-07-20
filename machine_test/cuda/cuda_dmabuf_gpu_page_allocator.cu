#include <absl/log/log.h>
#include <absl/strings/str_format.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/types.h>
#include <sys/ioctl.h>

#include "cuda/common.cuh"
#include "machine_test/cuda/cuda_dmabuf_gpu_page_allocator.cuh"

namespace gpudirect_tcpxd {

namespace {
#define GPUMEM_DMA_BUF_CREATE _IOW('c', 'c', struct gpumem_dma_buf_create_info)

#define DMA_BUF_BASE 'b'
#define DMA_BUF_CREATE_PAGES \
  _IOW(DMA_BUF_BASE, 2, struct dma_buf_create_pages_info)

#define PAGE_SHIFT (12)
#define PAGE_SIZE (1 << PAGE_SHIFT)
#define DEFAULT_POOL_SIZE 0x400000
#define GPUMEM_ALIGNMENT (1UL << 21)

struct gpumem_dma_buf_create_info {
  unsigned long gpu_vaddr;
  unsigned long size;
};

struct dma_buf_create_pages_info {
  __u64 pci_bdf[3];
  __s32 dma_buf_fd;
  __s32 create_page_pool;
};
}  // namespace

CudaDmabufGpuPageAllocator::CudaDmabufGpuPageAllocator(std::string gpu_pci_addr,
                                                       std::string nic_pci_addr)
    : gpu_pci_addr_(gpu_pci_addr), nic_pci_addr_(nic_pci_addr) {
  create_page_pool_ = false;
  pool_size_ = DEFAULT_POOL_SIZE;
}

CudaDmabufGpuPageAllocator::CudaDmabufGpuPageAllocator(std::string gpu_pci_addr,
                                                       std::string nic_pci_addr,
                                                       bool create_page_pool,
                                                       size_t pool_size)
    : gpu_pci_addr_(gpu_pci_addr),
      nic_pci_addr_(nic_pci_addr),
      create_page_pool_(create_page_pool),
      pool_size_(pool_size) {}

void CudaDmabufGpuPageAllocator::AllocatePage(size_t size, unsigned long *id,
                                              bool *success) {
  if (size % GPUMEM_ALIGNMENT != 0) {
    size += GPUMEM_ALIGNMENT - (size % GPUMEM_ALIGNMENT);
  }

  if (size + bytes_allocated_ > pool_size_) {
    *success = false;
    return;
  }

  *id = next_id_;
  next_id_++;

  GpuDmaBuf &gpu_dma_buf = gpu_dma_buf_map_[*id];

  CUDA_ASSERT_SUCCESS(cudaMalloc((void **)&gpu_dma_buf.gpu_mem_ptr, size));
  cuMemGetHandleForAddressRange((void *)&gpu_dma_buf.dma_buf_fd,
                                gpu_dma_buf.gpu_mem_ptr, size,
                                CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);

  unsigned int flag = 1;
  CU_ASSERT_SUCCESS(cuPointerSetAttribute(
      &flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, gpu_dma_buf.gpu_mem_ptr));

  struct dma_buf_create_pages_info frags_create_info;
  frags_create_info.dma_buf_fd = gpu_dma_buf.dma_buf_fd;
  frags_create_info.create_page_pool = create_page_pool_;

  uint16_t pci_bdf[3];

  int ret = sscanf(nic_pci_addr_.c_str(), "0000:%hx:%hx.%hx", &pci_bdf[0],
                   &pci_bdf[1], &pci_bdf[2]);

  frags_create_info.pci_bdf[0] = pci_bdf[0];
  frags_create_info.pci_bdf[1] = pci_bdf[1];
  frags_create_info.pci_bdf[2] = pci_bdf[2];

  if (ret != 3) {
    LOG(ERROR) << "Invalid pci address.";
    *success = false;
    return;
  }

  gpu_dma_buf.gpu_mem_fd =
      ioctl(gpu_dma_buf.dma_buf_fd, DMA_BUF_CREATE_PAGES, &frags_create_info);

  if (gpu_dma_buf.gpu_mem_fd < 0) {
    LOG(ERROR) << "DMA_BUF_CREATE_PAGES failed!";
    *success = false;
    return;
  }

  gpu_dma_buf.size = size;
  bytes_allocated_ += size;
  *success = true;
}

void CudaDmabufGpuPageAllocator::FreePage(unsigned long id) {
  if (gpu_dma_buf_map_.find(id) == gpu_dma_buf_map_.end()) return;

  auto &gpu_dma_buf = gpu_dma_buf_map_[id];

  if (gpu_dma_buf.gpu_mem_fd >= 0) {
    close(gpu_dma_buf.gpu_mem_fd);
  }
  if (gpu_dma_buf.dma_buf_fd >= 0) {
    close(gpu_dma_buf.dma_buf_fd);
  }
  if (gpu_dma_buf.gpu_mem_ptr) {
    cuMemFree(gpu_dma_buf.gpu_mem_ptr);
  }
  bytes_allocated_ -= gpu_dma_buf.size;
}

CUdeviceptr CudaDmabufGpuPageAllocator::GetGpuMem(unsigned long id) {
  if (gpu_dma_buf_map_.find(id) == gpu_dma_buf_map_.end()) return 0;
  return gpu_dma_buf_map_[id].gpu_mem_ptr;
}

int CudaDmabufGpuPageAllocator::GetGpuMemFd(unsigned long id) {
  if (gpu_dma_buf_map_.find(id) == gpu_dma_buf_map_.end()) return -1;
  return gpu_dma_buf_map_[id].gpu_mem_fd;
}

CudaDmabufGpuPageAllocator::~CudaDmabufGpuPageAllocator() {
  for (auto &[id, _] : gpu_dma_buf_map_) {
    FreePage(id);
  }
}
}  // namespace gpudirect_tcpxd
