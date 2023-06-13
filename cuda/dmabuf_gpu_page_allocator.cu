#include "cuda/dmabuf_gpu_page_allocator.cuh"

#include <errno.h>
#include <fcntl.h>
#include <linux/types.h>
#include <sys/ioctl.h>

#include "base/logging.h"
#include "cuda/common.cuh"
#include <absl/strings/str_format.h>

namespace tcpdirect {

namespace {
#define GPUMEM_DMA_BUF_CREATE _IOW('c', 'c', struct gpumem_dma_buf_create_info)

#define DMA_BUF_BASE 'b'
#define DMA_BUF_CREATE_PAGES \
  _IOW(DMA_BUF_BASE, 2, struct dma_buf_create_pages_info)

#define PAGE_SHIFT (12)
#define PAGE_SIZE (1 << PAGE_SHIFT)
#define DEFAULT_POOL_SIZE 0x400000

struct gpumem_dma_buf_create_info {
  unsigned long gpu_vaddr;
  unsigned long size;
};

struct dma_buf_create_pages_info {
  __u64 pci_bdf[3];
  __s32 dma_buf_fd;
  __s32 create_page_pool;
};

static const std::string kNvp2pDmabufProcfsPrefix =
    "/proc/driver/nvp2p_dma_buf";

int get_gpumem_dmabuf_pages_fd(const std::string &gpu_pci_addr,
                               const std::string &nic_pci_addr,
                               bool create_page_pool,
                               CUdeviceptr gpu_mem, size_t gpu_mem_sz,
                               int *dma_buf_fd) {
  std::string path =
      absl::StrFormat("%s/%s/new_fd", kNvp2pDmabufProcfsPrefix, gpu_pci_addr);

  int err, ret;
  int fd = open(path.c_str(), O_WRONLY);
  if (fd == -1) {
    LOG(ERROR) << absl::StrFormat("Error opening %s", path);
    return -EBADF;
  }

  struct gpumem_dma_buf_create_info create_info = {gpu_mem, gpu_mem_sz};
  ret = ioctl(fd, GPUMEM_DMA_BUF_CREATE, &create_info);
  if (ret < 0) {
    PLOG(ERROR) << "ioctl() failed: " << ret << " ";
    err = -EIO;
    goto err_close;
  }

  if (close(fd)) {
    PLOG(ERROR) << "close: ";
    err = -EIO;
    return err;
  }

  // LOG(INFO) << absl::StrFormat("Registered dmabuf region 0x%lx of %lu Bytes",
  //                              gpu_mem, gpu_mem_sz);

  *dma_buf_fd = ret;
  struct dma_buf_create_pages_info frags_create_info;
  frags_create_info.dma_buf_fd = *dma_buf_fd;
  frags_create_info.create_page_pool = create_page_pool;

  uint16_t pci_bdf[3];
  ret = sscanf(nic_pci_addr.c_str(), "0000:%hx:%hx.%hx", &pci_bdf[0],
               &pci_bdf[1], &pci_bdf[2]);
  frags_create_info.pci_bdf[0] = pci_bdf[0];
  frags_create_info.pci_bdf[1] = pci_bdf[1];
  frags_create_info.pci_bdf[2] = pci_bdf[2];
  if (ret != 3) {
    err = -EINVAL;
    goto err_close_dmabuf;
  }

  ret = ioctl(*dma_buf_fd, DMA_BUF_CREATE_PAGES, &frags_create_info);
  if (ret < 0) {
    PLOG(ERROR) << "Error getting dma_buf frags: ";
    err = -EIO;
    goto err_close_dmabuf;
  }
  return ret;

err_close_dmabuf:
  close(*dma_buf_fd);
  return err;
err_close:
  close(fd);
  return err;
}
}  // namespace

DmabufGpuPageAllocator::DmabufGpuPageAllocator(std::string gpu_pci_addr,
                                               std::string nic_pci_addr)
    : gpu_pci_addr_(gpu_pci_addr), nic_pci_addr_(nic_pci_addr) {
  create_page_pool_ = false;
  pool_size_ = DEFAULT_POOL_SIZE;
}

DmabufGpuPageAllocator::DmabufGpuPageAllocator(
    std::string gpu_pci_addr, std::string nic_pci_addr, bool create_page_pool,
    size_t pool_size)
    : gpu_pci_addr_(gpu_pci_addr),
      nic_pci_addr_(nic_pci_addr),
      create_page_pool_(create_page_pool),
      pool_size_(pool_size) {}

void DmabufGpuPageAllocator::AllocatePage(size_t size, unsigned long *id,
                                          bool *success) {
  // // may not need this for dmabuf
  // size_t alloc_size = std::max(size, (unsigned long)GPUMEM_MINSZ);
  // if (alloc_size % GPUMEM_ALIGNMENT != 0) {
  //   alloc_size += GPUMEM_ALIGNMENT - (alloc_size % GPUMEM_ALIGNMENT);
  // }
  if (size + bytes_allocated_ > pool_size_) {
    *success = false;
    return;
  }
  *id = next_id_;
  next_id_++;
  GpuDmaBuf &gpu_dma_buf = gpu_dma_buf_map_[*id];
  CU_ASSERT_SUCCESS(cuMemAlloc(&gpu_dma_buf.gpu_mem_ptr, size));

  unsigned int flag = 1;
  CU_ASSERT_SUCCESS(cuPointerSetAttribute(
      &flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, gpu_dma_buf.gpu_mem_ptr));

  // CHECK(gpu_dma_buf.gpu_mem_ptr % PAGE_SIZE == 0);
  gpu_dma_buf.gpu_mem_fd = get_gpumem_dmabuf_pages_fd(
      gpu_pci_addr_, nic_pci_addr_, create_page_pool_,
      gpu_dma_buf.gpu_mem_ptr, size, &gpu_dma_buf.dma_buf_fd);

  if (gpu_dma_buf.gpu_mem_fd < 0) {
    LOG(WARNING) << "get_gpumem_dmabuf_pages_fd() failed!";
    *success = false;
  }
  gpu_dma_buf.size = size;
  bytes_allocated_ += size;
  *success = true;
}

void DmabufGpuPageAllocator::FreePage(unsigned long id) {
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

CUdeviceptr DmabufGpuPageAllocator::GetGpuMem(unsigned long id) {
  if (gpu_dma_buf_map_.find(id) == gpu_dma_buf_map_.end()) return 0;
  return gpu_dma_buf_map_[id].gpu_mem_ptr;
}

int DmabufGpuPageAllocator::GetGpuMemFd(unsigned long id) {
  if (gpu_dma_buf_map_.find(id) == gpu_dma_buf_map_.end()) return -1;
  return gpu_dma_buf_map_[id].gpu_mem_fd;
}

DmabufGpuPageAllocator::~DmabufGpuPageAllocator() {
  for (auto &[id, _] : gpu_dma_buf_map_) {
    FreePage(id);
  }
}
}  // namespace tcpdirect
