#ifndef _CUDA_DMABUF_GPU_PAGE_ALLOCATOR_H_
#define _CUDA_DMABUF_GPU_PAGE_ALLOCATOR_H_

#include <cuda.h>

#include <string>
#include <unordered_map>

#include "cuda/gpu_page_allocator_interface.cuh"

namespace gpudirect_tcpxd {

struct GpuDmaBuf {
  int dma_buf_fd;
  int gpu_mem_fd;
  size_t size;
  CUdeviceptr gpu_mem_ptr;
};

// THIS IS NOT A THREAD-SAFE IMPLEMENTATION.
class CudaDmabufGpuPageAllocator : public GpuPageAllocatorInterface {
 public:
  CudaDmabufGpuPageAllocator(std::string gpu_pci_addr,
                             std::string nic_pci_addr);
  CudaDmabufGpuPageAllocator(std::string gpu_pci_addr, std::string nic_pci_addr,
                             bool create_page_pool, size_t pool_size);
  void AllocatePage(size_t size, unsigned long *id, bool *success) override;
  void FreePage(unsigned long id) override;
  CUdeviceptr GetGpuMem(unsigned long id) override;
  int GetGpuMemFd(unsigned long id) override;
  void Reset() override {}
  ~CudaDmabufGpuPageAllocator() override;

 private:
  size_t bytes_allocated_{0};
  std::string gpu_pci_addr_;
  std::string nic_pci_addr_;
  bool create_page_pool_{false};
  size_t pool_size_;
  std::unordered_map<unsigned long, GpuDmaBuf> gpu_dma_buf_map_;
  unsigned long next_id_{0};
  // disallow copy and assign
  CudaDmabufGpuPageAllocator(const CudaDmabufGpuPageAllocator &);
  void operator=(const CudaDmabufGpuPageAllocator &);
};
}  // namespace gpudirect_tcpxd
#endif
