#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_GPU_PAGE_ALLOCATOR_INTERFACE_H_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_GPU_PAGE_ALLOCATOR_INTERFACE_H_

#include <cuda.h>

namespace tcpdirect {
class GpuPageAllocatorInterface {
 public:
  virtual void AllocatePage(size_t pool_size, unsigned long *id,
                            bool *success) = 0;
  virtual void FreePage(unsigned long id) = 0;
  virtual CUdeviceptr GetGpuMem(unsigned long id) = 0;
  virtual int GetGpuMemFd(unsigned long id) = 0;
  virtual void Reset() = 0;
  virtual ~GpuPageAllocatorInterface() {}
};
}  // namespace tcpdirect

#endif
