#ifndef _EXPERIMENTAL_USERS_CHECHENGLIN_CUDA_IPC_MEMHANDLE_IMPORTER_H_
#define _EXPERIMENTAL_USERS_CHECHENGLIN_CUDA_IPC_MEMHANDLE_IMPORTER_H_

#include <absl/status/statusor.h>

#include <string>

#include "cuda/cuda_ipc_memhandle.cuh"

namespace gpudirect_tcpxd {

class CudaIpcMemhandleImporter {
 public:
  static absl::StatusOr<std::unique_ptr<GpuPageHandleInterface>> Import(
      const std::string& prefix, const std::string& gpu_pci_addr);
};

}  // namespace gpudirect_tcpxd

#endif
