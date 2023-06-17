#ifndef _EXPERIMENTAL_USERS_CHECHENGLIN_CU_IPC_MEMFD_EXPORTER_H_
#define _EXPERIMENTAL_USERS_CHECHENGLIN_CU_IPC_MEMFD_EXPORTER_H_

#include <absl/status/status.h>

#include <memory>
#include <vector>

#include "cuda/cu_dmabuf_gpu_page_allocator.cuh"
#include "cuda/cuda_context_manager.cuh"
#include "include/gpu_page_exporter_interface.h"
#include "include/ipc_gpumem_fd_metadata.h"
#include "include/unix_socket_server.h"
#include "proto/gpu_rxq_configuration.pb.h"

namespace tcpdirect {

class CuIpcMemfdExporter : public GpuPageExporterInterface {
 public:
  struct GpuRxqBinding {
    int dev_id;
    std::string gpu_pci_addr;
    std::string ifname;
    std::unique_ptr<CuDmabufGpuPageAllocator> page_allocator;
    std::vector<int> queue_ids;
    unsigned long page_id;
    IpcGpuMemFdMetadata gpumem_fd_metadata;
  };
  CuIpcMemfdExporter() = default;
  ~CuIpcMemfdExporter() { Cleanup(); }
  absl::Status Initialize(const GpuRxqConfigurationList& config_list,
                          const std::string& prefix) override;
  absl::Status Export() override;
  void Cleanup() override;

 private:
  std::string prefix_;
  std::vector<GpuRxqBinding> gpu_pci_bindings_;
  std::vector<std::unique_ptr<UnixSocketServer>> us_servers_;
};

}  // namespace tcpdirect

#endif
