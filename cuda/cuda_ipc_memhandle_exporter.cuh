#ifndef _EXPERIMENTAL_USERS_CHECHENGLIN_CUDA_IPC_MEMHANDLE_EXPORTER_H_
#define _EXPERIMENTAL_USERS_CHECHENGLIN_CUDA_IPC_MEMHANDLE_EXPORTER_H_

#include <absl/status/status.h>

#include <memory>
#include <vector>

#include "cuda/cuda_context_manager.cuh"
#include "cuda/dmabuf_gpu_page_allocator.cuh"
#include "include/gpu_page_exporter_interface.h"
#include "include/unix_socket_server.h"
#include "proto/gpu_rxq_configuration.pb.h"

namespace gpudirect_tcpxd {

class CudaIpcMemhandleExporter : public GpuPageExporterInterface {
 public:
  struct GpuRxqBinding {
    std::unique_ptr<CudaContextManager> cuda_ctx;
    std::unique_ptr<DmabufGpuPageAllocator> page_allocator;
    std::string ifname;
    std::vector<int> queue_ids;
    unsigned long page_id;
    cudaIpcMemHandle_t mem_handle;
  };
  CudaIpcMemhandleExporter() = default;
  ~CudaIpcMemhandleExporter() { Cleanup(); }
  absl::Status Initialize(const GpuRxqConfigurationList& config_list,
                          const std::string& prefix) override;
  absl::Status Export() override;
  void Cleanup() override;

 private:
  std::string prefix_;
  std::unordered_map<std::string, GpuRxqBinding> gpu_pci_binding_map_;
  std::vector<std::unique_ptr<UnixSocketServer>> us_servers_;
};

}  // namespace gpudirect_tcpxd

#endif
