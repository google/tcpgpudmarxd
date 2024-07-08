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
#include "telemetry/gpu_mem_exporter_telemetry.h"

namespace gpudirect_tcpxd {

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
  GpuMemExporterTelemetry gpu_fd_telemetry_;
  GpuMemExporterTelemetry gpu_metadata_telemetry_;
};

}  // namespace gpudirect_tcpxd

#endif