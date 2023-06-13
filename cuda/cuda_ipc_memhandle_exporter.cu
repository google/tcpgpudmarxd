#include "cuda/cuda_ipc_memhandle_exporter.cu.h"

#include <memory>
#include <vector>

#include "base/logging.h"
#include "cuda/cuda_context_manager.cu.h"
#include "cuda/dmabuf_gpu_page_allocator.cu.h"
#include "include/unix_socket_server.h"
#include "proto/gpu_rxq_configuration.pb.h"
#include "proto/unix_socket_message.pb.h"
#include <absl/status/status.h>

namespace tcpdirect {

absl::Status CudaIpcMemhandleExporter::Initialize(
    const GpuRxqConfigurationList &config_list, const std::string &prefix) {
  prefix_ = prefix;
  if (prefix_.back() == '/') {
    prefix_.pop_back();
  }
  // Setup CUDA context and DmabufPageAllocator
  LOG(INFO) << "Setting up CUDA context and dmabuf page allocator ...";

  int tcpd_qstart = config_list.rss_set_size();
  int tcpd_qend = tcpd_qstart + config_list.tcpd_queue_size();

  for (const auto &gpu_rxq_config : config_list.gpu_rxq_configs()) {
    std::string ifname = gpu_rxq_config.ifname();
    std::string gpu_pci_addr = gpu_rxq_config.gpu_pci_addr();
    std::string nic_pci_addr = gpu_rxq_config.nic_pci_addr();
    ifname_binding_map_[ifname] = {
        .config = gpu_rxq_config,
        .cuda_ctx = std::make_unique<CudaContextManager>(gpu_pci_addr),
        .page_allocator = std::make_unique<DmabufGpuPageAllocator>(
            gpu_pci_addr, nic_pci_addr, /*create_page_pool=*/true,
            /*pool_size=*/RX_POOL_SIZE),
    };
    gpu_pci_to_ifname_map_[gpu_pci_addr] = ifname;
  }

  // 3. Allocate gpu memory, bind rxq, and get cudaIpcMemHandle
  LOG(INFO)
      << "Allocating gpu memory, binding rxq, and getting cudaIpcMemHandle ...";

  for (auto &[ifname, gpu_rxq_binding] : ifname_binding_map_) {
    auto &cuda_ctx = *gpu_rxq_binding.cuda_ctx;
    auto &page_allocator = *gpu_rxq_binding.page_allocator;
    auto &page_id = gpu_rxq_binding.page_id;
    auto &mem_handle = gpu_rxq_binding.mem_handle;
    cuda_ctx.PushContext();
    bool allocation_success = false;
    page_allocator.AllocatePage(RX_POOL_SIZE, &page_id, &allocation_success);

    if (!allocation_success) {
      return absl::UnavailableError("Failed to allocate GPUMEM page: " +
                                    ifname);
    }

    for (int qid = tcpd_qstart; qid < tcpd_qend; ++qid) {
      if (int ret =
              gpumem_bind_rxq(page_allocator.GetGpuMemFd(page_id), ifname, qid);
          ret < 0) {
        return absl::UnavailableError("Failed to bind rxq: " + ifname);
      }
    }

    if (auto err = cudaIpcGetMemHandle(
            &mem_handle, (void *)page_allocator.GetGpuMem(page_id));
        err != 0) {
      return absl::UnavailableError("Failed to get cudaIpcMemHandle: " +
                                    ifname);
    }

    cuda_ctx.PopContext();
  }
  return absl::OkStatus();
}

absl::Status CudaIpcMemhandleExporter::Export() {
  LOG(INFO) << "Starting Unix socket servers ...";
  // Find memhandle by gpu pci
  us_servers_.emplace_back(std::make_unique<UnixSocketServer>(
      absl::StrFormat("%s/get_gpu_by_gpu_pci", prefix_),
      [this](UnixSocketMessage &&request, UnixSocketMessage *response,
             bool *fin) {
        UnixSocketProto *mutable_proto = response->mutable_proto();
        std::string *buffer = mutable_proto->mutable_raw_bytes();
        if (!request.has_proto() || !request.proto().has_raw_bytes()) {
          buffer->append("Error.\n\nExpecting text format request.\n");
          *fin = true;
          return;
        }
        const std::string &gpu_pci = request.proto().raw_bytes();
        if (gpu_pci_to_ifname_map_.find(gpu_pci) ==
            gpu_pci_to_ifname_map_.end()) {
          buffer->append(
              absl::StrFormat("ifname not found for gpu pci: %s\n", gpu_pci));
          *fin = true;
          return;
        }
        const std::string &ifname = gpu_pci_to_ifname_map_[gpu_pci];
        if (ifname_binding_map_.find(ifname) == ifname_binding_map_.end()) {
          buffer->append(
              absl::StrFormat("memhandle not found for %s\n", gpu_pci));
          *fin = true;
          return;
        }
        GpuRxqBinding &binding = ifname_binding_map_[ifname];
        for (int i = 0; i < sizeof(binding.mem_handle); ++i) {
          buffer->push_back(*((char *)&binding.mem_handle + i));
        }
      }));

  for (auto &server : us_servers_) {
    if (auto server_status = server->Start(); !server_status.ok()) {
      return server_status;
    }
  }

  LOG(INFO) << "CudaIpcMemHandle Unix socket servers started ...";
  return absl::OkStatus();
}
void CudaIpcMemhandleExporter::Cleanup() {
  for (auto &server : us_servers_) {
    server->Stop();
  }
}
}  // namespace tcpdirect
