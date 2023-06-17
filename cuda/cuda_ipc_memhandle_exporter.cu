#include <absl/log/check.h>
#include <absl/log/log.h>
#include <absl/status/status.h>

#include <memory>
#include <numeric>
#include <vector>

#include "cuda/cuda_context_manager.cuh"
#include "cuda/cuda_ipc_memhandle_exporter.cuh"
#include "cuda/dmabuf_gpu_page_allocator.cuh"
#include "include/unix_socket_server.h"
#include "proto/gpu_rxq_configuration.pb.h"
#include "proto/unix_socket_message.pb.h"

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

  for (const auto &gpu_rxq_config : config_list.gpu_rxq_configs()) {
    std::string ifname = gpu_rxq_config.ifname();
    std::string nic_pci_addr = gpu_rxq_config.nic_pci_addr();
    for (const auto &gpu_info : gpu_rxq_config.gpu_infos()) {
      std::string gpu_pci_addr = gpu_info.gpu_pci_addr();
      gpu_pci_binding_map_[gpu_info.gpu_pci_addr()] = {
          .cuda_ctx = std::make_unique<CudaContextManager>(gpu_pci_addr),
          .page_allocator = std::make_unique<DmabufGpuPageAllocator>(
              gpu_pci_addr, nic_pci_addr, /*create_page_pool=*/true,
              /*pool_size=*/RX_POOL_SIZE),
          .ifname = ifname,
          .queue_ids = {gpu_info.queue_ids().begin(),
                        gpu_info.queue_ids().end()},
      };
    }
  }

  // 3. Allocate gpu memory, bind rxq, and get cudaIpcMemHandle
  LOG(INFO)
      << "Allocating gpu memory, binding rxq, and getting cudaIpcMemHandle ...";

  for (auto &[gpu_pci, gpu_rxq_binding] : gpu_pci_binding_map_) {
    auto &cuda_ctx = *gpu_rxq_binding.cuda_ctx;
    auto &page_allocator = *gpu_rxq_binding.page_allocator;
    auto &page_id = gpu_rxq_binding.page_id;
    auto &mem_handle = gpu_rxq_binding.mem_handle;
    auto &ifname = gpu_rxq_binding.ifname;
    auto &qids = gpu_rxq_binding.queue_ids;
    cuda_ctx.PushContext();
    bool allocation_success = false;
    page_allocator.AllocatePage(RX_POOL_SIZE, &page_id, &allocation_success);

    if (!allocation_success) {
      return absl::UnavailableError("Failed to allocate GPUMEM page: " +
                                    ifname);
    }

    for (int qid : qids) {
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
        GpuRxqBinding &binding = gpu_pci_binding_map_[gpu_pci];
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
