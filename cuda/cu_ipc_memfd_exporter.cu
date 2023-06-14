#include "cuda/cu_ipc_memfd_exporter.cuh"

#include <memory>
#include <thread>
#include <vector>

#include <absl/log/log.h>
#include "cuda/cu_dmabuf_gpu_page_allocator.cuh"
#include "cuda/cuda_context_manager.cuh"
#include "include/gpu_page_exporter_interface.h"
#include "include/ipc_gpumem_fd_metadata.h"
#include "include/unix_socket_server.h"
#include "proto/gpu_rxq_configuration.pb.h"
#include <absl/status/status.h>

namespace tcpdirect {

absl::Status CuIpcMemfdExporter::Initialize(
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
    int dev_id;
    CUDA_ASSERT_SUCCESS(cudaDeviceGetByPCIBusId(&dev_id, gpu_pci_addr.c_str()))
    ifname_binding_map_[ifname] = {
        .dev_id = dev_id,
        .config = gpu_rxq_config,
        .page_allocator = std::make_unique<CuDmabufGpuPageAllocator>(
            dev_id, gpu_pci_addr, nic_pci_addr, /*pool_size=*/RX_POOL_SIZE),
    };
    gpu_pci_to_ifname_map_[gpu_pci_addr] = ifname;
  }

  // 3. Allocate gpu memory, bind rxq, and get IpcGpuMemFdMetadata
  LOG(INFO)
      << "Allocating gpu memory, binding rxq, and getting cudaIpcMemHandle ...";

  std::vector<std::thread> alloc_threads;
  for (auto &[_, gpu_rxq_binding] : ifname_binding_map_) {
    auto &config = gpu_rxq_binding.config;
    auto &dev_id = gpu_rxq_binding.dev_id;
    auto &page_allocator = *gpu_rxq_binding.page_allocator;
    auto &page_id = gpu_rxq_binding.page_id;
    auto &gpumem_fd_metadata = gpu_rxq_binding.gpumem_fd_metadata;
    alloc_threads.emplace_back([&]() {
      CUDA_ASSERT_SUCCESS(cudaSetDevice(dev_id));
      bool allocation_success = false;
      page_allocator.AllocatePage(RX_POOL_SIZE, &page_id, &allocation_success);

      if (!allocation_success) {
        LOG(ERROR) << "Failed to allocate GPUMEM page: " << config.ifname();
        return;
      }

      for (int qid = tcpd_qstart; qid < tcpd_qend; ++qid) {
        if (int ret = gpumem_bind_rxq(page_allocator.GetGpuMemFd(page_id),
                                      config.ifname(), qid);
            ret < 0) {
          LOG(ERROR) << "Failed to bind rxq: " << config.ifname();
          return;
        }
      }

      gpumem_fd_metadata = page_allocator.GetIpcGpuMemFdMetadata(page_id);
    });

    // Find memhandle by gpu pci
    us_servers_.emplace_back(std::make_unique<UnixSocketServer>(
        absl::StrFormat("%s/get_gpu_fd_%s", prefix_, config.gpu_pci_addr()),
        /*service_handler=*/
        [&](UnixSocketMessage &&request, UnixSocketMessage *response,
            bool *fin) {
          if (request.has_proto() &&
              request.proto().raw_bytes() == config.gpu_pci_addr()) {
            response->set_fd(gpumem_fd_metadata.fd);
          } else {
            response->set_fd(-1);
          }
        },
        /*service_setup=*/
        [&]() { CUDA_ASSERT_SUCCESS(cudaSetDevice(dev_id)); }));

    us_servers_.emplace_back(std::make_unique<UnixSocketServer>(
        absl::StrFormat("%s/get_gpu_metadata_%s", prefix_,
                        config.gpu_pci_addr()),
        /*service_handler=*/
        [&](UnixSocketMessage &&request, UnixSocketMessage *response,
            bool *fin) {
          std::string *buffer = response->mutable_proto()->mutable_raw_bytes();
          if (request.has_proto() &&
              request.proto().raw_bytes() == config.gpu_pci_addr()) {
            for (int i = 0; i < sizeof(gpumem_fd_metadata); ++i) {
              buffer->push_back(*((char *)&gpumem_fd_metadata + i));
            }
          } else {
            *buffer = "Not found.";
          }
        },
        /*service_setup=*/
        [&]() { CUDA_ASSERT_SUCCESS(cudaSetDevice(dev_id)); }));
  }
  for (auto &th : alloc_threads) {
    th.join();
  }
  return absl::OkStatus();
}

absl::Status CuIpcMemfdExporter::Export() {
  LOG(INFO) << "Starting Unix socket servers ...";

  for (auto &server : us_servers_) {
    if (auto server_status = server->Start(); !server_status.ok()) {
      return server_status;
    }
  }

  LOG(INFO) << "CuIpcMemFdHandle Unix socket servers started ...";
  return absl::OkStatus();
}
void CuIpcMemfdExporter::Cleanup() {
  for (auto &server : us_servers_) {
    server->Stop();
  }
}
}  // namespace tcpdirect
