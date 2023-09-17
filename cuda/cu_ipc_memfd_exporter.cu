#include <absl/log/check.h>
#include <absl/log/log.h>
#include <absl/status/status.h>

#include <memory>
#include <numeric>
#include <thread>
#include <vector>

#include "code.pb.h"
#include "cuda/cu_dmabuf_gpu_page_allocator.cuh"
#include "cuda/cu_ipc_memfd_exporter.cuh"
#include "cuda/cuda_context_manager.cuh"
#include "include/gpu_page_exporter_interface.h"
#include "include/ipc_gpumem_fd_metadata.h"
#include "include/unix_socket_server.h"
#include "proto/gpu_rxq_configuration.pb.h"
#include "proto/unix_socket_proto.pb.h"

namespace gpudirect_tcpxd {

absl::Status CuIpcMemfdExporter::Initialize(
    const GpuRxqConfigurationList& config_list, const std::string& prefix) {
  prefix_ = prefix;
  if (prefix_.back() == '/') {
    prefix_.pop_back();
  }

  gpu_fd_telemetry_.Start();
  gpu_metadata_telemetry_.Start();

  // Setup CUDA context and DmabufPageAllocator
  LOG(INFO) << "Setting up CUDA context and dmabuf page allocator ...";

  size_t rx_pool_size = RX_POOL_SIZE;
  if (config_list.has_rx_pool_size()) {
    rx_pool_size = config_list.rx_pool_size();
  }

  for (const auto& gpu_rxq_config : config_list.gpu_rxq_configs()) {
    std::string ifname = gpu_rxq_config.ifname();
    std::string nic_pci_addr = gpu_rxq_config.nic_pci_addr();
    for (const auto& gpu_info : gpu_rxq_config.gpu_infos()) {
      std::string gpu_pci_addr = gpu_info.gpu_pci_addr();
      int dev_id;

      CUDA_ASSERT_SUCCESS(
          cudaDeviceGetByPCIBusId(&dev_id, gpu_pci_addr.c_str()));

      gpu_pci_bindings_.emplace_back(GpuRxqBinding{
          .dev_id = dev_id,
          .gpu_pci_addr = gpu_pci_addr,
          .ifname = ifname,
          .page_allocator = std::make_unique<CuDmabufGpuPageAllocator>(
              dev_id, gpu_pci_addr, nic_pci_addr, rx_pool_size),
          .queue_ids = {gpu_info.queue_ids().begin(),
                        gpu_info.queue_ids().end()},
      });
    }
  }

  // 3. Allocate gpu memory, bind rxq, and get IpcGpuMemFdMetadata
  LOG(INFO)
      << "Allocating gpu memory, binding rxq, and getting cudaIpcMemHandle ...";

  std::vector<std::thread> alloc_threads;
  for (auto& gpu_rxq_binding : gpu_pci_bindings_) {
    const auto& gpu_pci_addr = gpu_rxq_binding.gpu_pci_addr;
    const auto& ifname = gpu_rxq_binding.ifname;
    const auto& dev_id = gpu_rxq_binding.dev_id;
    auto& page_allocator = *gpu_rxq_binding.page_allocator;
    auto& page_id = gpu_rxq_binding.page_id;
    auto& gpumem_fd_metadata = gpu_rxq_binding.gpumem_fd_metadata;
    auto& qids = gpu_rxq_binding.queue_ids;
    alloc_threads.emplace_back([&]() {
      CUDA_ASSERT_SUCCESS(cudaSetDevice(dev_id));
      bool allocation_success = false;
      page_allocator.AllocatePage(rx_pool_size, &page_id, &allocation_success);

      if (!allocation_success) {
        LOG(ERROR) << "Failed to allocate GPUMEM page: " << ifname;
        return;
      }

      for (int qid : qids) {
        if (int ret = gpumem_bind_rxq(page_allocator.GetGpuMemFd(page_id),
                                      ifname, qid);
            ret < 0) {
          LOG(ERROR) << "Failed to bind rxq: " << ifname;
          return;
        }
      }

      gpumem_fd_metadata = page_allocator.GetIpcGpuMemFdMetadata(page_id);
    });

    // Find memhandle by gpu pci
    us_servers_.emplace_back(std::make_unique<UnixSocketServer>(
        absl::StrFormat("%s/get_gpu_fd_%s", prefix_, gpu_pci_addr),
        /*service_handler=*/
        [&](UnixSocketMessage&& request, UnixSocketMessage* response,
            bool* fin) {
          absl::Time start = absl::Now();
          gpu_fd_telemetry_.IncrementRequests();
          if (request.has_proto() &&
              request.proto().raw_bytes() == gpu_pci_addr) {
            response->set_fd(gpumem_fd_metadata.fd);
            gpu_fd_telemetry_.IncrementIpcSuccess();
          } else {
            response->set_fd(-1);
            gpu_fd_telemetry_.IncrementIpcFailure();
            gpu_fd_telemetry_.IncrementIpcFailureAndCause("Fd Not Found");
          }
          gpu_fd_telemetry_.AddLatency(absl::Now() - start);
        },
        /*service_setup=*/
        [&]() { CUDA_ASSERT_SUCCESS(cudaSetDevice(dev_id)); }));

    us_servers_.emplace_back(std::make_unique<UnixSocketServer>(
        absl::StrFormat("%s/get_gpu_metadata_%s", prefix_, gpu_pci_addr),
        /*service_handler=*/
        [&](UnixSocketMessage&& request, UnixSocketMessage* response,
            bool* fin) {
          absl::Time start = absl::Now();
          gpu_metadata_telemetry_.IncrementRequests();
          UnixSocketProto* proto = response->mutable_proto();
          std::string* buffer = response->mutable_proto()->mutable_raw_bytes();
          if (request.has_proto() &&
              request.proto().raw_bytes() == gpu_pci_addr) {
            for (int i = 0; i < sizeof(gpumem_fd_metadata); ++i) {
              buffer->push_back(*((char*)&gpumem_fd_metadata + i));
            }
            gpu_metadata_telemetry_.IncrementIpcSuccess();
          } else {
            proto->mutable_status()->set_code(
                google::rpc::Code::INVALID_ARGUMENT);
            proto->mutable_status()->set_message(
                "Requested GPU PCI Addr not found.");
            *buffer = "Not found.";
            gpu_metadata_telemetry_.IncrementIpcFailure();
            gpu_metadata_telemetry_.IncrementIpcFailureAndCause(
                proto->mutable_status()->message());
          }
          gpu_metadata_telemetry_.AddLatency(absl::Now() - start);
        },
        /*service_setup=*/
        [&]() { CUDA_ASSERT_SUCCESS(cudaSetDevice(dev_id)); }));
  }
  for (auto& th : alloc_threads) {
    th.join();
  }
  return absl::OkStatus();
}

absl::Status CuIpcMemfdExporter::Export() {
  LOG(INFO) << "Starting Unix socket servers ...";

  for (auto& server : us_servers_) {
    if (auto server_status = server->Start(); !server_status.ok()) {
      return server_status;
    }
  }

  LOG(INFO) << "CuIpcMemFdHandle Unix socket servers started ...";
  return absl::OkStatus();
}
void CuIpcMemfdExporter::Cleanup() {
  for (auto& server : us_servers_) {
    server->Stop();
  }
  for (auto& gpu_rxq_binding : gpu_pci_bindings_) {
    gpu_rxq_binding.page_allocator->Cleanup();
  }
}
}  // namespace gpudirect_tcpxd