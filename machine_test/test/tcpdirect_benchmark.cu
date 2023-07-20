/*
 * TODO(chechenglin): this whole thing is not ported yet.  We already have a
 * stable multnic_benchmark in use now so I'll leave this as a TODO for now.

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/log/log.h>
#include <absl/strings/str_format.h>
#include <cuda.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/errqueue.h>
#include <linux/types.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <sys/epoll.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/types.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cuda/common.cuh"
#include "cuda/cuda_context_manager.cuh"
#include "include/gpu_rxq_configuration_factory.h"
#include "include/socket_helper.h"
#include "machine_test/cuda/connection_worker.cuh"
#include "machine_test/cuda/event_handler_factory.cuh"
#include "machine_test/include/benchmark_common.h"

ABSL_FLAG(std::string, server_address, "", "server address");
ABSL_FLAG(uint16_t, port, 50000, "port number for server");
ABSL_FLAG(bool, server, false, "is server");
ABSL_FLAG(bool, gpudirect_tcpxd_rx, true, "use TCPDirect Rx");
ABSL_FLAG(bool, gpudirect_tcpxd_tx, true, "use TCPDirect Tx");
ABSL_FLAG(int, num_gpus, 1, "number of GPUs to use");
ABSL_FLAG(int, threads_per_gpu, 1,
          "number of threads per GPU (half of them Rx and the other half Tx)");
ABSL_FLAG(int, socket_per_thread, 1, "number of sockets per thread");
ABSL_FLAG(size_t, message_size, 8192, "message size");
ABSL_FLAG(bool, do_validation, false, "perform validation on received data");
ABSL_FLAG(bool, use_dmabuf, true, "use dmabuf API");
ABSL_FLAG(std::string, gpu_nic_preset, "auto",
          "The preset configuration for GPU/NIC pairs.  Options: monstertruck, "
          "predvt, auto");
ABSL_FLAG(
    std::string, gpu_nic_topology, "",
    "The path to the textproto file that defines the gpu to nic topology");

namespace {
using namespace gpudirect_tcpxd;
struct GlobalState {};

static const double kKiloRatio = 1000.0;

static std::atomic<bool> gShouldStop(false);

void sig_handler(int signum) {
  if (signum == SIGINT) {
    gShouldStop.store(true, std::memory_order_release);
  }
}

void StatsWorker(
    std::vector<std::unique_ptr<ConnectionWorker>>* connection_workers,
    int num_gpus) {
  auto prev_epoch = std::chrono::system_clock::now();

  while (!gShouldStop.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    auto now = std::chrono::system_clock::now();
    std::chrono::duration<double> d = now - prev_epoch;
    double total_rx_gbps = 0.0;
    double total_tx_gbps = 0.0;
    double factor = 8.0 / kKiloRatio / kKiloRatio / kKiloRatio / d.count();
    std::vector<double> gpu_rx_gbps(num_gpus);
    std::vector<double> gpu_tx_gbps(num_gpus);

    for (auto& worker : *connection_workers) {
      double rx_gbps = worker->GetRxBytes() * factor;
      double tx_gbps = worker->GetTxBytes() * factor;
      gpu_rx_gbps[worker->GpuIdx()] += rx_gbps;
      gpu_tx_gbps[worker->GpuIdx()] += tx_gbps;
      total_rx_gbps += rx_gbps;
      total_tx_gbps += tx_gbps;
    }

    for (int i = 0; i < num_gpus; ++i) {
      LOG(INFO) << "GPU " << i << " rx gbps: " << gpu_rx_gbps[i];
      LOG(INFO) << "GPU " << i << " tx gbps: " << gpu_tx_gbps[i];
    }
    LOG(INFO) << "total rx gbps: " << total_rx_gbps;
    LOG(INFO) << "total tx gbps: " << total_tx_gbps;
    prev_epoch = now;
  }
  for (auto& worker : *connection_workers) {
    worker->Stop();
  }
}
}  // namespace
int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  CU_ASSERT_SUCCESS(cuInit(0));

  PciAddrToGpuIdxMap pci_addr_to_gpu_idx;
  NicPciAddrToIpMap nic_pci_addr_to_ip_addr;

  uint16_t control_port = absl::GetFlag(FLAGS_port);
  int num_gpus = absl::GetFlag(FLAGS_num_gpus);
  int threads_per_gpu = absl::GetFlag(FLAGS_threads_per_gpu);
  int socket_per_thread = absl::GetFlag(FLAGS_socket_per_thread);
  size_t message_size = absl::GetFlag(FLAGS_message_size);
  bool is_server = absl::GetFlag(FLAGS_server);
  bool use_gpudirect_tcpxd_rx = absl::GetFlag(FLAGS_gpudirect_tcpxd_rx);
  bool use_gpudirect_tcpxd_tx = absl::GetFlag(FLAGS_gpudirect_tcpxd_tx);
  bool do_validation = absl::GetFlag(FLAGS_do_validation);
  bool use_dmabuf = absl::GetFlag(FLAGS_use_dmabuf);
  std::vector<union SocketAddress> server_hpn_addresses;
  std::string gpu_nic_preset = absl::GetFlag(FLAGS_gpu_nic_preset);
  GpuRxqConfigurationList gpu_rxq_configs;
  if (gpu_nic_preset == "manual") {
    gpu_rxq_configs = GpuRxqConfigurationFactory::FromFile(
        absl::GetFlag(FLAGS_gpu_nic_topology));
  } else {
    gpu_rxq_configs = GpuRxqConfigurationFactory::BuildPreset(gpu_nic_preset);
  }

  CHECK(threads_per_gpu % 2 == 0);
  CHECK(absl::GetFlag(FLAGS_server_address) != "");

  // Exchange Server HPN addresses
  union SocketAddress server_addr =
      AddressFromStr(absl::GetFlag(FLAGS_server_address));
  SetAddressPort(&server_addr, control_port);

  if (is_server) {
    for (int i = 0; i < num_gpus; i++) {
      const std::string& ifname = gpu_rxq_configs.gpu_rxq_configs(i).ifname();
      server_hpn_addresses.emplace_back(GetIpv6FromIfName(ifname));
    }
    AcceptConnectionAndSendHpnAddress(&server_addr, server_hpn_addresses);
  } else {
    ConnectAndReceiveHpnAddress(&server_addr, &server_hpn_addresses);
  }

  // Setup global flags and handlers
  gShouldStop.store(false);
  signal(SIGINT, sig_handler);

  int total_n_thread = threads_per_gpu * num_gpus;

  std::vector<std::unique_ptr<ConnectionWorker>> connection_workers(
      total_n_thread);
  for (int thread_id = 0; thread_id < total_n_thread; thread_id++) {
    // 1. Collect hardware information (gpu address, cuda index, nic address,
    // nic ip, ...) for this worker
    int gpu_idx = thread_id / threads_per_gpu;
    int per_gpu_thread_idx = thread_id % threads_per_gpu;

    const std::string& gpu_pci =
        gpu_rxq_configs.gpu_rxq_configs(gpu_idx).gpu_pci_addr();
    const std::string& nic_pci =
        gpu_rxq_configs.gpu_rxq_configs(gpu_idx).nic_pci_addr();
    union SocketAddress nic_ip =
        GetIpv6FromIfName(gpu_rxq_configs.gpu_rxq_configs(gpu_idx).ifname());

    union SocketAddress server_hpn_addr = server_hpn_addresses[gpu_idx];
    uint16_t server_port = control_port + per_gpu_thread_idx;

    // 2. Create event handler factory and cuda context manager (if necessary)

    std::unique_ptr<EventHandlerFactoryInterface> ev_factory;
    std::unique_ptr<CudaContextManager> cuda_mgr;

    // 2.1 Decides which traffic direrection this worker is responsible for
    int server_client_bit = is_server ? 1 : 0;
    int thread_idx_bit = per_gpu_thread_idx & 1;
    TrafficDirection direction =
        (server_client_bit ^ thread_idx_bit) ? TCP_SENDER : TCP_RECEIVER;
    // 2.2 Set up CudaCOntextManager if this worker uses TcpDirect
    // sender/receiver
    if ((direction == TCP_SENDER && use_gpudirect_tcpxd_tx) ||
        (direction == TCP_RECEIVER && use_gpudirect_tcpxd_rx)) {
      cuda_mgr.reset(new CudaContextManager(gpu_pci));
    }
    // 2.3 Selects the event handler factory and configures it
    switch (direction) {
      case TCP_SENDER:
        if (use_gpudirect_tcpxd_tx && use_dmabuf) {
          // dma buf
          ev_factory.reset(new DmabufSendEventHandlerFactory(gpu_pci, nic_pci));
        } else if (use_gpudirect_tcpxd_tx) {
          // p2p dma
          LOG(FATAL) << "P2PDMA page allocator is no longer supported.";
          std::abort();
        } else {
          // tcp zero copy tx
          ev_factory.reset(new TcpSendEventHandlerFactory());
        }
        break;
      case TCP_RECEIVER:
        if (use_gpudirect_tcpxd_rx) {
          ev_factory.reset(new GpuReceiveEventHandlerFactory(gpu_pci));
        } else {
          ev_factory.reset(new TcpReceiveEventHandlerFactory());
        }
        break;
      default:
        LOG(FATAL) << "Unknown direction";
    }

    // 3. Create connection worker
    ConnectionWorkerConfig worker_config;
    worker_config.do_validation = do_validation;
    worker_config.num_sockets = socket_per_thread;
    worker_config.message_size = message_size;
    worker_config.thread_id = {.gpu_idx = gpu_idx,
                               .per_gpu_thread_idx = per_gpu_thread_idx};
    worker_config.server_port = server_port;
    worker_config.server_address = server_hpn_addr;
    if (is_server) {
      worker_config.is_server = true;
    } else {
      worker_config.is_server = false;
      worker_config.client_address = nic_ip;
    }

    connection_workers[thread_id].reset(new ConnectionWorker(
        worker_config, std::move(ev_factory), std::move(cuda_mgr)));
  }

  LOG(INFO) << "Starting worker threads...";

  for (auto& connection_worker : connection_workers) {
    connection_worker->Start([&](std::vector<std::string> errors) {
      gShouldStop.store(true, std::memory_order_release);
      for (const auto& err : errors) {
        LOG(ERROR) << err;
      }
    });
  }

  std::thread th(StatsWorker, &connection_workers, num_gpus);

  if (th.joinable()) {
    th.join();
  }

  return 0;
}
*/
