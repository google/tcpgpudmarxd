#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/log/check.h>
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
#include <cassert>
#include <chrono>
#include <cstddef>
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
ABSL_FLAG(uint16_t, port, 5201, "port number for server");
ABSL_FLAG(bool, server, false, "is server");
ABSL_FLAG(int, nic_index, 0, "which nic to test");
ABSL_FLAG(int, gpu_index, 0, "which gpu to test");
ABSL_FLAG(int, socket_per_thread, 1, "number of sockets per thread");
ABSL_FLAG(size_t, max_message_size, 1073741824, "message size. DEFAULT:1GB");
ABSL_FLAG(std::string, event_handler, "", "Type of the socket event handler");
ABSL_FLAG(int, length, 10, "length of the test in seconds");
ABSL_FLAG(std::string, gpu_nic_preset, "a3vm",
          "The preset configuration for GPU/NIC pairs.  Options: monstertruck, "
          "predvt, auto");
ABSL_FLAG(
    std::string, gpu_nic_topology, "",
    "The path to the textproto file that defines the gpu to nic topology");
ABSL_FLAG(std::string, gpu_nic_topology_proto, "",
          "The string of protobuf that defines the gpu to nic topology");

namespace {
using namespace gpudirect_tcpxd;
struct GlobalState {};

static const double kKiloRatio = 1000.0;
}  // namespace

void RunWorker(const ConnectionWorkerConfig &worker_config,
               std::unique_ptr<EventHandlerFactoryInterface> ev_factory,
               int count_down) {
  std::unique_ptr<ConnectionWorker> connection_worker;
  connection_worker.reset(
      new ConnectionWorker(worker_config, std::move(ev_factory)));

  LOG(INFO)
      << "Starting connection worker, sleep for 10 seconds in the main thread";

  std::atomic<bool> running{true};
  std::vector<std::string> errors;
  connection_worker->Start([&](std::vector<std::string> ev_errors) {
    running.store(false);
    errors = ev_errors;
  });

  auto prev_epoch = std::chrono::system_clock::now();

  for (int i = 0; i < count_down && running.load(); ++i) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    auto now = std::chrono::system_clock::now();
    std::chrono::duration<double> d = now - prev_epoch;
    double factor = 8.0 / kKiloRatio / kKiloRatio / kKiloRatio / d.count();

    double rx_gbps = connection_worker->GetRxBytes() * factor;
    double tx_gbps = connection_worker->GetTxBytes() * factor;

    LOG(INFO) << "total rx gbps: " << rx_gbps;
    LOG(INFO) << "total tx gbps: " << tx_gbps;
    prev_epoch = now;
  }

  // TODO(chechenglin): validate the error for the server
  for (const auto &error : errors) {
    LOG(INFO) << "Error: " << error;
  }

  LOG(INFO) << "Ending the test.";

  connection_worker->Stop();
}

int main(int argc, char *argv[]) {
  absl::ParseCommandLine(argc, argv);

  CU_ASSERT_SUCCESS(cuInit(0));

  uint16_t control_port = absl::GetFlag(FLAGS_port);
  int nic_index = absl::GetFlag(FLAGS_nic_index);
  int gpu_index = absl::GetFlag(FLAGS_gpu_index);
  CHECK((gpu_index / 2) == nic_index);
  size_t max_message_size = absl::GetFlag(FLAGS_max_message_size);
  bool is_server = absl::GetFlag(FLAGS_server);
  std::vector<union SocketAddress> server_hpn_addresses;
  std::vector<union SocketAddress> client_hpn_addresses;
  std::string event_handler_name = absl::GetFlag(FLAGS_event_handler);
  int count_down = absl::GetFlag(FLAGS_length);
  std::string gpu_nic_preset = absl::GetFlag(FLAGS_gpu_nic_preset);
  GpuRxqConfigurationList gpu_rxq_configs;
  if (gpu_nic_preset == "manual") {
    if (!absl::GetFlag(FLAGS_gpu_nic_topology_proto).empty() &&
        !absl::GetFlag(FLAGS_gpu_nic_topology).empty()) {
      LOG(FATAL) << "Can't set both gpu_nic_topology_proto and "
                    "gpu_nic_topology at the same time.";
    }
    if (!absl::GetFlag(FLAGS_gpu_nic_topology_proto).empty()) {
      gpu_rxq_configs = GpuRxqConfigurationFactory::FromCmdLine(
          absl::GetFlag(FLAGS_gpu_nic_topology_proto));
    } else if (!absl::GetFlag(FLAGS_gpu_nic_topology).empty()) {
      gpu_rxq_configs = GpuRxqConfigurationFactory::FromFile(
          absl::GetFlag(FLAGS_gpu_nic_topology));
    } else {
      LOG(FATAL)
          << "Can't select manual option because both gpu_nic_topology_proto "
             "and gpu_nic_topology are empty.";
    }
  } else {
    gpu_rxq_configs = GpuRxqConfigurationFactory::BuildPreset(gpu_nic_preset);
  }

  CHECK(!event_handler_name.empty());

  // Exchange Server HPN addresses
  union SocketAddress server_addr =
      AddressFromStr(absl::GetFlag(FLAGS_server_address));
  SetAddressPort(&server_addr, control_port);

  std::vector<NetifInfo> nic_infos;

  DiscoverNetif(nic_infos);

  std::unordered_map<std::string, union SocketAddress> ifname_sockaddr_map;

  for (auto &nic_info : nic_infos) {
    ifname_sockaddr_map[nic_info.ifname] = nic_info.addr;
  }

  const std::string &hpn_ifname =
      gpu_rxq_configs.gpu_rxq_configs(nic_index).ifname();

  if (is_server) {
    LOG(INFO) << "[Server] Starting Server Control Channel ...";
    server_hpn_addresses.emplace_back(ifname_sockaddr_map[hpn_ifname]);
    ServerAcceptControlChannelConnection(&server_addr, &client_hpn_addresses,
                                         server_hpn_addresses);
  } else {
    LOG(INFO) << "[Client] Connecting Control Channel ...";
    client_hpn_addresses.emplace_back(ifname_sockaddr_map[hpn_ifname]);
    ClientConnectControlChannel(&server_addr, &server_hpn_addresses,
                                client_hpn_addresses);
  }

  LOG(INFO) << "Handshake done.";

  LOG(INFO) << " -- Server HPN Addresses -- ";

  for (int i = 0; i < server_hpn_addresses.size(); ++i) {
    const auto &hpn_addr = server_hpn_addresses[i];
    LOG(INFO) << absl::StrFormat("hpn[%d]: %s", i + 1, AddressToStr(&hpn_addr));
  }

  LOG(INFO) << " -- Client HPN Addresses -- ";

  for (int i = 0; i < client_hpn_addresses.size(); ++i) {
    const auto &hpn_addr = client_hpn_addresses[i];
    LOG(INFO) << absl::StrFormat("hpn[%d]: %s", i + 1, AddressToStr(&hpn_addr));
  }

  // 1. Collect hardware information (gpu address, cuda index, nic address,
  // nic ip, ...) for this worker
  int gpu_idx = gpu_index;
  int per_gpu_thread_idx = 0;

  const std::string &gpu_pci = gpu_rxq_configs.gpu_rxq_configs(gpu_idx)
                                   .gpu_infos(gpu_idx)
                                   .gpu_pci_addr();
  const std::string &nic_pci =
      gpu_rxq_configs.gpu_rxq_configs(gpu_idx).nic_pci_addr();
  union SocketAddress nic_ip =
      ifname_sockaddr_map[gpu_rxq_configs.gpu_rxq_configs(gpu_idx).ifname()];

  uint16_t server_port = control_port + per_gpu_thread_idx;

  for (size_t message_size = 1; message_size <= max_message_size;
       message_size *= 2) {
    LOG(INFO) << "The current message size is " << message_size << " bytes";
    // 2. Create event handler factory
    std::unique_ptr<EventHandlerFactoryInterface> ev_factory =
        EventHandlerFactorySelector(event_handler_name, gpu_pci, nic_pci);

    // 3. Create connection worker
    ConnectionWorkerConfig worker_config;
    worker_config.do_validation = false;
    worker_config.num_sockets = 1;
    worker_config.message_size = message_size;
    worker_config.thread_id = {.gpu_idx = gpu_idx, .per_gpu_thread_idx = 0};
    worker_config.server_port = server_port;
    worker_config.server_address = server_hpn_addresses[0];
    worker_config.client_address = client_hpn_addresses[0];
    worker_config.gpu_pci_addr = gpu_pci;

    if (is_server) {
      worker_config.is_server = true;
    } else {
      worker_config.is_server = false;
      worker_config.client_address = nic_ip;
    }

    RunWorker(worker_config, std::move(ev_factory), count_down);
    std::this_thread::sleep_for(std::chrono::seconds(5));
  }

  return 0;
}
