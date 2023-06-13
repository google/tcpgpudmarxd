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
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "base/logging.h"
#include "experimental/users/chechenglin/tcpgpudmad/benchmark/benchmark_common.h"
#include "experimental/users/chechenglin/tcpgpudmad/benchmark/connection_worker.cu.h"
#include "experimental/users/chechenglin/tcpgpudmad/benchmark/event_handler_factory.cu.h"
#include "experimental/users/chechenglin/tcpgpudmad/benchmark/socket_helper.h"
#include "experimental/users/chechenglin/tcpgpudmad/cuda/common.cu.h"
#include "experimental/users/chechenglin/tcpgpudmad/cuda/cuda_context_manager.cu.h"
#include "experimental/users/chechenglin/tcpgpudmad/include/gpu_rxq_configuration_factory.h"
#include "third_party/absl/flags/flag.h"
#include "third_party/absl/flags/parse.h"
#include "third_party/absl/strings/str_format.h"
#include "third_party/gpus/cuda/include/cuda.h"

ABSL_FLAG(std::string, server_address, "", "server address");
ABSL_FLAG(uint16_t, port, 50000, "port number for server");
ABSL_FLAG(bool, server, false, "is server");
ABSL_FLAG(int, gpu_nic_index, 0, "which gpu/nic pair to test");
ABSL_FLAG(int, socket_per_thread, 1, "number of sockets per thread");
ABSL_FLAG(size_t, message_size, 8192, "message size");
ABSL_FLAG(std::string, event_handler, "", "Type of the socket event handler");
ABSL_FLAG(int, length, 10, "length of the test in seconds");
ABSL_FLAG(std::string, gpu_nic_preset, "auto",
          "The preset configuration for GPU/NIC pairs.  Options: monstertruck, "
          "predvt, auto");
ABSL_FLAG(
    std::string, gpu_nic_topology, "",
    "The path to the textproto file that defines the gpu to nic topology");
ABSL_FLAG(std::string, gpu_nic_topology_proto, "",
          "The string of protobuf that defines the gpu to nic topology");

namespace {
using namespace tcpdirect;
struct GlobalState {};

static const double kKiloRatio = 1000.0;
}  // namespace

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  CU_ASSERT_SUCCESS(cuInit(0));

  uint16_t control_port = absl::GetFlag(FLAGS_port);
  int gpu_nic_index = absl::GetFlag(FLAGS_gpu_nic_index);
  size_t message_size = absl::GetFlag(FLAGS_message_size);
  bool is_server = absl::GetFlag(FLAGS_server);
  std::vector<union SocketAddress> server_hpn_addresses;
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

  if (is_server) {
    const std::string& ifname =
        gpu_rxq_configs.gpu_rxq_configs(gpu_nic_index).ifname();
    server_hpn_addresses.emplace_back(GetIpv6FromIfName(ifname));
    AcceptConnectionAndSendHpnAddress(&server_addr, server_hpn_addresses);
  } else {
    ConnectAndReceiveHpnAddress(&server_addr, &server_hpn_addresses);
  }

  std::unique_ptr<ConnectionWorker> connection_worker;
  // 1. Collect hardware information (gpu address, cuda index, nic address,
  // nic ip, ...) for this worker
  int gpu_idx = gpu_nic_index;
  int per_gpu_thread_idx = 0;

  const std::string& gpu_pci =
      gpu_rxq_configs.gpu_rxq_configs(gpu_idx).gpu_pci_addr();
  const std::string& nic_pci =
      gpu_rxq_configs.gpu_rxq_configs(gpu_idx).nic_pci_addr();
  union SocketAddress nic_ip =
      GetIpv6FromIfName(gpu_rxq_configs.gpu_rxq_configs(gpu_idx).ifname());

  union SocketAddress& server_hpn_addr = server_hpn_addresses[0];
  uint16_t server_port = control_port + per_gpu_thread_idx;

  // 2. Create event handler factory and cuda context manager (if necessary)

  std::unique_ptr<EventHandlerFactoryInterface> ev_factory;
  std::unique_ptr<CudaContextManager> cuda_mgr;

  // 2.1 Set up CudaContextManager
  cuda_mgr.reset(new CudaContextManager(gpu_idx));
  // 2.2 Selects the event handler factory and configures it
  ev_factory =
      EventHandlerFactorySelector(event_handler_name, gpu_pci, nic_pci);

  // 3. Create connection worker
  ConnectionWorkerConfig worker_config;
  worker_config.do_validation = false;
  worker_config.num_sockets = 1;
  worker_config.message_size = message_size;
  worker_config.thread_id = {.gpu_idx = gpu_idx, .per_gpu_thread_idx = 0};
  worker_config.server_port = server_port;
  worker_config.server_address = server_hpn_addr;
  if (is_server) {
    worker_config.is_server = true;
  } else {
    worker_config.is_server = false;
    worker_config.client_address = nic_ip;
  }

  connection_worker.reset(new ConnectionWorker(
      worker_config, std::move(ev_factory), std::move(cuda_mgr)));

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
  for (const auto& error : errors) {
    LOG(INFO) << "Error: " << error;
  }

  LOG(INFO) << "Ending the test.";

  connection_worker->Stop();

  return 0;
}
