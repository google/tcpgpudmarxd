#include <errno.h>
#include <fcntl.h>
#include <linux/types.h>
#include <net/if.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "cuda/cuda_context_manager.cuh"
#include "cuda/gpu_page_exporter_factory.cuh"
#include "include/flow_steer_ntuple.h"
#include "include/gpu_page_exporter_interface.h"
#include "include/gpu_rxq_configuration_factory.h"
#include "include/nic_configurator_factory.h"
#include "include/nic_configurator_interface.h"
#include "include/rx_rule_manager.h"
#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/strings/str_format.h>

using tcpdirect::FlowSteerNtuple;
using tcpdirect::GpuPageExporterFactory;
using tcpdirect::GpuPageExporterInterface;
using tcpdirect::GpuRxqConfiguration;
using tcpdirect::GpuRxqConfigurationFactory;
using tcpdirect::GpuRxqConfigurationList;
using tcpdirect::NicConfiguratorFactory;
using tcpdirect::NicConfiguratorInterface;
using tcpdirect::RxRuleManager;

ABSL_FLAG(std::string, gpu_nic_preset, "auto",
          "The preset configuration for GPU/NIC pairs.  Options: monstertruck, "
          "predvt, auto");
ABSL_FLAG(
    std::string, gpu_nic_topology, "",
    "The path to the textproto file that defines the gpu to nic topology");
ABSL_FLAG(std::string, gpu_nic_topology_proto, "",
          "The string of protobuf that defines the gpu to nic topology");
ABSL_FLAG(bool, show_version, false, "Show the version of this binary.");
ABSL_FLAG(std::string, gpu_shmem_type, "file",
          "The type of GPU memory sharing to use. options: [file, fd]");
ABSL_FLAG(std::string, uds_path, "/tmp",
          "The path to the filesystem folder where unix domain sockets will be "
          "bound to.");

#define RETURN_IF_ERROR(x)               \
  if (auto status = (x); !status.ok()) { \
    std::cerr << status << std::endl;    \
    return (int)status.code();           \
  }

namespace {

constexpr int kDefaultRssNum{16};
constexpr std::string_view kVersion{"0.0.7"};

static void get_cuda_error(const char *fn, CUresult err) {
  const char *name = "[unknown]", *explanation = "[unknown]";

  if (cuGetErrorName(err, &name))
    std::cerr << "Error: error getting error name" << std::endl;

  if (cuGetErrorString(err, &explanation))
    std::cerr << "Error: error getting error string" << std::endl;

  std::cerr << absl::StrFormat("CUDA Error in func %s: %d %s (%s)", fn, err,
                               name, explanation)
            << std::endl;
}

static std::atomic<bool> gShouldStop(false);

void sig_handler(int signum) {
  if (signum == SIGINT || signum == SIGTERM) {
    gShouldStop.store(true, std::memory_order_release);
  }
}

}  // namespace

int main(int argc, char **argv) {
  umask(0);
  absl::ParseCommandLine(argc, argv);
  bool show_version = absl::GetFlag(FLAGS_show_version);
  if (show_version) {
    std::cout << kVersion << std::endl;
    return 0;
  }

  // 1. Collect GPU/NIC pair configurations
  std::string gpu_nic_preset = absl::GetFlag(FLAGS_gpu_nic_preset);
  std::cout << "Collecting GPU/NIC pair configurations ..." << std::endl;
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

  if (CUresult err = cuInit(0); err != CUDA_SUCCESS) {
    get_cuda_error("cuInit", err);
    return 1;
  }

  std::string gpu_shmem_type = absl::GetFlag(FLAGS_gpu_shmem_type);
  std::unique_ptr<GpuPageExporterInterface> gpu_page_exporter =
      GpuPageExporterFactory::Build(gpu_shmem_type);

  if (!gpu_page_exporter) {
    std::cerr << "Failed to create gpu_page_exporter";
    return 1;
  }

  // 2. Start the Gpu-Rxq exporter
  std::string uds_path = absl::GetFlag(FLAGS_uds_path);
  RETURN_IF_ERROR(gpu_page_exporter->Initialize(gpu_rxq_configs, uds_path));
  RETURN_IF_ERROR(gpu_page_exporter->Export());

  // 3. Setup NIC configurator
  std::unique_ptr<NicConfiguratorInterface> nic_configurator =
      NicConfiguratorFactory::Build(gpu_nic_preset);

  RETURN_IF_ERROR(nic_configurator->Init());

  for (auto &gpu_rxq_config : gpu_rxq_configs.gpu_rxq_configs()) {
    RETURN_IF_ERROR(nic_configurator->ToggleHeaderSplit(gpu_rxq_config.ifname(),
                                                        /*enable=*/true));
    RETURN_IF_ERROR(nic_configurator->SetNtuple(gpu_rxq_config.ifname()));
  }

  // 4. Start Rx Rule Manager
  RxRuleManager rx_rule_manager(gpu_rxq_configs, uds_path,
                                nic_configurator.get());
  RETURN_IF_ERROR(rx_rule_manager.Init());

  // Setup global flags and handlers
  gShouldStop.store(false);
  signal(SIGINT, sig_handler);
  signal(SIGTERM, sig_handler);

  while (!gShouldStop.load()) {
    sleep(10);
  }

  // Stopping the servers.
  std::cout << "Program terminates, stopping the servers ..." << std::endl;
  gpu_page_exporter->Cleanup();
  rx_rule_manager.Cleanup();

  for (auto &gpu_rxq_config : gpu_rxq_configs.gpu_rxq_configs()) {
    RETURN_IF_ERROR(nic_configurator->ToggleHeaderSplit(gpu_rxq_config.ifname(),
                                                        /*enable=*/false));
  }
}
