#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/log/check.h>
#include <absl/log/log.h>
#include <absl/strings/str_format.h>
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

#include "cuda/common.cuh"
#include "cuda/gpu_page_exporter_factory.cuh"
#include "include/gpu_page_exporter_interface.h"
#include "include/gpu_rxq_configuration_factory.h"
#include "include/nic_configurator_factory.h"
#include "include/nic_configurator_interface.h"
#include "include/rx_rule_manager.h"

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

namespace {

constexpr std::string_view kVersion{"1.1.0"};

static std::atomic<bool> gShouldStop(false);

void sig_handler(int signum) {
  if (signum == SIGINT || signum == SIGTERM) {
    gShouldStop.store(true, std::memory_order_release);
  }
}

}  // namespace

int main(int argc, char **argv) {
  int ret_code = 0;
#define RETURN_IF_ERROR(x)               \
  if (auto status = (x); !status.ok()) { \
    LOG(ERROR) << status;                \
    return -1;                           \
  }
#define CLEANUP_IF_ERROR(x)              \
  if (auto status = (x); !status.ok()) { \
    LOG(ERROR) << status;                \
    ret_code = -1;                       \
    goto CLEANUP;                        \
  }
#define LOG_IF_ERROR(x)                  \
  if (auto status = (x); !status.ok()) { \
    LOG(ERROR) << status;                \
  }

  umask(0);
  absl::ParseCommandLine(argc, argv);
  bool show_version = absl::GetFlag(FLAGS_show_version);
  if (show_version) {
    std::cout << kVersion << std::endl;
    return 0;
  }

  // 0. Version Info

  LOG(INFO) << absl::StrFormat("Running TCPD Receive Data Path Manager, version (%s)", kVersion);

  // 1. Collect GPU/NIC pair configurations
  std::string gpu_nic_preset = absl::GetFlag(FLAGS_gpu_nic_preset);

  LOG(INFO) << absl::StrFormat(
      "Collecting GPU/NIC pair configurations with preset: %s ...",
      gpu_nic_preset);

  GpuRxqConfigurationList gpu_rxq_configs;
  if (gpu_nic_preset == "manual") {
    if (!absl::GetFlag(FLAGS_gpu_nic_topology_proto).empty() &&
        !absl::GetFlag(FLAGS_gpu_nic_topology).empty()) {
      LOG(FATAL) << "Can't set both gpu_nic_topology_proto and "
                    "gpu_nic_topology at the same time.";
    }
    if (!absl::GetFlag(FLAGS_gpu_nic_topology_proto).empty()) {
      LOG(INFO) << "Getting GPU/NIC topology from text-format proto.";
      gpu_rxq_configs = GpuRxqConfigurationFactory::FromCmdLine(
          absl::GetFlag(FLAGS_gpu_nic_topology_proto));
    } else if (!absl::GetFlag(FLAGS_gpu_nic_topology).empty()) {
      LOG(INFO) << "Getting GPU/NIC toplogy from proto file.";
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

  CHECK(gpu_rxq_configs.gpu_rxq_configs().size() > 0);
  CHECK(gpu_rxq_configs.tcpd_queue_size() > 0);

  LOG(INFO) << "Intializing CUDA ...";
  CU_ASSERT_SUCCESS(cuInit(0));

  // 2. Setup NIC configurator
  LOG(INFO) << absl::StrFormat(
      "Setting up NIC configurator with gpu_nic_preset: %s ...",
      gpu_nic_preset);

  std::unique_ptr<NicConfiguratorInterface> nic_configurator =
      NicConfiguratorFactory::Build(gpu_nic_preset);

  RETURN_IF_ERROR(nic_configurator->Init());

  // 3. Construct rxq_exporters and rx_rule_manager

  LOG(INFO) << "Creating GPU-RXQ exporters and Rx Rule Manager ...";

  std::string gpu_shmem_type = absl::GetFlag(FLAGS_gpu_shmem_type);
  std::unique_ptr<GpuPageExporterInterface> gpu_page_exporter =
      GpuPageExporterFactory::Build(gpu_shmem_type);

  if (!gpu_page_exporter) {
    LOG(ERROR) << "Failed to create gpu_page_exporter";
    return 1;
  }

  std::string uds_path = absl::GetFlag(FLAGS_uds_path);

  RxRuleManager rx_rule_manager(
      /*config_list=*/gpu_rxq_configs,
      /*prefix=*/uds_path,
      /*nic_configurator=*/nic_configurator.get());

  // 4. Configure NIC for TCPDirect
  LOG(INFO) << "Priming the NICs for GPU-RXQ use case ...";

  for (auto &gpu_rxq_config : gpu_rxq_configs.gpu_rxq_configs()) {
    CLEANUP_IF_ERROR(nic_configurator->TogglePrivateFeature(
        gpu_rxq_config.ifname(), "enable-max-rx-buffer-size", true));
    CLEANUP_IF_ERROR(nic_configurator->ToggleFeature(gpu_rxq_config.ifname(),
                                                     "ntuple", true));
    CLEANUP_IF_ERROR(nic_configurator->SetRss(
        gpu_rxq_config.ifname(),
        /*num_queues=*/gpu_rxq_configs.rss_set_size()));
  }

  // 5. Start the Gpu-Rxq exporter
  LOG(INFO) << absl::StrFormat("Starting GPU-RXQ exporters at path: %s ...",
                               uds_path);

  RETURN_IF_ERROR(gpu_page_exporter->Initialize(gpu_rxq_configs, uds_path));
  RETURN_IF_ERROR(gpu_page_exporter->Export());

  for (auto &gpu_rxq_config : gpu_rxq_configs.gpu_rxq_configs()) {
    RETURN_IF_ERROR(nic_configurator->TogglePrivateFeature(
        gpu_rxq_config.ifname(), "enable-header-split", true));
  }

  // 6. Start Rx Rule Manager

  LOG(INFO) << "Starting Rx Rule Manager ...";
  RETURN_IF_ERROR(rx_rule_manager.Init());

  // Setup global flags and handlers
  gShouldStop.store(false);
  signal(SIGINT, sig_handler);
  signal(SIGTERM, sig_handler);

  while (!gShouldStop.load()) {
    sleep(10);
  }

  // Stopping the servers.
CLEANUP:
  LOG(INFO) << "Program terminates, starting clean-up procedure ...";

  LOG(INFO) << "Stopping Rx Rule Manager, recyling stale rules ...";
  rx_rule_manager.Cleanup();

  LOG(INFO) << "Stopping GPU-RXQ exporter, unbind RX queues and de-allocating "
               "GPU memories ...";
  gpu_page_exporter->Cleanup();

  int total_queue =
      gpu_rxq_configs.rss_set_size() + gpu_rxq_configs.tcpd_queue_size();

  LOG(INFO) << "Recovering NIC configurations ...";
  for (auto &gpu_rxq_config : gpu_rxq_configs.gpu_rxq_configs()) {
    LOG_IF_ERROR(nic_configurator->TogglePrivateFeature(
        gpu_rxq_config.ifname(), "enable-header-split", false));
    LOG_IF_ERROR(nic_configurator->TogglePrivateFeature(
        gpu_rxq_config.ifname(), "enable-max-rx-buffer-size", false));
    LOG_IF_ERROR(nic_configurator->SetRss(gpu_rxq_config.ifname(),
                                          /*num_queues=*/total_queue));
    LOG_IF_ERROR(nic_configurator->ToggleFeature(gpu_rxq_config.ifname(),
                                                 "ntuple", false));
  }

  LOG(INFO) << "Clean-up procedure finishes.";
#undef CLEANUP_IF_ERROR
#undef LOG_IF_ERROR
#undef RETURN_IF_ERROR
  return ret_code;
}
