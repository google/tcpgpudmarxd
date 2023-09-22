#include <absl/debugging/failure_signal_handler.h>
#include <absl/debugging/symbolize.h>
#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/log/check.h>
#include <absl/log/log.h>
#include <absl/status/status.h>
#include <absl/strings/str_format.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/netlink.h>
#include <linux/rtnetlink.h>
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
#include <thread>
#include <unordered_map>
#include <vector>

#include "cuda/common.cuh"
#include "cuda/gpu_page_exporter_factory.cuh"
#include "include/gpu_page_exporter_interface.h"
#include "include/gpu_rxq_configuration_factory.h"
#include "include/nic_configurator_factory.h"
#include "include/nic_configurator_interface.h"
#include "include/rx_rule_manager.h"

using gpudirect_tcpxd::GpuPageExporterFactory;
using gpudirect_tcpxd::GpuPageExporterInterface;
using gpudirect_tcpxd::GpuRxqConfiguration;
using gpudirect_tcpxd::GpuRxqConfigurationFactory;
using gpudirect_tcpxd::GpuRxqConfigurationList;
using gpudirect_tcpxd::NicConfiguratorFactory;
using gpudirect_tcpxd::NicConfiguratorInterface;
using gpudirect_tcpxd::RxRuleManager;

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
ABSL_FLAG(uint64_t, rx_pool_size, 0,
          "Receive buffer size. Default: 0, meaning no override and either the "
          "value from GpuRxqConfigurationList (if present) or the component "
          "level default value will be used.");
ABSL_FLAG(uint32_t, max_rx_rules, 0,
          "Default: 0, meaning no override and either the value from the "
          "config (if present) or the component level default will be used.  "
          "Maximum number of flow steering rules to use.");
ABSL_FLAG(bool, enable_quickack, false,
          "Default: 0, meaning quickack is not added to route");
ABSL_FLAG(std::string, tuning_script_path, "/a3-tuning-scripts",
          "The path where networking tuning script is kept, "
          "updated separately from this binary. ");

namespace {

constexpr std::string_view kVersion{"v2.0.5"};

static std::atomic<bool> gShouldStop(false);

void sig_handler(int signum) {
  if (signum == SIGINT || signum == SIGTERM) {
    gShouldStop.store(true, std::memory_order_release);
  }
}

absl::Status DisableMtuProbing() {
  auto ret = system("sysctl -w net.ipv4.tcp_mtu_probing=0");
  if (ret != 0) {
    std::string error_msg =
        absl::StrFormat("Disable Mtu Probing Failed. Ret: %d", ret);
    return absl::InternalError(error_msg);
  }
  LOG(INFO) << "Disable Mtu Probing Succeeded.";
  return absl::OkStatus();
}

}  // namespace

int init_netlink() {
  struct sockaddr_nl addr;
  int nl_socket = socket(AF_NETLINK, SOCK_RAW, NETLINK_ROUTE);

  if (nl_socket < 0) {
    LOG(ERROR) << "Netlink socket open";
    return nl_socket;
  }

  memset((void*)&addr, 0, sizeof(addr));

  addr.nl_family = AF_NETLINK;
  addr.nl_groups = RTMGRP_LINK;

  if (bind(nl_socket, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
    LOG(ERROR) << "Netlink socket bind";
    return -1;
  }
  return nl_socket;
}

void* handle_netlink_event(void* main_thread) {
  int status;
  char buf[4096];
  struct iovec iov = {buf, sizeof(buf)};
  struct sockaddr_nl snl;
  struct msghdr msg = {(void*)&snl, sizeof(snl), &iov, 1, NULL, 0, 0};
  struct nlmsghdr* h;
  struct ifinfomsg* ifi;
  int nl_socket = init_netlink();
  unsigned int ifindices[4] = {if_nametoindex("eth1"), if_nametoindex("eth2"),
                               if_nametoindex("eth3"), if_nametoindex("eth4")};
  int main_id = *(int*)main_thread;
  bool tracked_netdev = false;

  if (nl_socket < 0) return NULL;

  while (1) {
    status = recvmsg(nl_socket, &msg, 0);

    if (status < 0) {
      if (errno == EWOULDBLOCK || errno == EAGAIN)
        LOG(ERROR) << "read_netlink: Error recvmsg: " << status;
      pthread_kill((pthread_t)main_id, SIGTERM);
      return NULL;
    }

    /* We need to handle more than one message per 'recvmsg' */
    for (h = (struct nlmsghdr*)buf; NLMSG_OK(h, (unsigned int)status);
         h = NLMSG_NEXT(h, status)) {
      /* Message is some kind of error */
      if (h->nlmsg_type == NLMSG_ERROR) {
        LOG(ERROR) << "read_netlink: Message is an error";
        pthread_kill((pthread_t)main_id, SIGTERM);
        return NULL;
      }
      ifi = (struct ifinfomsg*)NLMSG_DATA(h);
      tracked_netdev = std::find(std::begin(ifindices), std::end(ifindices),
                                 ifi->ifi_index) != std::end(ifindices);
      if (tracked_netdev) {
        if (h->nlmsg_type == RTM_DELLINK ||
            (ifi->ifi_change & IFF_UP) && !(ifi->ifi_flags & IFF_UP)) {
          LOG(ERROR) << "DEBUG: read_netlink: device down received ";
          pthread_kill((pthread_t)main_id, SIGTERM);
          return NULL;
        }
      }
    }
  }
}

int main(int argc, char** argv) {
  absl::InitializeSymbolizer(argv[0]);
  absl::FailureSignalHandlerOptions options;

  absl::InstallFailureSignalHandler(options);

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

  RETURN_IF_ERROR(DisableMtuProbing());

  umask(0);
  absl::ParseCommandLine(argc, argv);
  bool show_version = absl::GetFlag(FLAGS_show_version);
  if (show_version) {
    std::cout << kVersion << std::endl;
    return 0;
  }

  // 0. Version Info

  LOG(INFO) << absl::StrFormat(
      "Running GPUDirect-TCPX Receive Data Path Manager, version (%s)",
      kVersion);

  pthread_t netdev_status;
  int main_id = pthread_self();
  pthread_create(&netdev_status, NULL, handle_netlink_event, &main_id);

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

  if (absl::GetFlag(FLAGS_rx_pool_size) > 0) {
    // overrides rx_pool_size in the configs
    size_t rx_pool_size = absl::GetFlag(FLAGS_rx_pool_size);
    LOG(INFO) << absl::StrFormat("Overriding rx_pool_size: %ld", rx_pool_size);
    gpu_rxq_configs.set_rx_pool_size(rx_pool_size);
  }

  if (absl::GetFlag(FLAGS_max_rx_rules) > 0) {
    // overrides rx_rule_limit in the configs
    size_t max_rx_rules = absl::GetFlag(FLAGS_max_rx_rules);
    LOG(INFO) << absl::StrFormat("Overriding max_rx_rules: %ld", max_rx_rules);
    if (gpu_nic_preset == "a3vm") {
      LOG(WARNING) << "Ignore max_rx_rules override for 'a3vm', using "
                      "component level default value: "
                   << gpu_rxq_configs.max_rx_rules();
    } else {
      gpu_rxq_configs.set_max_rx_rules(max_rx_rules);
    }
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

  for (auto& gpu_rxq_config : gpu_rxq_configs.gpu_rxq_configs()) {
    // Resetting header-split here to ensure that the subsequent enablement will
    // trigger re-initializing the receive buffer pool.
    LOG_IF_ERROR(nic_configurator->TogglePrivateFeature(
        gpu_rxq_config.ifname(), "enable-strict-header-split", false));
    LOG_IF_ERROR(nic_configurator->TogglePrivateFeature(
        gpu_rxq_config.ifname(), "enable-header-split", false));
    // Resetting Ntuple here to flush all stale flow steering rules.
    LOG_IF_ERROR(nic_configurator->ToggleFeature(gpu_rxq_config.ifname(),
                                                 "ntuple", false));
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

  for (auto& gpu_rxq_config : gpu_rxq_configs.gpu_rxq_configs()) {
    RETURN_IF_ERROR(nic_configurator->TogglePrivateFeature(
        gpu_rxq_config.ifname(), "enable-strict-header-split", true));
    CLEANUP_IF_ERROR(nic_configurator->SetIpRoute(
        gpu_rxq_config.ifname(), /*min_rto=*/5 /*ms*/,
        /* quickack = */ absl::GetFlag(FLAGS_enable_quickack)));
  }
  CLEANUP_IF_ERROR(nic_configurator->RunSystem(absl::StrFormat("%s/setup.sh", absl::GetFlag(FLAGS_tuning_script_path))));

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

  pthread_cancel(netdev_status);
  pthread_join(netdev_status, NULL);
  LOG(INFO) << "Canceling netdev monitoring thread";

  LOG(INFO) << "Stopping Rx Rule Manager, recyling stale rules ...";
  rx_rule_manager.Cleanup();

  LOG(INFO) << "Stopping GPU-RXQ exporter, unbind RX queues and de-allocating "
               "GPU memories ...";
  gpu_page_exporter->Cleanup();

  int total_queue =
      gpu_rxq_configs.rss_set_size() + gpu_rxq_configs.tcpd_queue_size();

  LOG(INFO) << "Recovering NIC configurations ...";
  for (auto& gpu_rxq_config : gpu_rxq_configs.gpu_rxq_configs()) {
    LOG_IF_ERROR(nic_configurator->TogglePrivateFeature(
        gpu_rxq_config.ifname(), "enable-strict-header-split", true));
    LOG_IF_ERROR(nic_configurator->TogglePrivateFeature(
        gpu_rxq_config.ifname(), "enable-strict-header-split", false));
    LOG_IF_ERROR(nic_configurator->TogglePrivateFeature(
        gpu_rxq_config.ifname(), "enable-header-split", false));
    LOG_IF_ERROR(nic_configurator->TogglePrivateFeature(
        gpu_rxq_config.ifname(), "enable-max-rx-buffer-size", false));
    LOG_IF_ERROR(nic_configurator->SetRss(gpu_rxq_config.ifname(),
                                          /*num_queues=*/total_queue));
    LOG_IF_ERROR(nic_configurator->ToggleFeature(gpu_rxq_config.ifname(),
                                                 "ntuple", false));
    LOG_IF_ERROR(nic_configurator->SetIpRoute(gpu_rxq_config.ifname(),
                                              /*min_rto=*/0 /*ms*/,
                                              /* quickack = */ false));
  }
  LOG_IF_ERROR(nic_configurator->RunSystem(absl::StrFormat("%s/teardown.sh", absl::GetFlag(FLAGS_tuning_script_path))));

  LOG(INFO) << "Clean-up procedure finishes.";
#undef CLEANUP_IF_ERROR
#undef LOG_IF_ERROR
#undef RETURN_IF_ERROR
  return ret_code;
}
