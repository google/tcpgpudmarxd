#include "include/a3_gpu_rxq_configurator.cuh"
#include "include/pci_helpers.h"

#include <sys/types.h>
#include <ifaddrs.h>
#include <dirent.h>
#include <stdio.h>
#include <memory>
#include <vector>
#include <absl/log/log.h>
#include <absl/flags/flag.h>
#include <absl/strings/ascii.h>

#define PCI_INFO_LEN 1024
// <2-4 digit domain>:<2-4 digit bus>:<2 digit device>:<1 digit function>
#define MAX_PCI_ADDR_LEN 16
#define MAX_HOPS 4
ABSL_FLAG(
    int, num_hops, 2,
    "Number of hops to the PCIE switch shared by the 2 GPUs and the NIC(s).");


namespace tcpdirect {

GpuRxqConfigurationList A3GpuRxqConfigurator::GetConfigurations() {
  GpuRxqConfigurationList config_list;
  absl::flat_hash_map<std::string, std::string> netdev_to_pci;
  absl::flat_hash_map<std::string, std::string> pci_to_netdev;
  struct ifaddrs *all_ifs = nullptr;
  if (getifaddrs(&all_ifs) != 0 || all_ifs == nullptr) {
    LOG(ERROR) << "Failed to retrieve network ifs, error: " << strerror(errno);
    return config_list;
  }
  struct ifaddrs *head = all_ifs;
  do {
    // Skip non-IPV4 and non-IPV6 interfaces
    if (head->ifa_addr->sa_family != AF_INET &&
        head->ifa_addr->sa_family != AF_INET6) {
      continue;
    }
    // Skip interfaces we have already seen before
    if (netdev_to_pci.contains(head->ifa_name)) continue;
    char if_sysfs_path[PATH_MAX] = {0};
    snprintf(if_sysfs_path, PATH_MAX, "/sys/class/net/%s/device/",
             head->ifa_name);
    char if_sysfs_realpath[PATH_MAX] = {0};
    // Only pick interfaces that has an actual PCI device associated
    if (realpath(if_sysfs_path, if_sysfs_realpath) == nullptr) continue;
    int last_char_idx = strlen(if_sysfs_realpath) - 1;
    if (if_sysfs_realpath[last_char_idx] == '/')
      if_sysfs_realpath[last_char_idx] = '\0';
    int path_length = 0;
    for (int i = 0; i < strlen(if_sysfs_realpath); i++) {
      if (if_sysfs_realpath[i] == '/') ++path_length;
    }
    // The host NIC should be closest to the CPU, exclude it.
    // TODO (penzhao@): consider using pciutil
    if (path_length <= 5) continue;
    int kNumHops = std::min(absl::GetFlag(FLAGS_num_hops), MAX_HOPS);
    char *pci_addr = nullptr;
    for (int i = 0; i < kNumHops; i++) {
      char *slash = strrchr(if_sysfs_realpath, '/');
      /* First delimiter gives us the pci address*/
      if (i == 0) pci_addr = slash + 1;
      *slash = '\0';
    }
    uint16_t temp_domain, temp_bus, temp_device, temp_function;
    /* Not a valid PCI address */
    LOG(INFO) << "PCI addr for net if " << head->ifa_name << ": "
              << std::string(pci_addr);
    if (parse_pci_addr(pci_addr, &temp_domain, &temp_bus, &temp_device,
                       &temp_function))
      continue;
    netdev_to_pci.emplace(head->ifa_name, pci_addr);
    LOG(INFO) << "Root dir: " << if_sysfs_realpath;
    std::vector<std::string> candidates;
    list_vendor_devices(if_sysfs_realpath, &candidates, "0x10de");
    for (auto &candidate : candidates) {
      LOG(INFO) << "Potential candidate: " << candidate;
      pci_to_netdev.emplace(absl::AsciiStrToLower(candidate), head->ifa_name);
    }
  } while ((head = head->ifa_next) != nullptr);
  // Get PCI addrs for CUDA devices and find the closest NIC.
  int num_cuda_device = 0;
  CUDA_ASSERT_SUCCESS(cudaGetDeviceCount(&num_cuda_device));
  for (int i = 0; i < num_cuda_device; i++) {
    char gpu_pci_addr[MAX_PCI_ADDR_LEN];
    CUDA_ASSERT_SUCCESS(
        cudaDeviceGetPCIBusId(gpu_pci_addr, MAX_PCI_ADDR_LEN, i));
    for (int i = 0; i < MAX_PCI_ADDR_LEN; i++) {
      gpu_pci_addr[i] = tolower(gpu_pci_addr[i]);
    }
    if (!pci_to_netdev.contains(gpu_pci_addr)) {
      LOG(ERROR) << "Cannot find corresponding GPU NIC for GPU " << gpu_pci_addr
                 << ".";
      continue;
    }
    auto &netdev_name = pci_to_netdev[gpu_pci_addr];
    if (!netdev_to_pci.contains(netdev_name)) {
      LOG(ERROR) << "Net dev " << netdev_name << " is not discovered before.";
      continue;
    }
    LOG(INFO) << "Corresponding PCI NIC for GPU PCI addr " << gpu_pci_addr
              << " is " << netdev_name;

    GpuRxqConfiguration configuration;
    configuration.set_gpu_pci_addr(std::string(gpu_pci_addr));
    configuration.set_nic_pci_addr(netdev_to_pci[netdev_name]);
    configuration.set_ifname(netdev_name);
    *(config_list.add_gpu_rxq_configs()) = std::move(configuration);
  }
  freeifaddrs(all_ifs);
  config_list.set_tcpd_queue_size(15);
  config_list.set_rss_set_size(0);
  return config_list;
}
}  // namespace tcpdirect
