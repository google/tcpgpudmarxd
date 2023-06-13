#include "include/auto_discovery_gpu_rxq_configurator.h"

#include <memory>
#include <utility>
#include <vector>

#include "include/a3_gpu_rxq_configurator.cuh"

namespace tcpdirect {
GpuRxqConfigurationList AutoDiscoveryGpuRxqConfigurator::GetConfigurations() {
  GpuRxqConfigurationList config_list;

  std::unique_ptr<tcpdirect::GpuRxqConfiguratorInterface> gpu_rxq_configurator =
      std::make_unique<tcpdirect::A3GpuRxqConfigurator>();

  auto a3_config_list = gpu_rxq_configurator->GetConfigurations();
  for (auto& a3_config : *a3_config_list.mutable_gpu_rxq_configs()) {
    if (a3_config.nic_pci_addr() > a3_config.gpu_pci_addr()) {
      *(config_list.add_gpu_rxq_configs()) = std::move(a3_config);
    }
  }
  return config_list;
}
}  // namespace tcpdirect
