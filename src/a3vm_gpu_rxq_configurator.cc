#include "experimental/users/chechenglin/tcpgpudmad/include/a3vm_gpu_rxq_configurator.h"

#include <memory>
#include <utility>
#include <vector>

namespace tcpdirect {
GpuRxqConfigurationList A3VmGpuRxqConfigurator::GetConfigurations() {
  GpuRxqConfigurationList config_list;
  {
    GpuRxqConfiguration* configuration = config_list.add_gpu_rxq_configs();
    configuration->set_gpu_pci_addr("0000:04:00.0");
    configuration->set_nic_pci_addr("0000:06:00.0");
    configuration->set_ifname("eth1");
  }
  {
    GpuRxqConfiguration* configuration = config_list.add_gpu_rxq_configs();
    configuration->set_gpu_pci_addr("0000:0a:00.0");
    configuration->set_nic_pci_addr("0000:0c:00.0");
    configuration->set_ifname("eth2");
  }
  {
    GpuRxqConfiguration* configuration = config_list.add_gpu_rxq_configs();
    configuration->set_gpu_pci_addr("0000:84:00.0");
    configuration->set_nic_pci_addr("0000:86:00.0");
    configuration->set_ifname("eth3");
  }
  {
    GpuRxqConfiguration* configuration = config_list.add_gpu_rxq_configs();
    configuration->set_gpu_pci_addr("0000:8a:00.0");
    configuration->set_nic_pci_addr("0000:8c:00.0");
    configuration->set_ifname("eth4");
  }
  config_list.set_rss_set_size(8);
  config_list.set_tcpd_queue_size(8);
  config_list.set_max_rx_rules(256);
  return config_list;
}
}  // namespace tcpdirect
