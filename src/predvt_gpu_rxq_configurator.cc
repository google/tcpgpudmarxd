#include "experimental/users/chechenglin/tcpgpudmad/include/predvt_gpu_rxq_configurator.h"

#include <memory>
#include <utility>
#include <vector>

namespace tcpdirect {
GpuRxqConfigurationList PreDvtGpuRxqConfigurator::GetConfigurations() {
  GpuRxqConfigurationList config_list;
  {
    GpuRxqConfiguration* configuration = config_list.add_gpu_rxq_configs();
    configuration->set_gpu_pci_addr("0000:1d:00.0");
    configuration->set_nic_pci_addr("0000:0a:00.0");
    configuration->set_ifname("dcn1");
  }

  {
    GpuRxqConfiguration* configuration = config_list.add_gpu_rxq_configs();
    configuration->set_gpu_pci_addr("0000:51:00.0");
    configuration->set_nic_pci_addr("0000:43:00.0");
    configuration->set_ifname("dcn2");
  }

  {
    GpuRxqConfiguration* configuration = config_list.add_gpu_rxq_configs();
    configuration->set_gpu_pci_addr("0000:96:00.0");
    configuration->set_nic_pci_addr("0000:88:00.0");
    configuration->set_ifname("dcn3");
  }

  {
    GpuRxqConfiguration* configuration = config_list.add_gpu_rxq_configs();
    configuration->set_gpu_pci_addr("0000:c9:00.0");
    configuration->set_nic_pci_addr("0000:bb:00.0");
    configuration->set_ifname("dcn4");
  }
  config_list.set_tcpd_queue_size(14);
  config_list.set_rss_set_size(0);
  return config_list;
}
}  // namespace tcpdirect
