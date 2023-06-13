#include "experimental/users/chechenglin/tcpgpudmad/include/monstertruck_gpu_rxq_configurator.h"

#include <memory>
#include <utility>
#include <vector>

namespace tcpdirect {
GpuRxqConfigurationList MonstertruckGpuRxqConfigurator::GetConfigurations() {
  GpuRxqConfigurationList config_list;
  {
    GpuRxqConfiguration* configuration = config_list.add_gpu_rxq_configs();
    configuration->set_gpu_pci_addr("0000:62:00.0");
    configuration->set_nic_pci_addr("0000:63:00.0");
    configuration->set_ifname("hpn2");
  }

  {
    GpuRxqConfiguration* configuration = config_list.add_gpu_rxq_configs();
    configuration->set_gpu_pci_addr("0000:6a:00.0");
    configuration->set_nic_pci_addr("0000:6b:00.0");
    configuration->set_ifname("hpn1");
  }

  {
    GpuRxqConfiguration* configuration = config_list.add_gpu_rxq_configs();
    configuration->set_gpu_pci_addr("0000:b3:00.0");
    configuration->set_nic_pci_addr("0000:b4:00.0");
    configuration->set_ifname("hpn3");
  }

  {
    GpuRxqConfiguration* configuration = config_list.add_gpu_rxq_configs();
    configuration->set_gpu_pci_addr("0000:bb:00.0");
    configuration->set_nic_pci_addr("0000:bc:00.0");
    configuration->set_ifname("hpn4");
  }
  config_list.set_tcpd_queue_size(14);
  config_list.set_rss_set_size(0);
  return config_list;
}
}  // namespace tcpdirect
