#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_A3VM_GPU_RXQ_CONFIGURATOR_H_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_A3VM_GPU_RXQ_CONFIGURATOR_H_

#include <memory>
#include <vector>

#include "experimental/users/chechenglin/tcpgpudmad/include/gpu_rxq_configurator_interface.h"

namespace tcpdirect {
class A3VmGpuRxqConfigurator : public GpuRxqConfiguratorInterface {
 public:
  GpuRxqConfigurationList GetConfigurations() override;
};
}  // namespace tcpdirect
#endif
