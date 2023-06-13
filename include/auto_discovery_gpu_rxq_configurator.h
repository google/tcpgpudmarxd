#ifndef EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_AUTO_DISCOVERY_GPU_RXQ_CONFIGURATOR_H_
#define EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_AUTO_DISCOVERY_GPU_RXQ_CONFIGURATOR_H_


#include <memory>
#include <vector>

#include "include/gpu_rxq_configurator_interface.h"

namespace tcpdirect {
class AutoDiscoveryGpuRxqConfigurator : public GpuRxqConfiguratorInterface {
 public:
  GpuRxqConfigurationList GetConfigurations() override;
};
}  // namespace tcpdirect
#endif  // EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_AUTO_DISCOVERY_GPU_RXQ_CONFIGURATOR_H_
