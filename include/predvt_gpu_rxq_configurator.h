#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_PREDVT_GPU_RXQ_CONFIGURATOR_H_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_PREDVT_GPU_RXQ_CONFIGURATOR_H_

#include <memory>
#include <vector>

#include "include/gpu_rxq_configurator_interface.h"

namespace tcpdirect {
class PreDvtGpuRxqConfigurator : public GpuRxqConfiguratorInterface {
 public:
  GpuRxqConfigurationList GetConfigurations() override;
};
}  // namespace tcpdirect
#endif
