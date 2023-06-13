#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_GPU_RXQ_CONFIGURATOR_INTERFACE_H_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_GPU_RXQ_CONFIGURATOR_INTERFACE_H_

#include <memory>
#include <string>
#include <vector>

#include "experimental/users/chechenglin/tcpgpudmad/proto/gpu_rxq_configuration.proto.h"

namespace tcpdirect {

class GpuRxqConfiguratorInterface {
 public:
  virtual GpuRxqConfigurationList GetConfigurations() = 0;
  virtual ~GpuRxqConfiguratorInterface() = default;
};
}  // namespace tcpdirect
#endif
