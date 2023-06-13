#ifndef __EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_GPU_RXQ_CONFIGURATOR_FACTORY_H_
#define __EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_GPU_RXQ_CONFIGURATOR_FACTORY_H_

#include <memory>
#include <string>
#include <vector>

#include "include/gpu_rxq_configurator_interface.h"

namespace tcpdirect {
class GpuRxqConfigurationFactory {
 public:
  static GpuRxqConfigurationList FromCmdLine(const std::string& proto_string);
  static GpuRxqConfigurationList BuildPreset(const std::string& name);
  static GpuRxqConfigurationList FromFile(const std::string& filename);
};
}  // namespace tcpdirect
#endif
