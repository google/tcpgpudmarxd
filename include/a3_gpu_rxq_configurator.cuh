#ifndef EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_A3_GPU_RXQ_CONFIGURATOR_CU_H_
#define EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_A3_GPU_RXQ_CONFIGURATOR_CU_H_

#include <memory>
#include <vector>

#include "cuda/common.cuh"
#include "include/gpu_rxq_configurator_interface.h"
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>


namespace tcpdirect {
class A3GpuRxqConfigurator : public GpuRxqConfiguratorInterface {
 public:
  GpuRxqConfigurationList GetConfigurations() override;
};
}  // namespace tcpdirect

#endif  // EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_A3_GPU_RXQ_CONFIGURATOR_CU_H_
