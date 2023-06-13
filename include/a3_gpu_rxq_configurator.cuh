#ifndef EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_A3_GPU_RXQ_CONFIGURATOR_CU_H_
#define EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_A3_GPU_RXQ_CONFIGURATOR_CU_H_

#include <memory>
#include <vector>

#include "experimental/users/chechenglin/tcpgpudmad/cuda/common.cu.h"
#include "experimental/users/chechenglin/tcpgpudmad/include/gpu_rxq_configurator_interface.h"
#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/container/flat_hash_set.h"
#include "third_party/gpus/cuda/include/cuda.h"

namespace tcpdirect {
class A3GpuRxqConfigurator : public GpuRxqConfiguratorInterface {
 public:
  GpuRxqConfigurationList GetConfigurations() override;
};
}  // namespace tcpdirect

#endif  // EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_A3_GPU_RXQ_CONFIGURATOR_CU_H_
