/*
 Copyright 2024 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

#ifndef EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_A3_GPU_RXQ_CONFIGURATOR_CU_H_
#define EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_A3_GPU_RXQ_CONFIGURATOR_CU_H_

#include <memory>
#include <vector>

#include "cuda/common.cuh"
#include "include/gpu_rxq_configurator_interface.h"
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>


namespace gpudirect_tcpxd {
class A3GpuRxqConfigurator : public GpuRxqConfiguratorInterface {
 public:
  GpuRxqConfigurationList GetConfigurations() override;
};
}  // namespace gpudirect_tcpxd

#endif  // EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_A3_GPU_RXQ_CONFIGURATOR_CU_H_
