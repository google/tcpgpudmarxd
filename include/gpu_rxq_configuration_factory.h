/*
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_GPU_RXQ_CONFIGURATOR_FACTORY_H_
#define __EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_GPU_RXQ_CONFIGURATOR_FACTORY_H_

#include <memory>
#include <string>
#include <vector>

#include "include/gpu_rxq_configurator_interface.h"

namespace gpudirect_tcpxd {
class GpuRxqConfigurationFactory {
 public:
  static GpuRxqConfigurationList FromCmdLine(const std::string& proto_string);
  static GpuRxqConfigurationList BuildPreset(const std::string& name);
  static GpuRxqConfigurationList FromFile(const std::string& filename);
};
}  // namespace gpudirect_tcpxd
#endif
