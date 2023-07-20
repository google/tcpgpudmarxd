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

#ifndef _EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_NIC_CONFIGURATOR_FACTORY_
#define _EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_NIC_CONFIGURATOR_FACTORY_

#include <memory>
#include <string>

#include "include/nic_configurator_interface.h"

namespace gpudirect_tcpxd {
class NicConfiguratorFactory {
 public:
  static std::unique_ptr<NicConfiguratorInterface> Build(
      const std::string& name);
};
}  // namespace gpudirect_tcpxd
#endif
