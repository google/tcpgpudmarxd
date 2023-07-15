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

#ifndef _EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_DUMMY_ETHTOOL_NIC_CONFIGURATOR_H_
#define _EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_DUMMY_ETHTOOL_NIC_CONFIGURATOR_H_

#include <string>

#include "include/ethtool_nic_configurator.h"
#include <absl/status/status.h>

namespace tcpdirect {
class DummyEthtoolNicConfigurator : public EthtoolNicConfigurator {
 public:
  DummyEthtoolNicConfigurator() = default;
  absl::Status RunSystem(const std::string& command) override;
};
}  // namespace tcpdirect
#endif
