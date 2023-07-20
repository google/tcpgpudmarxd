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

#ifndef _EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_ETHTOOL_NO_HEADERSPLIT_NIC_CONFIGURATOR_H_
#define _EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_ETHTOOL_NO_HEADERSPLIT_NIC_CONFIGURATOR_H_

#include <absl/status/status.h>

#include <string>

#include "include/ethtool_nic_configurator.h"
#include "include/flow_steer_ntuple.h"

namespace gpudirect_tcpxd {
class EthtoolNoHeaderSplitNicConfigurator : public EthtoolNicConfigurator {
 public:
  EthtoolNoHeaderSplitNicConfigurator() = default;
  ~EthtoolNoHeaderSplitNicConfigurator() override = default;
  absl::Status TogglePrivateFeature(const std::string& ifname,
                                    const std::string& feature,
                                    bool on) override;
};
}  // namespace gpudirect_tcpxd
#endif
