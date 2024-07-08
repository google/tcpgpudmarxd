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

#include "include/ethtool_no_headersplit_nic_configurator.h"

#include <absl/log/log.h>
#include <absl/status/status.h>
#include <absl/strings/str_format.h>
#include <stdlib.h>

#include <string>

#include "include/flow_steer_ntuple.h"

namespace gpudirect_tcpxd {
namespace {
constexpr int kInitRxSize{512};
constexpr int kEndRxSize{1024};
}  // namespace
absl::Status EthtoolNoHeaderSplitNicConfigurator::TogglePrivateFeature(
    const std::string& ifname, const std::string& feature, bool on) {
  // System does not support header-split configuration.  We change the rx
  // buffer size to trigger ring pool re-allocation.
  if (feature == "enable-header-split") {
    auto first =
        RunSystem(absl::StrFormat("ethtool -G %s rx %d", ifname, kInitRxSize));
    if (!first.ok()) {
      LOG(INFO) << "First ethtool -G command failed, probably due to same "
                   "setting is already applied: "
                << first;
    }
    return RunSystem(
        absl::StrFormat("ethtool -G %s rx %d", ifname, kEndRxSize));
  }
  return EthtoolNicConfigurator::TogglePrivateFeature(ifname, feature, on);
}
}  // namespace gpudirect_tcpxd
