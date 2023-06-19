#include "include/ethtool_no_headersplit_nic_configurator.h"

#include <absl/log/log.h>
#include <absl/status/status.h>
#include <absl/strings/str_format.h>
#include <stdlib.h>

#include <string>

#include "include/flow_steer_ntuple.h"

namespace tcpdirect {
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
}  // namespace tcpdirect
