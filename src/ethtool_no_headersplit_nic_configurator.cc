#include "experimental/users/chechenglin/tcpgpudmad/include/ethtool_no_headersplit_nic_configurator.h"

#include <stdlib.h>

#include <string>

#include "experimental/users/chechenglin/tcpgpudmad/include/flow_steer_ntuple.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/str_format.h"

namespace tcpdirect {
namespace {
constexpr int kInitRxSize{512};
constexpr int kEndRxSize{1024};
}  // namespace
absl::Status EthtoolNoHeaderSplitNicConfigurator::ToggleHeaderSplit(
    const std::string& ifname, bool enable) {
  // System does not support header-split configuration.  We change the rx
  // buffer size to trigger ring pool re-allocation.
  return RunSystem(absl::StrFormat("ethtool -G %s rx %d", ifname,
                                   (enable ? kInitRxSize : kEndRxSize)));
}
}  // namespace tcpdirect
