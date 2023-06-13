#ifndef _EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_ETHTOOL_NO_HEADERSPLIT_NIC_CONFIGURATOR_H_
#define _EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_ETHTOOL_NO_HEADERSPLIT_NIC_CONFIGURATOR_H_

#include <string>

#include "include/ethtool_nic_configurator.h"
#include "include/flow_steer_ntuple.h"
#include <absl/status/status.h>

namespace tcpdirect {
class EthtoolNoHeaderSplitNicConfigurator : public EthtoolNicConfigurator {
 public:
  EthtoolNoHeaderSplitNicConfigurator() = default;
  ~EthtoolNoHeaderSplitNicConfigurator() override = default;
  absl::Status ToggleHeaderSplit(const std::string& ifname,
                                 bool enable) override;
};
}  // namespace tcpdirect
#endif
