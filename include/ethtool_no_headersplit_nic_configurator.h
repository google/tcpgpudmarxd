#ifndef _EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_ETHTOOL_NO_HEADERSPLIT_NIC_CONFIGURATOR_H_
#define _EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_ETHTOOL_NO_HEADERSPLIT_NIC_CONFIGURATOR_H_

#include <absl/status/status.h>

#include <string>

#include "include/ethtool_nic_configurator.h"
#include "include/flow_steer_ntuple.h"

namespace tcpdirect {
class EthtoolNoHeaderSplitNicConfigurator : public EthtoolNicConfigurator {
 public:
  EthtoolNoHeaderSplitNicConfigurator() = default;
  ~EthtoolNoHeaderSplitNicConfigurator() override = default;
  absl::Status TogglePrivateFeature(const std::string& ifname,
                                    const std::string& feature,
                                    bool on) override;
};
}  // namespace tcpdirect
#endif
