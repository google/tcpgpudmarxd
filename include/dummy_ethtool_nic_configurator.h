#ifndef _EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_DUMMY_ETHTOOL_NIC_CONFIGURATOR_H_
#define _EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_DUMMY_ETHTOOL_NIC_CONFIGURATOR_H_

#include <string>

#include "experimental/users/chechenglin/tcpgpudmad/include/ethtool_nic_configurator.h"
#include "third_party/absl/status/status.h"

namespace tcpdirect {
class DummyEthtoolNicConfigurator : public EthtoolNicConfigurator {
 public:
  DummyEthtoolNicConfigurator() = default;
  absl::Status RunSystem(const std::string& command) override;
};
}  // namespace tcpdirect
#endif
