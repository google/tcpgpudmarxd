#ifndef _EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_ETHTOOL_NIC_CONFIGURATOR_H_
#define _EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_ETHTOOL_NIC_CONFIGURATOR_H_

#include <string>

#include "experimental/users/chechenglin/tcpgpudmad/include/flow_steer_ntuple.h"
#include "experimental/users/chechenglin/tcpgpudmad/include/nic_configurator_interface.h"
#include "third_party/absl/status/status.h"

namespace tcpdirect {
class EthtoolNicConfigurator : public NicConfiguratorInterface {
 public:
  EthtoolNicConfigurator() = default;
  ~EthtoolNicConfigurator() override { Cleanup(); }
  absl::Status Init() override { return absl::OkStatus(); }
  void Cleanup() override {}
  absl::Status ToggleHeaderSplit(const std::string& ifname,
                                 bool enable) override;
  absl::Status SetRss(const std::string& ifname, int num_queues) override;
  absl::Status SetNtuple(const std::string& ifname) override;
  absl::Status AddFlow(const std::string& ifname,
                       const struct FlowSteerNtuple& ntuple, int queue_id,
                       int location_id) override;
  absl::Status RemoveFlow(const std::string& ifname, int location_id) override;
  virtual absl::Status RunSystem(const std::string& command);
};
}  // namespace tcpdirect
#endif
