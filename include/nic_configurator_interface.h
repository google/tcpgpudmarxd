#ifndef _EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_NIC_CONFIGURATOR_INTERFACE_H_
#define _EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_NIC_CONFIGURATOR_INTERFACE_H_

#include <cstdint>
#include <string>

#include "experimental/users/chechenglin/tcpgpudmad/include/flow_steer_ntuple.h"
#include "third_party/absl/status/status.h"

namespace tcpdirect {
class NicConfiguratorInterface {
 public:
  virtual ~NicConfiguratorInterface() = default;
  virtual absl::Status Init() = 0;
  virtual void Cleanup() = 0;
  virtual absl::Status ToggleHeaderSplit(const std::string& ifname,
                                         bool enable) = 0;
  virtual absl::Status SetRss(const std::string& ifname, int num_queues) = 0;
  virtual absl::Status SetNtuple(const std::string& ifname) = 0;
  virtual absl::Status AddFlow(const std::string& ifname,
                               const struct FlowSteerNtuple& ntuple,
                               int queue_id, int location_id) = 0;
  virtual absl::Status RemoveFlow(const std::string& ifname,
                                  int location_id) = 0;
};
}  // namespace tcpdirect
#endif
