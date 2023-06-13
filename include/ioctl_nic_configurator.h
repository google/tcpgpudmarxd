#ifndef _EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_IOCTL_NIC_CONFIGURATOR_H_
#define _EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_IOCTL_NIC_CONFIGURATOR_H_

#include <linux/ethtool.h>

#include <cstdint>
#include <string>

#include "experimental/users/chechenglin/tcpgpudmad/include/flow_steer_ntuple.h"
#include "experimental/users/chechenglin/tcpgpudmad/include/nic_configurator_interface.h"
#include "third_party/absl/status/status.h"

namespace tcpdirect {
class IoctlNicConfigurator : public NicConfiguratorInterface {
 public:
  IoctlNicConfigurator() = default;
  ~IoctlNicConfigurator() override { Cleanup(); }
  absl::Status Init() override;
  void Cleanup() override;
  absl::Status ToggleHeaderSplit(const std::string& ifname,
                                 bool enable) override;
  absl::Status SetRss(const std::string& ifname, int num_queues) override;
  absl::Status SetNtuple(const std::string& ifname) override;
  absl::Status AddFlow(const std::string& ifname,
                       const struct FlowSteerNtuple& ntuple, int queue_id,
                       int location_id) override;
  absl::Status RemoveFlow(const std::string& ifname, int location_id) override;

 private:
  int SendIoctl(const std::string& ifname, char* cmd);
  struct ethtool_gstrings* GetStringsSet(const std::string& ifname,
                                         ethtool_stringset set_id,
                                         ptrdiff_t drvinfo_offset,
                                         int null_terminate);

  struct feature_state* GetFeatures(const std::string& ifname,
                                    const struct feature_defs* defs);
  struct feature_defs* GetFeatureDefs(const std::string& ifname);
// ethtool_flag should be the value from include/linux/ethtool.h
  absl::Status ControlEthtoolFeatures(const std::string& ifname,
                            uint32_t ethtool_flag, bool enable);

  int fd_{-1};
};
}  // namespace tcpdirect
#endif
