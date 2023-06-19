#include "include/ethtool_nic_configurator.h"

#include <absl/log/log.h>
#include <absl/status/status.h>
#include <absl/strings/str_format.h>
#include <arpa/inet.h>
#include <linux/ethtool.h>
#include <stdlib.h>

#include <string>

#include "include/flow_steer_ntuple.h"

namespace tcpdirect {
absl::Status EthtoolNicConfigurator::TogglePrivateFeature(
    const std::string& ifname, const std::string& feature, bool on) {
  return RunSystem(absl::StrFormat("ethtool --set-priv-flags %s %s %s", ifname,
                                   feature, (on ? "on" : "off")));
}

absl::Status EthtoolNicConfigurator::ToggleFeature(const std::string& ifname,
                                                   const std::string& feature,
                                                   bool on) {
  return RunSystem(absl::StrFormat("ethtool -K %s %s %s", ifname, feature,
                                   (on ? "on" : "off")));
}

absl::Status EthtoolNicConfigurator::SetRss(const std::string& ifname,
                                            int num_queues) {
  if (num_queues > 0)
    return RunSystem(absl::StrFormat("ethtool --set-rxfh-indir %s equal %d",
                                     ifname, num_queues));
  return absl::OkStatus();
}
absl::Status EthtoolNicConfigurator::AddFlow(
    const std::string& ifname, const struct FlowSteerNtuple& ntuple,
    int queue_id, int location_id) {
  if (ntuple.flow_type != TCP_V4_FLOW) {
    return absl::InvalidArgumentError(
        "Only tcp4 is supported for flow steering now.");
  }
  char buf[256];
  std::string src_ip(
      inet_ntop(AF_INET, (const char*)&ntuple.src_sin.sin_addr, buf, 256));
  std::string dst_ip(
      inet_ntop(AF_INET, (const char*)&ntuple.dst_sin.sin_addr, buf, 256));
  return RunSystem(
      absl::StrFormat("ethtool -N %s flow-type tcp4 src-ip %s dst-ip %s "
                      "src-port %d dst-port %d queue %d loc %d",
                      ifname, src_ip, dst_ip, ntohs(ntuple.src_sin.sin_port),
                      ntohs(ntuple.dst_sin.sin_port), queue_id, location_id));
}
absl::Status EthtoolNicConfigurator::RemoveFlow(const std::string& ifname,
                                                int location_id) {
  return RunSystem(
      absl::StrFormat("ethtool -N %s delete %d", ifname, location_id));
}

absl::Status EthtoolNicConfigurator::RunSystem(const std::string& command) {
  LOG(INFO) << "Run system: " << command;
  if (auto ret = system(command.c_str()); ret != 0) {
    std::string error_msg = absl::StrFormat(
        "Run system failed.  Ret: %d, Command: %s", ret, command);
    return absl::InternalError(error_msg);
  }
  return absl::OkStatus();
}
}  // namespace tcpdirect
