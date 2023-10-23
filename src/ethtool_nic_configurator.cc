// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "include/ethtool_nic_configurator.h"

#include <absl/log/log.h>
#include <absl/status/status.h>
#include <absl/status/statusor.h>
#include <absl/strings/str_format.h>
#include <arpa/inet.h>
#include <linux/ethtool.h>
#include <linux/types.h>
#include <stdlib.h>

#include <string>

#include "include/flow_steer_ntuple.h"

namespace gpudirect_tcpxd {
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

absl::Status EthtoolNicConfigurator::SetIpRoute(const std::string& ifname,
                                                int min_rto, bool quickack) {
  if (!min_rto && !quickack)
    return RunSystem(
        absl::StrFormat("ip route replace %s", prev_route_[ifname]));

  std::array<char, 128> buffer;
  std::string cur_route, suffix;
  std::string command =
      absl::StrFormat("ip route show dev %s | grep mtu", ifname);
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"),
                                                pclose);

  if (!pipe) {
    return absl::InternalError("popen() failed!");
  }

  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    cur_route += buffer.data();
  }
  prev_route_[ifname] = cur_route.c_str();

  cur_route.erase(std::remove(cur_route.begin(), cur_route.end(), '\n'),
                  cur_route.cend());

  size_t ind = cur_route.find("quickack");
  if (ind != std::string::npos) {
    cur_route = cur_route.substr(0, ind);
  }

  ind = cur_route.find("rto_min");
  if (ind != std::string::npos) {
    cur_route = cur_route.substr(0, ind);
  }

  if (min_rto) suffix = absl::StrFormat("rto_min %dms", min_rto).c_str();
  if (quickack) suffix = absl::StrFormat("%s quickack 1", suffix).c_str();

  return RunSystem(
      absl::StrFormat("ip route replace %s %s", cur_route, suffix));
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

// GetStat("eth1", "reset_cnt") gets a stat from `ethtool -S eth1`
// and returns it. If it can't find that stat, returns a failed Status
absl::StatusOr<__u32> EthtoolNicConfigurator::GetStat(
    const std::string& ifname, const std::string& statname) {
  __u32 stat_value;
  int scanned;
  char path[4096];
  std::string path_s;
  std::string command = absl::StrFormat(
      "ethtool -S %s | grep '%s:' | awk '{print $NF}'", ifname, statname);

  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"),
                                                pclose);

  if (!pipe) {
    return -1;
  }

  while (fgets(path, sizeof(path), pipe.get())) {
    path_s = std::string(path);

    scanned = sscanf(path, "%d", &stat_value);
    if (scanned == 1) return stat_value;
  }

  return absl::InvalidArgumentError("bad stat name");
}
}  // namespace gpudirect_tcpxd
