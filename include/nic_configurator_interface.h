/*
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_NIC_CONFIGURATOR_INTERFACE_H_
#define _EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_NIC_CONFIGURATOR_INTERFACE_H_

#include <absl/status/status.h>
#include <absl/status/statusor.h>
#include <linux/types.h>

#include <cstdint>
#include <string>

#include "include/flow_steer_ntuple.h"

namespace gpudirect_tcpxd {
class NicConfiguratorInterface {
 public:
  virtual ~NicConfiguratorInterface() = default;
  virtual absl::Status Init() = 0;
  virtual void Cleanup() = 0;
  virtual absl::Status TogglePrivateFeature(const std::string& ifname,
                                            const std::string& feature,
                                            bool on) = 0;
  virtual absl::Status ToggleFeature(const std::string& ifname,
                                     const std::string& feature, bool on) = 0;
  virtual absl::Status SetRss(const std::string& ifname, int num_queues) = 0;
  virtual absl::Status AddFlow(const std::string& ifname,
                               const struct FlowSteerNtuple& ntuple,
                               int queue_id, int location_id) = 0;
  virtual absl::Status RemoveFlow(const std::string& ifname,
                                  int location_id) = 0;
  virtual absl::Status SetIpRoute(const std::string& ifname, int min_rto,
                                  bool quickack) = 0;
  virtual absl::Status RunSystem(const std::string& command) = 0;
  virtual absl::StatusOr<__u32> GetStat(const std::string& ifname,
                                        const std::string& statname) = 0;
};
}  // namespace gpudirect_tcpxd
#endif
