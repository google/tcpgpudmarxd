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

#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_RX_RULE_MANAGER_H_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_RX_RULE_MANAGER_H_

#include <absl/container/flat_hash_map.h>
#include <absl/status/status.h>
#include <absl/status/statusor.h>
#include <net/if.h>

#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "gtest/gtest_prod.h"
#include "include/flow_steer_ntuple.h"
#include "include/nic_configurator_interface.h"
#include "include/unix_socket_server.h"
#include "proto/gpu_rxq_configuration.pb.h"
#include "proto/unix_socket_message.pb.h"

namespace tcpdirect {

struct QueueInfo {
  int queue_id;
  int flow_counts;
};

enum RxRuleServerType {
  INSTALL,
  UNINSTALL,
};

class NicRulesBank {
 public:
  NicRulesBank(const std::string &ifname, int max_rx_rules);
  bool Availble() const { return unused_locations_.size() > 0; }
  bool RuleExists(size_t flow_hash) {
    return flow_hash_to_location_map_.find(flow_hash) !=
           flow_hash_to_location_map_.end();
  }
  int UseRule(size_t flow_hash);
  int ReturnRule(size_t flow_hash);
  std::vector<int> LocationsInUse() const;

 private:
  std::string ifname_;
  int max_rx_rules_;
  std::queue<int> unused_locations_;
  absl::flat_hash_map<int, int> flow_hash_to_location_map_;
};

class GpuRxqScaler {
 public:
  GpuRxqScaler(const std::string &gpu_pci_addr,
               const std::vector<int> &queue_ids);
  int AddFlow(int location_id);
  void AddFlow(int location_id, int queue_id);
  void RemoveFlow(int location_id);
  bool valid() {
    return qid_min_heap_.size() == qid_qinfo_map_.size() &&
           qid_min_heap_.size() > 0;
  }
  bool valid_queue_id(int queue_id) {
    return qid_qinfo_map_.find(queue_id) != qid_qinfo_map_.end();
  }
  std::vector<int> QueueIds();

 private:
  FRIEND_TEST(GpuRxqScalerTest, AddFlowWithQueueIdSuccess);
  FRIEND_TEST(GpuRxqScalerTest, AddFlowWithoutQueueIdSuccess);
  FRIEND_TEST(GpuRxqScalerTest, RemoveFlowSuccess);
  FRIEND_TEST(GpuRxqScalerTest, RemoveFlowFail);

  bool GreaterFlowCounts(int qa, int qb) {
    return qid_qinfo_map_[qa].flow_counts > qid_qinfo_map_[qb].flow_counts;
  }
  std::string gpu_pci_addr_;
  std::vector<int> qid_min_heap_;
  absl::flat_hash_map<int, QueueInfo> qid_qinfo_map_;
  absl::flat_hash_map<int, int> location_to_queue_map_;
};

class RxRuleManager {
 public:
  explicit RxRuleManager(const GpuRxqConfigurationList &config_list,
                         const std::string &prefix,
                         NicConfiguratorInterface *nic_configurator);
  ~RxRuleManager() { Cleanup(); }
  absl::Status Init();
  void Cleanup();

 private:
  FRIEND_TEST(RxRuleManagerConstructorTest, InitGpuRxqConfigSuccess);
  void AddFlowSteerRuleServer(const std::string &suffix);
  void AddGpuQueueIdsServer();
  absl::StatusOr<std::string> GetGpuFromFlowSteerRuleRequest(
      const FlowSteerRuleRequest &fsr);
  absl::Status ConfigFlowSteering(const FlowSteerRuleRequest &fsr);
  absl::Status DeleteFlowSteering(const FlowSteerRuleRequest &fsr);
  size_t GetFlowHash(const struct FlowSteerNtuple &ntuple);
  std::string prefix_;
  NicConfiguratorInterface *nic_configurator_;
  std::unordered_map<std::string, std::unique_ptr<NicRulesBank>>
      ifname_to_rules_bank_map_;
  std::unordered_map<std::string, std::string> ifname_to_first_gpu_map_;
  std::unordered_map<std::string, std::string> gpu_to_ifname_map_;
  std::unordered_map<std::string, std::unique_ptr<GpuRxqScaler>>
      gpu_to_rxq_scaler_map_;
  std::unordered_map<std::string, std::string> ip_to_ifname_map_;
  std::vector<std::unique_ptr<UnixSocketServer>> us_servers_;
};
}  // namespace tcpdirect
#endif
