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

#include <absl/base/thread_annotations.h>
#include <absl/container/flat_hash_map.h>
#include <absl/status/status.h>
#include <absl/status/statusor.h>
#include <absl/synchronization/mutex.h>
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
#include "proto/unix_socket_proto.pb.h"
#include "telemetry/rx_rule_manager_telemetry.h"
#include "telemetry/telemetry_interface.h"

namespace gpudirect_tcpxd {

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
  NicRulesBank(const std::string& ifname, int max_rx_rules);
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
  GpuRxqScaler(const std::string& gpu_pci_addr,
               const std::vector<int>& queue_ids);
  int AddFlow(int location_id);
  void AddFlow(int location_id, int queue_id);
  void RemoveFlow(int location_id);
  bool valid() {
    return qid_min_heap_.size() == qid_qinfo_map_.size() &&
           qid_min_heap_.size() > 0;
  }
  bool valid_queue_id(int queue_id) {
    absl::MutexLock lock(&mu_);
    return qid_qinfo_map_.find(queue_id) != qid_qinfo_map_.end();
  }
  std::vector<int> QueueIds();

 private:
  FRIEND_TEST(GpuRxqScalerTest, AddFlowWithQueueIdSuccess);
  FRIEND_TEST(GpuRxqScalerTest, AddFlowWithoutQueueIdSuccess);
  FRIEND_TEST(GpuRxqScalerTest, RemoveFlowSuccess);
  FRIEND_TEST(GpuRxqScalerTest, RemoveFlowFail);

  bool GreaterFlowCounts(int qa, int qb) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  std::string gpu_pci_addr_;
  std::vector<int> qid_min_heap_;
  absl::flat_hash_map<int, QueueInfo> qid_qinfo_map_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<int, int> location_to_queue_map_;
  absl::Mutex mu_;
};

class NicWorker {
 public:
  struct NicWorkingItem {
    NicWorkingItem(FlowSteerRuleRequest&& request,
                   const std::string& gpu_pci_addr, bool installing,
                   absl::Time received_time,
                   std::function<void(UnixSocketMessage&&, bool)> callback)
        : request(std::move(request)),
          gpu_pci_addr(gpu_pci_addr),
          installing(installing),
          received_time(received_time),
          callback(callback) {}
    FlowSteerRuleRequest request;
    std::string gpu_pci_addr;
    bool installing;
    absl::Time received_time;
    std::function<void(UnixSocketMessage&&, bool)> callback;
  };
  NicWorker(const std::string& ifname, int max_rx_rules,
            NicConfiguratorInterface* nic_configurator,
            const GpuRxqConfiguration& gpu_rxq_config,
            FlowSteerRuleManagerTelemetry* telemetry);
  ~NicWorker() {
    stopped_ = true;
    thread_.join();
  }
  void AddWorkingItem(
      FlowSteerRuleRequest&& request, const std::string& gpu_pci_addr,
      bool installing, absl::Time received_time,
      std::function<void(UnixSocketMessage&&, bool)>&& callback);
  absl::StatusOr<QueueIdResponse> GetQueueIds(const std::string& gpu_pci_addr);

  absl::Status ConfigFlowSteering(const FlowSteerRuleRequest& fsr,
                                  const std::string& gpu_pci_addr);
  absl::Status DeleteFlowSteering(const FlowSteerRuleRequest& fsr,
                                  const std::string& gpu_pci_addr);

 private:
  void NicWorkerLoop();
  void ProcessRequest(NicWorkingItem* item);

  std::string ifname_;
  NicRulesBank rules_bank_;
  NicConfiguratorInterface* nic_configurator_;
  std::unordered_map<std::string, std::unique_ptr<GpuRxqScaler>>
      gpu_to_rxq_scaler_map_ ABSL_GUARDED_BY(mu_);
  FlowSteerRuleManagerTelemetry* telemetry_;

  std::thread thread_;
  std::atomic<bool> stopped_;
  std::queue<std::unique_ptr<NicWorkingItem>> request_queue_
      ABSL_GUARDED_BY(mu_);
  absl::Mutex mu_;
};

class RxRuleManager {
 public:
  explicit RxRuleManager(const GpuRxqConfigurationList& config_list,
                         const std::string& prefix,
                         NicConfiguratorInterface* nic_configurator);
  ~RxRuleManager() { Cleanup(); }
  absl::Status Init();
  void Cleanup();

 private:
  FRIEND_TEST(RxRuleManagerConstructorTest, InitGpuRxqConfigSuccess);
  void AddFlowSteerRuleServer(const std::string& suffix);
  void AddGpuQueueIdsServer();
  absl::StatusOr<std::string> GetGpuFromFlowSteerRuleRequest(
      const FlowSteerRuleRequest& fsr) const;
  absl::StatusOr<QueueIdResponse> GetQueueIds(const std::string& gpu_pci_addr);

  // Dispatch request to the correct nic worker
  absl::Status DispatchRequest(
      FlowSteerRuleRequest&& fsr, bool installing, absl::Time received_time,
      std::function<void(UnixSocketMessage&&, bool)>&& callback) const;
  absl::Status ProcessRequest(
      FlowSteerRuleRequest&& fsr, bool installing, absl::Time received_time) ;

  // Synchronously process the request
  void FlowSteerRuleSyncHandler(std::string suffix, UnixSocketMessage&& request,
                                UnixSocketMessage* response, bool* fin);
  // Asynchronously process the request and invoke callback after finish.
  void FlowSteerRuleAsyncHandler(
      std::string suffix, UnixSocketMessage&& request,
      std::function<void(UnixSocketMessage&&, bool)> callback);

  std::string prefix_;
  NicConfiguratorInterface* nic_configurator_;
  std::unordered_map<std::string, std::string> ifname_to_first_gpu_map_;
  std::unordered_map<std::string, std::string> gpu_to_ifname_map_;

  std::unordered_map<std::string, std::string> ip_to_ifname_map_;
  std::vector<std::unique_ptr<UnixSocketServer>> us_servers_;
  FlowSteerRuleManagerTelemetry telemetry_;

  absl::flat_hash_map<std::string, std::unique_ptr<NicWorker>> if_workers_;
};
}  // namespace gpudirect_tcpxd
#endif