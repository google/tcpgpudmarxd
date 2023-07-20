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

#include "include/rx_rule_manager.h"

#include <absl/functional/bind_front.h>
#include <absl/hash/hash.h>
#include <absl/log/log.h>
#include <absl/strings/str_format.h>
#include <linux/ethtool.h>
#include <linux/netlink.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "include/flow_steer_ntuple.h"
#include "include/nic_configurator_interface.h"
#include "include/proto_utils.h"
#include "include/socket_helper.h"

namespace gpudirect_tcpxd {

#define ASSIGN_OR_RETURN(var, expression)                    \
  auto var_status = expression;                              \
  if (!var_status.status().ok()) return var_status.status(); \
  auto& var = var_status.value();

#define ASSIGN_VALUE_BY_KEY_IN_MAP_OR_ERROR(value, map, key)                   \
  if (map.find(key) == map.end()) {                                            \
    return absl::NotFoundError(absl::StrFormat("%s not found in " #map, key)); \
  }                                                                            \
  auto& value = map[key];

#define LOG_IF_ERROR(expression)                \
  if (auto status = expression; !status.ok()) { \
    LOG(ERROR) << status;                       \
  }

NicRulesBank::NicRulesBank(const std::string& ifname, int max_rx_rules) {
  ifname_ = ifname;
  max_rx_rules_ = max_rx_rules;
  for (int i = 0; i < max_rx_rules_; ++i) {
    unused_locations_.push(i);
  }
}

int NicRulesBank::UseRule(size_t flow_hash) {
  if (unused_locations_.empty()) return -1;
  int location_id = unused_locations_.front();
  unused_locations_.pop();
  flow_hash_to_location_map_[flow_hash] = location_id;
  return location_id;
}

int NicRulesBank::ReturnRule(size_t flow_hash) {
  if (flow_hash_to_location_map_.find(flow_hash) ==
      flow_hash_to_location_map_.end()) {
    return -1;
  }
  int location_id = flow_hash_to_location_map_[flow_hash];
  flow_hash_to_location_map_.erase(flow_hash);
  unused_locations_.push(location_id);
  return location_id;
}

std::vector<int> NicRulesBank::LocationsInUse() const {
  std::vector<int> location_ids;
  for (const auto& [_, location_id] : flow_hash_to_location_map_) {
    location_ids.push_back(location_id);
  }
  return location_ids;
}

GpuRxqScaler::GpuRxqScaler(const std::string& gpu_pci_addr,
                           const std::vector<int>& queue_ids) {
  gpu_pci_addr_ = gpu_pci_addr;
  for (int qid : queue_ids) {
    qid_qinfo_map_[qid] = {.queue_id = qid, .flow_counts = 0};
    qid_min_heap_.push_back(qid);
  }
}

int GpuRxqScaler::AddFlow(int location_id) {
  std::pop_heap(qid_min_heap_.begin(), qid_min_heap_.end(),
                [this](int a, int b) { return GreaterFlowCounts(a, b); });
  int qid = qid_min_heap_.back();
  qid_qinfo_map_[qid].flow_counts++;
  location_to_queue_map_[location_id] = qid;
  std::push_heap(qid_min_heap_.begin(), qid_min_heap_.end(),
                 [this](int a, int b) { return GreaterFlowCounts(a, b); });
  return qid;
}

void GpuRxqScaler::AddFlow(int location_id, int queue_id) {
  qid_qinfo_map_[queue_id].flow_counts++;
  location_to_queue_map_[location_id] = queue_id;
  std::make_heap(qid_min_heap_.begin(), qid_min_heap_.end(),
                 [this](int a, int b) { return GreaterFlowCounts(a, b); });
}

void GpuRxqScaler::RemoveFlow(int location_id) {
  if (location_to_queue_map_.find(location_id) ==
      location_to_queue_map_.end()) {
    return;
  }
  int qid = location_to_queue_map_[location_id];
  qid_qinfo_map_[qid].flow_counts--;
  location_to_queue_map_.erase(location_id);
  std::make_heap(qid_min_heap_.begin(), qid_min_heap_.end(),
                 [this](int a, int b) { return GreaterFlowCounts(a, b); });
}

std::vector<int> GpuRxqScaler::QueueIds() {
  std::vector<int> qids;
  for (const auto& [qid, _] : qid_qinfo_map_) {
    qids.push_back(qid);
  }
  return qids;
}

RxRuleManager::RxRuleManager(const GpuRxqConfigurationList& config_list,
                             const std::string& prefix,
                             NicConfiguratorInterface* nic_configurator) {
  prefix_ = prefix;
  if (prefix_.back() == '/') {
    prefix_.pop_back();
  }

  nic_configurator_ = nic_configurator;

  for (const auto& gpu_rxq_config : config_list.gpu_rxq_configs()) {
    const auto& ifname = gpu_rxq_config.ifname();

    ifname_to_rules_bank_map_[ifname] =
        std::make_unique<NicRulesBank>(ifname, config_list.max_rx_rules());

    for (const auto& gpu_info : gpu_rxq_config.gpu_infos()) {
      const auto& gpu_pci_addr = gpu_info.gpu_pci_addr();
      if (ifname_to_first_gpu_map_[ifname].empty()) {
        ifname_to_first_gpu_map_[ifname] = gpu_pci_addr;
      }

      gpu_to_ifname_map_[gpu_pci_addr] = ifname;

      gpu_to_rxq_scaler_map_[gpu_info.gpu_pci_addr()] =
          std::make_unique<GpuRxqScaler>(
              gpu_info.gpu_pci_addr(),
              std::vector<int>(gpu_info.queue_ids().begin(),
                               gpu_info.queue_ids().end()));
    }
  }

  std::vector<NetifInfo> netif_infos;
  DiscoverNetif(netif_infos);
  for (const auto& netif_info : netif_infos) {
    ip_to_ifname_map_[AddressToStr(&netif_info.addr)] = netif_info.ifname;
  }
}

absl::Status RxRuleManager::Init() {
  AddFlowSteerRuleServer("rx_rule_manager");
  AddFlowSteerRuleServer("rx_rule_uninstall");
  AddGpuQueueIdsServer();

  for (auto& us_server : us_servers_) {
    if (auto server_status = us_server->Start(); !server_status.ok()) {
      return server_status;
    }
  }

  LOG(INFO) << "Rx Rule Manager server(s) started...";

  return absl::OkStatus();
}

void RxRuleManager::Cleanup() {
  for (auto& us_server : us_servers_) {
    us_server->Stop();
  }
}

void RxRuleManager::AddFlowSteerRuleServer(const std::string& suffix) {
  std::function<absl::Status(const FlowSteerRuleRequest&)> operation =
      suffix == "rx_rule_manager"
          ? absl::bind_front(&RxRuleManager::ConfigFlowSteering, this)
          : absl::bind_front(&RxRuleManager::DeleteFlowSteering, this);

  std::string server_addr = absl::StrFormat("%s/%s", prefix_, suffix);
  LOG(INFO) << absl::StrFormat(
      "Starting FlowSteerRule %s server at %s",
      (suffix == "rx_rule_manager" ? "Install" : "Uninstall"), server_addr);
  us_servers_.emplace_back(std::make_unique<UnixSocketServer>(
      server_addr, [operation](UnixSocketMessage&& request,
                               UnixSocketMessage* response, bool* fin) {
        UnixSocketProto* proto = response->mutable_proto();
        std::string* buffer = proto->mutable_raw_bytes();
        if (!request.has_proto() ||
            !(request.proto().has_flow_steer_rule_request())) {
          std::string err =
              absl::StrFormat("Invalid Argument: %s\n", request.DebugString());
          LOG(ERROR) << err;
          buffer->append(err);
          *fin = true;
          return;
        }

        if (auto status = operation(request.proto().flow_steer_rule_request());
            !status.ok()) {
          *fin = true;
          buffer->append(
              absl::StrFormat("Failed to set flow steering rule, error: %s.",
                              status.ToString()));
          return;
        }
        buffer->append("Ok.");
      }));
}

void RxRuleManager::AddGpuQueueIdsServer() {
  std::string server_addr = absl::StrFormat("%s/queue_ids_by_gpu", prefix_);
  LOG(INFO) << absl::StrFormat("Starting GPU Queue IDs Query Server at %s",
                               server_addr);
  us_servers_.emplace_back(std::make_unique<UnixSocketServer>(
      server_addr, [this](UnixSocketMessage&& request,
                          UnixSocketMessage* response, bool* fin) {
        UnixSocketProto* proto = response->mutable_proto();
        if (!request.has_proto() || !(request.proto().has_queue_id_query())) {
          std::string err =
              absl::StrFormat("Invalid Argument: %s\n", request.DebugString());
          LOG(ERROR) << err;
          std::string* buffer = proto->mutable_raw_bytes();
          buffer->append(err);
          *fin = true;
          return;
        }
        QueueIdQuery query = request.proto().queue_id_query();
        if (gpu_to_rxq_scaler_map_.find(query.gpu_pci_addr()) ==
            gpu_to_rxq_scaler_map_.end()) {
          std::string err =
              absl::StrFormat("GPU Not Found: %s\n", query.gpu_pci_addr());
          LOG(ERROR) << err;
          std::string* buffer = proto->mutable_raw_bytes();
          buffer->append(err);
          *fin = true;
          return;
        }
        QueueIdResponse* qid_resp = proto->mutable_queue_id_response();
        for (int qid :
             gpu_to_rxq_scaler_map_[query.gpu_pci_addr()]->QueueIds()) {
          qid_resp->add_queue_ids(qid);
        }
      }));
}

absl::StatusOr<std::string> RxRuleManager::GetGpuFromFlowSteerRuleRequest(
    const FlowSteerRuleRequest& fsr) {
  if (fsr.has_gpu_pci_addr()) {
    return fsr.gpu_pci_addr();
  }
  const auto& ip = fsr.flow_steer_ntuple().dst().ip_address();
  ASSIGN_VALUE_BY_KEY_IN_MAP_OR_ERROR(ifname, ip_to_ifname_map_, ip);
  ASSIGN_VALUE_BY_KEY_IN_MAP_OR_ERROR(first_gpu, ifname_to_first_gpu_map_,
                                      ifname);
  return first_gpu;
}

absl::Status RxRuleManager::ConfigFlowSteering(
    const FlowSteerRuleRequest& fsr) {
  ASSIGN_OR_RETURN(gpu_pci_addr, GetGpuFromFlowSteerRuleRequest(fsr));

  FlowSteerNtuple ntuple = ConvertProtoToStruct(fsr.flow_steer_ntuple());

  ASSIGN_VALUE_BY_KEY_IN_MAP_OR_ERROR(ifname, gpu_to_ifname_map_, gpu_pci_addr);
  ASSIGN_VALUE_BY_KEY_IN_MAP_OR_ERROR(nic_rules_bank, ifname_to_rules_bank_map_,
                                      ifname);
  ASSIGN_VALUE_BY_KEY_IN_MAP_OR_ERROR(rxq_scaler, gpu_to_rxq_scaler_map_,
                                      gpu_pci_addr);

  size_t flow_hash = GetFlowHash(ntuple);

  if (nic_rules_bank->RuleExists(flow_hash)) {
    return absl::OkStatus();
  }

  int location_id = nic_rules_bank->UseRule(flow_hash);

  if (location_id < 0) {
    return absl::ResourceExhaustedError(
        absl::StrFormat("No more rx rules available for NIC: %s", ifname));
  }

  int queue_id = -1;

  if (fsr.has_queue_id() && rxq_scaler->valid_queue_id(fsr.queue_id())) {
    queue_id = fsr.queue_id();
    rxq_scaler->AddFlow(location_id, queue_id);
  } else {
    queue_id = rxq_scaler->AddFlow(location_id);
  }

  if (auto status =
          nic_configurator_->AddFlow(ifname, ntuple, queue_id, location_id);
      !status.ok()) {
    nic_rules_bank->ReturnRule(flow_hash);
    rxq_scaler->RemoveFlow(location_id);
    return status;
  }
  return absl::OkStatus();
}

absl::Status RxRuleManager::DeleteFlowSteering(
    const FlowSteerRuleRequest& fsr) {
  ASSIGN_OR_RETURN(gpu_pci_addr, GetGpuFromFlowSteerRuleRequest(fsr));

  FlowSteerNtuple ntuple = ConvertProtoToStruct(fsr.flow_steer_ntuple());

  ASSIGN_VALUE_BY_KEY_IN_MAP_OR_ERROR(ifname, gpu_to_ifname_map_, gpu_pci_addr);
  ASSIGN_VALUE_BY_KEY_IN_MAP_OR_ERROR(nic_rules_bank, ifname_to_rules_bank_map_,
                                      ifname);
  ASSIGN_VALUE_BY_KEY_IN_MAP_OR_ERROR(rxq_scaler, gpu_to_rxq_scaler_map_,
                                      gpu_pci_addr);

  size_t flow_hash = GetFlowHash(ntuple);
  int location_id = nic_rules_bank->ReturnRule(flow_hash);
  rxq_scaler->RemoveFlow(location_id);
  LOG_IF_ERROR(nic_configurator_->RemoveFlow(ifname, location_id));
  return absl::OkStatus();
}

size_t RxRuleManager::GetFlowHash(const struct FlowSteerNtuple& ntuple) {
  if (ntuple.flow_type == TCP_V4_FLOW) {
    return absl::HashOf(ntuple.src_sin.sin_port, ntuple.dst_sin.sin_port,
                        ntuple.src_sin.sin_addr.s_addr,
                        ntuple.dst_sin.sin_addr.s_addr);
  } else if (ntuple.flow_type == TCP_V6_FLOW) {
    return absl::HashOf(ntuple.src_sin6.sin6_port, ntuple.dst_sin6.sin6_port,
                        ntuple.src_sin6.sin6_addr.__in6_u.__u6_addr32[0],
                        ntuple.src_sin6.sin6_addr.__in6_u.__u6_addr32[1],
                        ntuple.src_sin6.sin6_addr.__in6_u.__u6_addr32[2],
                        ntuple.src_sin6.sin6_addr.__in6_u.__u6_addr32[3],
                        ntuple.dst_sin6.sin6_addr.__in6_u.__u6_addr32[0],
                        ntuple.dst_sin6.sin6_addr.__in6_u.__u6_addr32[1],
                        ntuple.dst_sin6.sin6_addr.__in6_u.__u6_addr32[2],
                        ntuple.dst_sin6.sin6_addr.__in6_u.__u6_addr32[3]);
  }
  return 0;
}

#undef ASSIGN_OR_RETURN
#undef ASSIGN_VALUE_BY_KEY_IN_MAP_OR_ERROR
#undef LOG_IF_ERROR
}  // namespace gpudirect_tcpxd
