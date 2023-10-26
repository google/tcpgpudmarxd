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

#include <absl/flags/flag.h>
#include <absl/flags/internal/flag.h>
#include <absl/functional/bind_front.h>
#include <absl/hash/hash.h>
#include <absl/log/log.h>
#include <absl/status/status.h>
#include <absl/strings/str_format.h>
#include <absl/synchronization/mutex.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>
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
#include <thread>
#include <vector>

#include "code.pb.h"
#include "include/flow_steer_ntuple.h"
#include "include/nic_configurator_interface.h"
#include "include/proto_utils.h"
#include "include/socket_helper.h"
#include "proto/gpu_rxq_configuration.pb.h"
#include "proto/unix_socket_message.pb.h"
#include "telemetry/telemetry_interface.h"

ABSL_FLAG(bool, async_handler, true,
          "Allow RxDM to process flow steering rules asynchronously.");

namespace gpudirect_tcpxd {

#define ASSIGN_OR_RETURN(var, expression)                    \
  auto var_status = expression;                              \
  if (!var_status.status().ok()) return var_status.status(); \
  auto& var = var_status.value();

#define ASSIGN_VALUE_BY_KEY_IN_MAP_OR_ERROR(value, map, key)                   \
  const auto it_##map = map.find(key);                                         \
  if (it_##map == map.end()) {                                                 \
    return absl::NotFoundError(absl::StrFormat("%s not found in " #map, key)); \
  }                                                                            \
  const auto& value = it_##map->second;

#define LOG_IF_ERROR(expression)                \
  if (auto status = expression; !status.ok()) { \
    LOG(ERROR) << status;                       \
  }

namespace {
// Use src/dst ip and port to construct hash for each flow
size_t GetFlowHash(const struct FlowSteerNtuple& ntuple) {
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

UnixSocketMessage ConstructErrorResponse(const absl::Status& status) {
  UnixSocketMessage response;
  UnixSocketProto* proto = response.mutable_proto();
  std::string* buffer = proto->mutable_raw_bytes();
  proto->mutable_status()->set_code(status.raw_code());
  proto->mutable_status()->set_message(status.ToString());
  buffer->append(absl::StrFormat(
      "Failed to update flow steering rule, error: %s.", status.ToString()));
  return response;
}

bool InvalidFlowSteerRequest(const UnixSocketMessage& request) {
  return !request.has_proto() ||
         !(request.proto().has_flow_steer_rule_request());
}
}  // namespace

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
  absl::MutexLock lock(&mu_);
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
  absl::MutexLock lock(&mu_);
  qid_qinfo_map_[queue_id].flow_counts++;
  location_to_queue_map_[location_id] = queue_id;
  std::make_heap(qid_min_heap_.begin(), qid_min_heap_.end(),
                 [this](int a, int b) { return GreaterFlowCounts(a, b); });
}

void GpuRxqScaler::RemoveFlow(int location_id) {
  absl::MutexLock lock(&mu_);
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
  absl::MutexLock lock(&mu_);
  std::vector<int> qids;
  for (const auto& [qid, _] : qid_qinfo_map_) {
    qids.push_back(qid);
  }
  return qids;
}

bool GpuRxqScaler::GreaterFlowCounts(int qa, int qb) {
  return qid_qinfo_map_[qa].flow_counts > qid_qinfo_map_[qb].flow_counts;
}

NicWorker::NicWorker(const std::string& ifname, int max_rx_rules,
                     NicConfiguratorInterface* nic_configurator,
                     const GpuRxqConfiguration& gpu_rxq_config,
                     FlowSteerRuleManagerTelemetry* telemetry)
    : ifname_(ifname),
      rules_bank_(ifname, max_rx_rules),
      nic_configurator_(nic_configurator),
      telemetry_(telemetry),
      stopped_(false) {
  for (const auto& gpu_info : gpu_rxq_config.gpu_infos()) {
    gpu_to_rxq_scaler_map_[gpu_info.gpu_pci_addr()] =
        std::make_unique<GpuRxqScaler>(
            gpu_info.gpu_pci_addr(),
            std::vector<int>(gpu_info.queue_ids().begin(),
                             gpu_info.queue_ids().end()));
  }

  thread_ = std::thread(&NicWorker::NicWorkerLoop, this);
}

void NicWorker::AddWorkingItem(
    FlowSteerRuleRequest&& request, const std::string& gpu_pci_addr,
    bool installing, absl::Time received_time,
    std::function<void(UnixSocketMessage&&, bool)>&& callback) {
  absl::MutexLock lock(&mu_);
  request_queue_.emplace(std::make_unique<NicWorkingItem>(
      std::move(request), gpu_pci_addr, installing, received_time, callback));
}

void NicWorker::NicWorkerLoop() {
  while (true) {
    if (stopped_) {
      return;
    }
    std::unique_ptr<NicWorkingItem> item;
    {
      mu_.Lock();
      if (request_queue_.empty()) {
        mu_.Unlock();
        absl::SleepFor(absl::Milliseconds(10));
        continue;
      }
      item = std::move(request_queue_.front());
      request_queue_.pop();
      mu_.Unlock();
    }

    ProcessRequest(item.get());
  }
}

absl::StatusOr<QueueIdResponse> NicWorker::GetQueueIds(
    const std::string& gpu_pci_addr) {
  absl::MutexLock lock(&mu_);
  ASSIGN_VALUE_BY_KEY_IN_MAP_OR_ERROR(rxq_scaler, gpu_to_rxq_scaler_map_,
                                      gpu_pci_addr);

  QueueIdResponse qid_resp;
  for (int qid : rxq_scaler->QueueIds()) {
    qid_resp.add_queue_ids(qid);
  }
  return qid_resp;
}

void NicWorker::ProcessRequest(NicWorkingItem* item) {
  auto status = item->installing
                    ? ConfigFlowSteering(item->request, item->gpu_pci_addr)
                    : DeleteFlowSteering(item->request, item->gpu_pci_addr);

  UnixSocketMessage response;

  bool fin = false;
  if (!status.ok()) {
    LOG(ERROR) << status;
    response = ConstructErrorResponse(status);
    if (item->installing) {
      telemetry_->IncrementInstallFailure();
      telemetry_->IncrementFailureAndCause(status.ToString());
    }
  } else {
    if (item->installing) {
      telemetry_->IncrementInstallSuccess();
    } else {
      telemetry_->IncrementUninstallSuccess();
    }
    UnixSocketProto* proto = response.mutable_proto();
    std::string* buffer = proto->mutable_raw_bytes();
    buffer->append("Ok.");
  }

  item->callback(std::move(response), fin);
  telemetry_->AddLatency(absl::Now() - item->received_time);
}

absl::Status NicWorker::ConfigFlowSteering(const FlowSteerRuleRequest& fsr,
                                           const std::string& gpu_pci_addr) {
  ASSIGN_VALUE_BY_KEY_IN_MAP_OR_ERROR(rxq_scaler, gpu_to_rxq_scaler_map_,
                                      gpu_pci_addr);
  FlowSteerNtuple ntuple = ConvertProtoToStruct(fsr.flow_steer_ntuple());
  size_t flow_hash = GetFlowHash(ntuple);

  if (rules_bank_.RuleExists(flow_hash)) {
    return absl::OkStatus();
  }

  int location_id = rules_bank_.UseRule(flow_hash);

  if (location_id < 0) {
    return absl::ResourceExhaustedError(
        absl::StrFormat("No more rx rules available for NIC: %s", ifname_));
  }

  int queue_id = -1;

  if (fsr.has_queue_id() && rxq_scaler->valid_queue_id(fsr.queue_id())) {
    queue_id = fsr.queue_id();
    rxq_scaler->AddFlow(location_id, queue_id);
  } else {
    queue_id = rxq_scaler->AddFlow(location_id);
  }

  telemetry_->IncrementRulesInstalledOnQueues(queue_id);

  if (auto status =
          nic_configurator_->AddFlow(ifname_, ntuple, queue_id, location_id);
      !status.ok()) {
    rules_bank_.ReturnRule(flow_hash);
    rxq_scaler->RemoveFlow(location_id);
    return status;
  }
  return absl::OkStatus();
}

absl::Status NicWorker::DeleteFlowSteering(const FlowSteerRuleRequest& fsr,
                                           const std::string& gpu_pci_addr) {
  ASSIGN_VALUE_BY_KEY_IN_MAP_OR_ERROR(rxq_scaler, gpu_to_rxq_scaler_map_,
                                      gpu_pci_addr);
  FlowSteerNtuple ntuple = ConvertProtoToStruct(fsr.flow_steer_ntuple());

  size_t flow_hash = GetFlowHash(ntuple);
  int location_id = rules_bank_.ReturnRule(flow_hash);
  rxq_scaler->RemoveFlow(location_id);
  LOG_IF_ERROR(nic_configurator_->RemoveFlow(ifname_, location_id));
  return absl::OkStatus();
}

RxRuleManager::RxRuleManager(const GpuRxqConfigurationList& config_list,
                             const std::string& prefix,
                             NicConfiguratorInterface* nic_configurator) {
  prefix_ = prefix;
  if (prefix_.back() == '/') {
    prefix_.pop_back();
  }

  nic_configurator_ = nic_configurator;
  telemetry_.Start();

  for (const auto& gpu_rxq_config : config_list.gpu_rxq_configs()) {
    const auto& ifname = gpu_rxq_config.ifname();

    for (const auto& gpu_info : gpu_rxq_config.gpu_infos()) {
      const auto& gpu_pci_addr = gpu_info.gpu_pci_addr();
      if (ifname_to_first_gpu_map_[ifname].empty()) {
        ifname_to_first_gpu_map_[ifname] = gpu_pci_addr;
      }

      gpu_to_ifname_map_[gpu_pci_addr] = ifname;
    }

    if_workers_[ifname] = std::make_unique<NicWorker>(
        ifname, config_list.max_rx_rules(), nic_configurator_, gpu_rxq_config,
        &telemetry_);
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
  std::string server_addr = absl::StrFormat("%s/%s", prefix_, suffix);
  LOG(INFO) << absl::StrFormat(
      "Starting FlowSteerRule %s server at %s",
      (suffix == "rx_rule_manager" ? "Install" : "Uninstall"), server_addr);

  if (absl::GetFlag(FLAGS_async_handler)) {
    LOG(INFO) << "Using parallel rule installing across NICs";
    us_servers_.emplace_back(std::make_unique<UnixSocketServer>(
        server_addr, absl::bind_front(&RxRuleManager::FlowSteerRuleAsyncHandler,
                                      this, suffix)));
  } else {
    LOG(INFO) << "Using single-threaded rule installing across NICs";
    us_servers_.emplace_back(std::make_unique<UnixSocketServer>(
        server_addr, absl::bind_front(&RxRuleManager::FlowSteerRuleSyncHandler,
                                      this, suffix)));
  }
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
          proto->mutable_status()->set_code(
              google::rpc::Code::INVALID_ARGUMENT);
          proto->mutable_status()->set_message(err);
          *fin = true;
          buffer->append(err);
          return;
        }
        QueueIdQuery query = request.proto().queue_id_query();
        auto qid_resp = GetQueueIds(query.gpu_pci_addr());
        if (!qid_resp.ok()) {
          *response = ConstructErrorResponse(qid_resp.status());
          *fin = true;
          return;
        }
        proto->mutable_queue_id_response()->Swap(&(*qid_resp));
      }));
}

absl::StatusOr<std::string> RxRuleManager::GetGpuFromFlowSteerRuleRequest(
    const FlowSteerRuleRequest& fsr) const {
  if (fsr.has_gpu_pci_addr()) {
    return fsr.gpu_pci_addr();
  }
  const auto& ip = fsr.flow_steer_ntuple().dst().ip_address();
  ASSIGN_VALUE_BY_KEY_IN_MAP_OR_ERROR(ifname, ip_to_ifname_map_, ip);
  ASSIGN_VALUE_BY_KEY_IN_MAP_OR_ERROR(first_gpu, ifname_to_first_gpu_map_,
                                      ifname);
  return first_gpu;
}

absl::StatusOr<QueueIdResponse> RxRuleManager::GetQueueIds(
    const std::string& gpu_pci_addr) {
  ASSIGN_VALUE_BY_KEY_IN_MAP_OR_ERROR(ifname, gpu_to_ifname_map_, gpu_pci_addr);

  ASSIGN_VALUE_BY_KEY_IN_MAP_OR_ERROR(nic_worker, if_workers_, ifname);

  return nic_worker->GetQueueIds(gpu_pci_addr);
}

absl::Status RxRuleManager::DispatchRequest(
    FlowSteerRuleRequest&& fsr, bool installing, absl::Time received_time,
    std::function<void(UnixSocketMessage&&, bool)>&& callback) const {
  ASSIGN_OR_RETURN(gpu_pci_addr, GetGpuFromFlowSteerRuleRequest(fsr));
  FlowSteerNtuple ntuple = ConvertProtoToStruct(fsr.flow_steer_ntuple());

  ASSIGN_VALUE_BY_KEY_IN_MAP_OR_ERROR(ifname, gpu_to_ifname_map_, gpu_pci_addr);

  ASSIGN_VALUE_BY_KEY_IN_MAP_OR_ERROR(nic_worker, if_workers_, ifname);

  nic_worker->AddWorkingItem(std::move(fsr), gpu_pci_addr, installing,
                             received_time, std::move(callback));

  return absl::OkStatus();
}

absl::Status RxRuleManager::ProcessRequest(FlowSteerRuleRequest&& fsr,
                                           bool installing,
                                           absl::Time received_time) {
  ASSIGN_OR_RETURN(gpu_pci_addr, GetGpuFromFlowSteerRuleRequest(fsr));
  FlowSteerNtuple ntuple = ConvertProtoToStruct(fsr.flow_steer_ntuple());

  ASSIGN_VALUE_BY_KEY_IN_MAP_OR_ERROR(ifname, gpu_to_ifname_map_, gpu_pci_addr);

  ASSIGN_VALUE_BY_KEY_IN_MAP_OR_ERROR(nic_worker, if_workers_, ifname);

  auto status = installing ? nic_worker->ConfigFlowSteering(fsr, gpu_pci_addr)
                           : nic_worker->DeleteFlowSteering(fsr, gpu_pci_addr);

  return status;
}

void RxRuleManager::FlowSteerRuleSyncHandler(std::string suffix,
                                             UnixSocketMessage&& request,
                                             UnixSocketMessage* response,
                                             bool* fin) {
  absl::Time start = absl::Now();
  telemetry_.IncrementRequests();

  if (InvalidFlowSteerRequest(request)) {
    *response = ConstructErrorResponse(
        absl::InvalidArgumentError("Invalid Flow Steering Request"));
    if (suffix == "rx_rule_manager") {
      telemetry_.IncrementInstallFailure();
      telemetry_.IncrementFailureAndCause("Invalid Argument.");
    }
    *fin = true;
    return;
  }

  auto status = ProcessRequest(
      std::move(*request.mutable_proto()->mutable_flow_steer_rule_request()),
      suffix == "rx_rule_manager", start);
  if (!status.ok()) {
    auto response = ConstructErrorResponse(status);
    if (suffix == "rx_rule_manager") {
      telemetry_.IncrementInstallFailure();
      telemetry_.IncrementFailureAndCause(status.ToString());
    }
    *fin = true;
    return;
  }

  UnixSocketProto* proto = response->mutable_proto();
  std::string* buffer = proto->mutable_raw_bytes();

  if (suffix == "rx_rule_manager") {
    telemetry_.IncrementInstallSuccess();
  } else {
    telemetry_.IncrementUninstallSuccess();
  }
  buffer->append("Ok.");
  telemetry_.AddLatency(absl::Now() - start);
}

void RxRuleManager::FlowSteerRuleAsyncHandler(
    std::string suffix, UnixSocketMessage&& request,
    std::function<void(UnixSocketMessage&&, bool)> callback) {
  absl::Time start = absl::Now();
  telemetry_.IncrementRequests();

  if (InvalidFlowSteerRequest(request)) {
    auto response = ConstructErrorResponse(
        absl::InvalidArgumentError("Invalid Flow Steering Request"));
    if (suffix == "rx_rule_manager") {
      telemetry_.IncrementInstallFailure();
      telemetry_.IncrementFailureAndCause("Invalid Argument.");
    }
    callback(std::move(response), /*fin=*/true);
    return;
  }

  auto status = DispatchRequest(
      std::move(*request.mutable_proto()->mutable_flow_steer_rule_request()),
      suffix == "rx_rule_manager", start, std::move(callback));
  if (!status.ok()) {
    auto response = ConstructErrorResponse(status);
    if (suffix == "rx_rule_manager") {
      telemetry_.IncrementInstallFailure();
      telemetry_.IncrementFailureAndCause(status.ToString());
    }
    callback(std::move(response), /*fin=*/true);
  }
}

#undef ASSIGN_OR_RETURN
#undef ASSIGN_VALUE_BY_KEY_IN_MAP_OR_ERROR
#undef LOG_IF_ERROR
}  // namespace gpudirect_tcpxd