#include "include/rx_rule_manager.h"

#include <linux/ethtool.h>
#include <linux/netlink.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <absl/log/log.h>
#include <absl/functional/bind_front.h>
#include <absl/hash/hash.h>
#include <absl/strings/str_format.h>

#include "include/flow_steer_ntuple.h"
#include "include/proto_utils.h"
#include "include/nic_configurator_interface.h"

namespace tcpdirect {

RxRuleManager::RxRuleManager(const GpuRxqConfigurationList& config_list,
                             const std::string& prefix,
                             NicConfiguratorInterface* nic_configurator) {
  max_rx_rules_ = config_list.max_rx_rules();
  for (int i = 0; i < max_rx_rules_; ++i) {
    unused_locations_.push(i);
  }
  tcpd_queue_size_ = config_list.tcpd_queue_size();
  rss_set_size_ = config_list.rss_set_size();
  for (const auto& config : config_list.gpu_rxq_configs()) {
    ifnames_.push_back(config.ifname());
  }
  prefix_ = prefix;
  if (prefix_.back() == '/') {
    prefix_.pop_back();
  }
  nic_configurator_ = nic_configurator;
}

absl::Status RxRuleManager::Init() {
  AddUnixSocketServer("rx_rule_manager");
  AddUnixSocketServer("rx_rule_uninstall");

  for (auto& us_server : us_servers_) {
    if (auto server_status = us_server->Start(); !server_status.ok()) {
      return server_status;
    }
  }

  LOG(INFO) << "Rx Rule Manager server(s) started...";

  return absl::OkStatus();
}

void RxRuleManager::Cleanup() {
  for (auto& [_, loc] : flow_hash_to_location_map_) {
    for (const auto& ifname : ifnames_) {
      if (auto status = nic_configurator_->RemoveFlow(ifname, loc);
          !status.ok()) {
        LOG(ERROR) << status;
      }
    }
  }
  for (auto& us_server : us_servers_) {
    us_server->Stop();
  }
}

void RxRuleManager::AddUnixSocketServer(const std::string& suffix) {
  std::function<absl::Status(const struct FlowSteerNtuple)> operation =
      suffix == "rx_rule_manager"
          ? absl::bind_front(&RxRuleManager::ConfigFlowSteering, this)
          : absl::bind_front(&RxRuleManager::DeleteFlowSteering, this);

  us_servers_.emplace_back(std::make_unique<UnixSocketServer>(
      absl::StrFormat("%s/%s", prefix_, suffix),
      [operation](UnixSocketMessage&& request, UnixSocketMessage* response,
                  bool* fin) {
        UnixSocketProto* proto = response->mutable_proto();
        std::string* buffer = proto->mutable_raw_bytes();
       if (!request.has_proto() ||
            !(request.proto().has_flow_steer_rule_request())) {
          LOG(ERROR) << "Expecting proto format request. "
                     << request.DebugString();
          buffer->append("Error.\n\nExpecting proto format request.\n");
          *fin = true;
          return;
        }

        FlowSteerNtuple ntuple = ConvertProtoToStruct(
            request.proto().flow_steer_rule_request().flow_steer_ntuple());

        if (auto status = operation(ntuple); !status.ok()) {
          *fin = true;
          buffer->append("Failed to set flow steering rule.");
          return;
        }
        buffer->append("Ok.");
      }));
}

absl::Status RxRuleManager::ConfigFlowSteering(
    const struct FlowSteerNtuple &ntuple) {
  if (rss_set_size_ < 0) {
    return absl::FailedPreconditionError(
        "RSS is not set yet for non-tcpdirect traffic!");
  }

  if (unused_locations_.empty()) {
    return absl::ResourceExhaustedError("No rule location available.");
  }

  size_t flow_hash = GetFlowHash(ntuple);

  if (flow_hash_to_location_map_.find(flow_hash) !=
      flow_hash_to_location_map_.end()) {
    return absl::OkStatus();
  }

  int location_id = unused_locations_.front();
  int queue_id = LocationToQueueId(location_id);
  std::vector<std::string> succeeded_ifnames;
  for (const auto& ifname : ifnames_) {
    if (auto status =
            nic_configurator_->AddFlow(ifname, ntuple, queue_id, location_id);
        !status.ok()) {
      for (const auto& succeeded_ifname : succeeded_ifnames) {
        if (auto remove_status =
                nic_configurator_->RemoveFlow(succeeded_ifname, location_id);
            !remove_status.ok()) {
          LOG(ERROR) << status;
        }
      }
      return status;
    }
    succeeded_ifnames.push_back(ifname);
  }
  unused_locations_.pop();
  flow_hash_to_location_map_[flow_hash] = location_id;
  return absl::OkStatus();
}

absl::Status RxRuleManager::DeleteFlowSteering(
    const struct FlowSteerNtuple &ntuple) {
  size_t flow_hash = GetFlowHash(ntuple);

  if (flow_hash_to_location_map_.find(flow_hash) ==
      flow_hash_to_location_map_.end()) {
    return absl::NotFoundError("Flow does not exist.");
  }

  int location_id = flow_hash_to_location_map_[flow_hash];
  for (const auto& ifname : ifnames_) {
    if (auto status = nic_configurator_->RemoveFlow(ifname, location_id);
        !status.ok()) {
      LOG(ERROR) << status;
    }
  }
  flow_hash_to_location_map_.erase(flow_hash);
  unused_locations_.push(location_id);
  return absl::OkStatus();
}

size_t RxRuleManager::GetFlowHash(const struct FlowSteerNtuple &ntuple) {
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

int RxRuleManager::LocationToQueueId(int location) {
  if (rss_set_size_ < 0 || tcpd_queue_size_ < 0) return 0;
  if (tcpd_queue_size_ < 1) return 0;
  return (location % tcpd_queue_size_) + rss_set_size_;
}

}  // namespace tcpdirect
