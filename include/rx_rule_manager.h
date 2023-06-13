#ifndef _THIRD_PARTY_TCPDIRECT_RX_MANAGER_RX_RULE_MANAGER_H_
#define _THIRD_PARTY_TCPDIRECT_RX_MANAGER_RX_RULE_MANAGER_H_

#include <net/if.h>

#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "include/flow_steer_ntuple.h"
#include "include/nic_configurator_interface.h"
#include "include/unix_socket_server.h"
#include "proto/gpu_rxq_configuration.pb.h"
#include "proto/unix_socket_message.pb.h"
#include <absl/container/flat_hash_map.h>
#include <absl/status/status.h>

namespace tcpdirect {

class RxRuleManager {
 public:
  explicit RxRuleManager(const GpuRxqConfigurationList& config_list,
                         const std::string& prefix,
                         NicConfiguratorInterface* nic_configurator);
  ~RxRuleManager() { Cleanup(); }
  absl::Status Init();
  void Cleanup();

 private:
  void AddUnixSocketServer(const std::string& suffix);
  absl::Status ConfigFlowSteering(const struct FlowSteerNtuple& ntuple);
  absl::Status DeleteFlowSteering(const struct FlowSteerNtuple& ntuple);
  size_t GetFlowHash(const struct FlowSteerNtuple& ntuple);
  int LocationToQueueId(int location);
  int max_rx_rules_{-1};
  int tcpd_queue_size_{-1};
  int rss_set_size_{-1};
  std::vector<std::string> ifnames_;
  std::string prefix_;
  std::queue<int> unused_locations_;
  absl::flat_hash_map<int, int> flow_hash_to_location_map_;
  NicConfiguratorInterface* nic_configurator_;
  std::vector<std::unique_ptr<UnixSocketServer>> us_servers_;
};
}  // namespace tcpdirect
#endif
