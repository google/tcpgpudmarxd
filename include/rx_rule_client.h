#ifndef EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_RX_RULE_REQUESTER_H_
#define EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_RX_RULE_REQUESTER_H_

#include <memory>
#include <string>

#include "include/flow_steer_ntuple.h"
#include "include/unix_socket_client.h"
#include "third_party/absl/status/status.h"
namespace tcpdirect {
class RxRuleClient {
 public:
  explicit RxRuleClient(const std::string& prefix) {
    prefix_ = prefix;
    if (prefix_.back() == '/') {
      prefix_.pop_back();
    }
  }
  absl::Status RequestFlowSteerRule(const FlowSteerNtuple& flow_steer_ntuple);
  absl::Status DeleteFlowSteerRule(const FlowSteerNtuple& flow_steer_ntuple);

 private:
  std::string prefix_;
};
}  // namespace tcpdirect

#endif  // EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_RX_RULE_REQUESTER_H_
