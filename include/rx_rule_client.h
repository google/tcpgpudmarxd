#ifndef EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_RX_RULE_REQUESTER_H_
#define EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_RX_RULE_REQUESTER_H_

#include <absl/status/status.h>

#include <memory>
#include <string>

#include "include/flow_steer_ntuple.h"
#include "include/unix_socket_client.h"
namespace tcpdirect {

enum FlowSteerRuleOp {
  CREATE,
  DELETE,
};

class RxRuleClient {
 public:
  explicit RxRuleClient(const std::string& prefix);
  absl::Status UpdateFlowSteerRule(FlowSteerRuleOp op,
                                   const FlowSteerNtuple& flow_steer_ntuple,
                                   std::string gpu_pci_addr = "", int qid = -1);

 private:
  std::string prefix_;
};
}  // namespace tcpdirect

#endif  // EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_RX_RULE_REQUESTER_H_
