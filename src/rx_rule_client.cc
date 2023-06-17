#include "include/rx_rule_client.h"

#include <absl/status/status.h>
#include <absl/strings/str_format.h>

#include <memory>

#include "include/flow_steer_ntuple.h"
#include "include/proto_utils.h"
#include "include/unix_socket_client.h"
#include "proto/unix_socket_message.pb.h"

namespace tcpdirect {

RxRuleClient::RxRuleClient(const std::string& prefix) {
  prefix_ = prefix;
  if (prefix_.back() == '/') {
    prefix_.pop_back();
  }
}

absl::Status ConnectAndSendMessage(UnixSocketMessage message,
                                   UnixSocketMessage* response,
                                   UnixSocketClient* client) {
  auto status = client->Connect();
  if (!status.ok()) return status;

  client->Send(message);

  auto response_status = client->Receive();

  if (!response_status.ok()) return response_status.status();

  *response = response_status.value();

  return absl::OkStatus();
}

absl::Status RxRuleClient::UpdateFlowSteerRule(
    FlowSteerRuleOp op, const FlowSteerNtuple& flow_steer_ntuple,
    std::string gpu_pci_addr, int qid) {
  std::string server_addr =
      (op == CREATE) ? "rx_rule_manager" : "rx_rule_uninstall";

  auto us_client = std::make_unique<UnixSocketClient>(
      absl::StrFormat("%s/%s", prefix_, server_addr));

  UnixSocketMessage message;

  UnixSocketProto* proto = message.mutable_proto();
  FlowSteerRuleRequest* fsr = proto->mutable_flow_steer_rule_request();
  *fsr->mutable_flow_steer_ntuple() = ConvertStructToProto(flow_steer_ntuple);

  if (!gpu_pci_addr.empty()) {
    fsr->set_gpu_pci_addr(gpu_pci_addr);
  }

  if (qid >= 0) {
    fsr->set_queue_id(qid);
  }

  UnixSocketMessage response;

  if (auto status = ConnectAndSendMessage(message, &response, us_client.get());
      !status.ok()) {
    return status;
  }

  if (!response.has_proto() || !response.proto().has_raw_bytes() ||
      response.proto().raw_bytes() != "Ok.") {
    return absl::InternalError(absl::StrFormat(
        "Updating FlowSteerRule Failed: %s", response.DebugString()));
  }

  return absl::OkStatus();
}

}  // namespace tcpdirect
