#include "include/rx_rule_client.h"

#include <memory>

#include "include/flow_steer_ntuple.h"
#include "include/unix_socket_client.h"
#include "proto/unix_socket_message.pb.h"

#include <absl/status/status.h>
#include <absl/strings/str_format.h>

namespace tcpdirect {

absl::Status ConnectAndSendMessage(const FlowSteerNtuple& flow_steer_ntuple,
                                   UnixSocketClient* client) {
  auto status = client->Connect();
  if (!status.ok()) return status;

  UnixSocketMessage message;
  UnixSocketProto* proto = message.mutable_proto();
  proto->mutable_raw_bytes()->resize(sizeof(FlowSteerNtuple));
  memcpy(proto->mutable_raw_bytes()->data(), &flow_steer_ntuple,
         sizeof(FlowSteerNtuple));

  client->Send(message);

  auto response = client->Receive();
  if (!response.ok()) return response.status();

  if (!response->has_proto() || !response->proto().has_raw_bytes() ||
      response->proto().raw_bytes() != "Ok.") {
    return absl::InternalError(response->DebugString());
  }

  return absl::OkStatus();
}

absl::Status RxRuleClient::RequestFlowSteerRule(
    const FlowSteerNtuple& flow_steer_ntuple) {
  auto us_client = std::make_unique<UnixSocketClient>(
      absl::StrFormat("%s/rx_rule_manager", prefix_));
  return ConnectAndSendMessage(flow_steer_ntuple, us_client.get());
}

absl::Status RxRuleClient::DeleteFlowSteerRule(
    const FlowSteerNtuple& flow_steer_ntuple) {
  auto us_client = std::make_unique<UnixSocketClient>(
      absl::StrFormat("%s/rx_rule_uninstall", prefix_));
  return ConnectAndSendMessage(flow_steer_ntuple, us_client.get());
}

}  // namespace tcpdirect
