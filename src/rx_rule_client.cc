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

#include "include/rx_rule_client.h"

#include <absl/log/log.h>
#include <absl/status/status.h>
#include <absl/status/statusor.h>
#include <absl/strings/str_format.h>

#include <memory>

#include "code.pb.h"
#include "include/flow_steer_ntuple.h"
#include "include/proto_utils.h"
#include "include/unix_socket_client.h"
#include "proto/unix_socket_message.pb.h"
#include "proto/unix_socket_proto.pb.h"
#include "telemetry/rx_rule_client_telemetry.h"

namespace gpudirect_tcpxd {

RxRuleClient::RxRuleClient(const std::string& prefix,
                           std::function<int()> vf_reset_cb)
    : vf_reset_cb_(vf_reset_cb) {
  prefix_ = prefix;
  if (prefix_.back() == '/') {
    prefix_.pop_back();
  }
}

// Create a UnixSocketClient for this RxRuleClient
// if one doesn't exist for it already, and return it.
//
// Technically RxRuleClient has 2 UnixSocketClients:
// 1 for installing flow-steering rules, and another for
// uninstalling them.
//
// Only the install UnixSocketClient will trigger the VF
// reset callback.
//
// TODO: This function should just be rolled into the
// constructor. And a socket for install/uninstall might
// be better than each operation using its own socket.
absl::StatusOr<UnixSocketClient*> RxRuleClient::CreateSkIfReq(
    FlowSteerRuleOp op) {
  std::string server_addr;
  UnixSocketClient* us_client;
  absl::Status status;

  static FlowSteerRuleClientTelemetry telemetry;
  telemetry.Start();  // No-op if started

  if (op == CREATE) {
    // TODO basically the same code in the else-clause, move to macro?
    if (!sk_cli_) {
      server_addr = "rx_rule_manager";
      sk_cli_ = std::make_unique<UnixSocketClient>(
          absl::StrFormat("%s/%s", prefix_, server_addr), vf_reset_cb_);

      status = sk_cli_.get()->Connect();
      if (!status.ok()) {
        telemetry.IncrementInstallFailure();
        telemetry.IncrementFailureAndCause(status.ToString());
        LOG(ERROR) << absl::StrFormat(
            "%s Unix Socket Client failed to connect %s", server_addr,
            status.ToString());
        sk_cli_.release();
        return status;
      }
    }
    us_client = sk_cli_.get();
  } else {
    if (!sk_cli_uninstall_) {
      server_addr = "rx_rule_uninstall";
      sk_cli_uninstall_ = std::make_unique<UnixSocketClient>(
          absl::StrFormat("%s/%s", prefix_, server_addr), nullptr);

      status = sk_cli_uninstall_.get()->Connect();
      if (!status.ok()) {
        telemetry.IncrementInstallFailure();
        telemetry.IncrementFailureAndCause(status.ToString());
        LOG(ERROR) << absl::StrFormat("%s Unix Socket Client failed to connect",
                                      server_addr);
        sk_cli_uninstall_.release();
        return status;
      }
    }
    us_client = sk_cli_uninstall_.get();
  }

  return us_client;
}

absl::Status RxRuleClient::SendMsg(UnixSocketMessage message,
                                   UnixSocketMessage* response,
                                   UnixSocketClient* client) {
  if (!client) LOG(ERROR) << "SendMsg with invalid socket";

  client->Send(message);

  auto response_status = client->Receive();

  if (!response_status.ok()) return response_status.status();

  *response = response_status.value();

  return absl::OkStatus();
}

absl::Status RxRuleClient::UpdateFlowSteerRule(
    FlowSteerRuleOp op, const FlowSteerNtuple& flow_steer_ntuple,
    std::string gpu_pci_addr, int qid) {
  absl::Status status;
  absl::StatusOr<UnixSocketClient*> us_client;
  static FlowSteerRuleClientTelemetry telemetry;
  telemetry.Start();  // No-op if started

  us_client = CreateSkIfReq(op);

  if (!us_client.ok()) {
    return us_client.status();
  }

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

  absl::Time start = absl::Now();
  status = SendMsg(message, &response, *us_client);
  telemetry.AddLatency(absl::Now() - start);

  if (!status.ok()) {
    telemetry.IncrementInstallFailure();
    telemetry.IncrementFailureAndCause(status.ToString());
    return status;
  }

  if (response.proto().status().code() != google::rpc::Code::OK) {
    telemetry.IncrementInstallFailure();
    telemetry.IncrementFailureAndCause(response.proto().status().message());
    return absl::Status(absl::StatusCode(response.proto().status().code()),
                        response.proto().status().message());
  }
  telemetry.IncrementInstallSuccess();
  return absl::OkStatus();
}

}  // namespace gpudirect_tcpxd
