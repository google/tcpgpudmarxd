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

#ifndef EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_RX_RULE_REQUESTER_H_
#define EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_RX_RULE_REQUESTER_H_

#include <absl/status/status.h>

#include <memory>
#include <string>

#include "include/flow_steer_ntuple.h"
#include "include/unix_socket_client.h"
#include "proto/unix_socket_message.pb.h"

namespace gpudirect_tcpxd {

enum FlowSteerRuleOp {
  CREATE,
  DELETE,
};

class RxRuleClient {
 public:
  explicit RxRuleClient(const std::string& prefix,
                        std::function<int()> vf_reset_cb = nullptr);
  absl::Status UpdateFlowSteerRule(FlowSteerRuleOp op,
                                   const FlowSteerNtuple& flow_steer_ntuple,
                                   std::string gpu_pci_addr = "", int qid = -1);
  absl::StatusOr<UnixSocketClient*> CreateSkIfReq(FlowSteerRuleOp op);

  std::unique_ptr<UnixSocketClient> sk_cli_;

 private:
  absl::Status SendMsg(UnixSocketMessage message, UnixSocketMessage* response,
                       UnixSocketClient* client);

  std::string prefix_;
  std::unique_ptr<UnixSocketClient> sk_cli_uninstall_;
  int epoll_fd_{-1};
  std::function<int()> vf_reset_cb_;
};
}  // namespace gpudirect_tcpxd

#endif  // EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_RX_RULE_REQUESTER_H_
