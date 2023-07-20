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

#include <absl/functional/bind_front.h>
#include <absl/log/check.h>
#include <absl/log/log.h>
#include <absl/status/status.h>
#include <absl/status/statusor.h>
#include <absl/strings/str_format.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sys/un.h>
#include <unistd.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>

#include "gmock/gmock.h"
#include "include/flow_steer_ntuple.h"
#include "include/nic_configurator_interface.h"
#include "include/rx_rule_manager.h"
#include "include/socket_helper.h"
#include "proto/unix_socket_message.pb.h"

namespace {

using gpudirect_tcpxd::FlowSteerNtuple;
using gpudirect_tcpxd::FlowSteerRuleRequest;
using gpudirect_tcpxd::RxRuleClient;
using gpudirect_tcpxd::SocketAddress;
using gpudirect_tcpxd::UnixSocketMessage;
using gpudirect_tcpxd::UnixSocketProto;
using gpudirect_tcpxd::UnixSocketServer;

class RxRuleClientTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void AddMockFlowSteerRuleServer(const std::string &server_addr,
                                  const std::string &reply) {
    auto mock_server = std::make_unique<UnixSocketServer>(
        server_addr, [reply](UnixSocketMessage &&request,
                             UnixSocketMessage *response, bool *fin) {
          UnixSocketProto *proto = response->mutable_proto();
          std::string *buffer = proto->mutable_raw_bytes();
          buffer->append(reply.c_str());
        });

    CHECK_OK(mock_server->Start());
    mock_us_servers_.push_back(std::move(mock_server));
  }

  std::vector<std::unique_ptr<UnixSocketServer>> mock_us_servers_;
};

TEST_F(RxRuleClientTest, UpdateFlowSteerRuleSuccess) {
  std::string reply = "Ok.";
  AddMockFlowSteerRuleServer("/tmp/rx_rule_manager", reply);

  std::string prefix = "/tmp";
  RxRuleClient rx_rule_client(prefix);

  gpudirect_tcpxd::FlowSteerRuleOp op = gpudirect_tcpxd::CREATE;
  FlowSteerNtuple flow_steer_ntuple;
  flow_steer_ntuple.src_sin = gpudirect_tcpxd::AddressFromStr("1.2.3.4").sin;
  flow_steer_ntuple.dst_sin = gpudirect_tcpxd::AddressFromStr("5.6.7.8").sin;
  gpudirect_tcpxd::SetAddressPort((union SocketAddress *)&flow_steer_ntuple.src_sin,
                            1234);
  gpudirect_tcpxd::SetAddressPort((union SocketAddress *)&flow_steer_ntuple.dst_sin,
                            5678);
  flow_steer_ntuple.flow_type = 1;

  std::string gpu_pci_addr = "addr";
  int qid = 1;

  EXPECT_EQ(rx_rule_client.UpdateFlowSteerRule(op, flow_steer_ntuple,
                                               gpu_pci_addr, qid),
            absl::OkStatus());
}

TEST_F(RxRuleClientTest, UpdateFlowSteerRuleFailed) {
  std::string reply = "Not Ok.";
  AddMockFlowSteerRuleServer("/tmp/rx_rule_manager", reply);

  std::string prefix = "/tmp";
  RxRuleClient rx_rule_client(prefix);

  gpudirect_tcpxd::FlowSteerRuleOp op = gpudirect_tcpxd::CREATE;
  FlowSteerNtuple flow_steer_ntuple;
  flow_steer_ntuple.src_sin = gpudirect_tcpxd::AddressFromStr("1.2.3.4").sin;
  flow_steer_ntuple.dst_sin = gpudirect_tcpxd::AddressFromStr("5.6.7.8").sin;
  gpudirect_tcpxd::SetAddressPort((union SocketAddress *)&flow_steer_ntuple.src_sin,
                            1234);
  gpudirect_tcpxd::SetAddressPort((union SocketAddress *)&flow_steer_ntuple.dst_sin,
                            5678);
  flow_steer_ntuple.flow_type = 1;

  std::string gpu_pci_addr = "addr";
  int qid = 1;

  auto status = rx_rule_client.UpdateFlowSteerRule(op, flow_steer_ntuple,
                                                   gpu_pci_addr, qid);
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(status.message(),
              testing::HasSubstr("Updating FlowSteerRule Failed:"));
}

TEST_F(RxRuleClientTest, PopBackSuccess) {
  std::string reply = "Ok.";
  AddMockFlowSteerRuleServer("/tmp/rx_rule_manager", reply);

  std::string prefix = "/tmp/";
  RxRuleClient rx_rule_client(prefix);

  gpudirect_tcpxd::FlowSteerRuleOp op = gpudirect_tcpxd::CREATE;
  FlowSteerNtuple flow_steer_ntuple;
  flow_steer_ntuple.src_sin = gpudirect_tcpxd::AddressFromStr("1.2.3.4").sin;
  flow_steer_ntuple.dst_sin = gpudirect_tcpxd::AddressFromStr("5.6.7.8").sin;
  gpudirect_tcpxd::SetAddressPort((union SocketAddress *)&flow_steer_ntuple.src_sin,
                            1234);
  gpudirect_tcpxd::SetAddressPort((union SocketAddress *)&flow_steer_ntuple.dst_sin,
                            5678);
  flow_steer_ntuple.flow_type = 1;

  std::string gpu_pci_addr = "addr";
  int qid = 1;

  EXPECT_EQ(rx_rule_client.UpdateFlowSteerRule(op, flow_steer_ntuple,
                                               gpu_pci_addr, qid),
            absl::OkStatus());
}

TEST_F(RxRuleClientTest, NoServer) {
  std::string prefix = "/tmp";
  RxRuleClient rx_rule_client(prefix);

  gpudirect_tcpxd::FlowSteerRuleOp op = gpudirect_tcpxd::CREATE;
  FlowSteerNtuple flow_steer_ntuple;
  flow_steer_ntuple.src_sin = gpudirect_tcpxd::AddressFromStr("1.2.3.4").sin;
  flow_steer_ntuple.dst_sin = gpudirect_tcpxd::AddressFromStr("5.6.7.8").sin;
  gpudirect_tcpxd::SetAddressPort((union SocketAddress *)&flow_steer_ntuple.src_sin,
                            1234);
  gpudirect_tcpxd::SetAddressPort((union SocketAddress *)&flow_steer_ntuple.dst_sin,
                            5678);
  flow_steer_ntuple.flow_type = 1;

  std::string gpu_pci_addr = "addr";
  int qid = 1;

  auto status = rx_rule_client.UpdateFlowSteerRule(op, flow_steer_ntuple,
                                                   gpu_pci_addr, qid);
  EXPECT_EQ(status.code(), absl::StatusCode::kUnavailable);
}

}  // namespace