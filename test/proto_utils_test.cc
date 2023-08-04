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

#include "include/proto_utils.h"

#include <absl/functional/bind_front.h>
#include <absl/log/check.h>
#include <absl/log/log.h>
#include <absl/status/status.h>
#include <absl/status/statusor.h>
#include <absl/strings/str_format.h>
#include <absl/time/time.h>
#include <gmock/gmock.h>
#include <google/protobuf/text_format.h>
#include <gtest/gtest.h>
#include <sys/un.h>
#include <unistd.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>

#include "gmock/gmock.h"
#include "include/flow_steer_ntuple.h"
#include "include/gpu_rxq_configuration_factory.h"
#include "include/nic_configurator_interface.h"
#include "include/rx_rule_client.h"
#include "include/rx_rule_manager.h"
#include "include/socket_helper.h"
#include "proto/unix_socket_message.pb.h"
#include "proto/unix_socket_proto.pb.h"

namespace {
using gpudirect_tcpxd::FlowSteerNtuple;
using gpudirect_tcpxd::FlowSteerNtupleProto;
using gpudirect_tcpxd::SocketAddress;

TEST(ProtoUtilsTest, ConvertProtoToStructSuccess) {
  FlowSteerNtupleProto ntuple_proto;
  ntuple_proto.set_flow_type(1);
  auto *src = ntuple_proto.mutable_src();
  src->set_ip_address("1.2.3.4");
  src->set_port(2);
  auto *dst = ntuple_proto.mutable_dst();
  dst->set_ip_address("5.6.7.8");
  dst->set_port(3);
  auto proto_struct = gpudirect_tcpxd::ConvertProtoToStruct(ntuple_proto);

  EXPECT_EQ(proto_struct.flow_type, 1);
  EXPECT_EQ(gpudirect_tcpxd::AddressToStr(
                (union SocketAddress *)&proto_struct.src_sin),
            "1.2.3.4");
  EXPECT_EQ(gpudirect_tcpxd::GetAddressPort(
                (union SocketAddress *)&proto_struct.src_sin),
            2);
  EXPECT_EQ(gpudirect_tcpxd::AddressToStr(
                (union SocketAddress *)&proto_struct.dst_sin),
            "5.6.7.8");
  EXPECT_EQ(gpudirect_tcpxd::GetAddressPort(
                (union SocketAddress *)&proto_struct.dst_sin),
            3);
}

TEST(ProtoUtilsTest, ConvertStructToProtoSuccess) {
  FlowSteerNtuple flow_steer_ntuple;
  flow_steer_ntuple.flow_type = 1;
  flow_steer_ntuple.src_sin = gpudirect_tcpxd::AddressFromStr("1.2.3.4").sin;
  flow_steer_ntuple.dst_sin = gpudirect_tcpxd::AddressFromStr("5.6.7.8").sin;
  gpudirect_tcpxd::SetAddressPort(
      (union SocketAddress *)&flow_steer_ntuple.src_sin, 2);
  gpudirect_tcpxd::SetAddressPort(
      (union SocketAddress *)&flow_steer_ntuple.dst_sin, 3);
  auto proto = gpudirect_tcpxd::ConvertStructToProto(flow_steer_ntuple);

  EXPECT_EQ(proto.flow_type(), 1);
  EXPECT_EQ(proto.src().ip_address(), "1.2.3.4");
  EXPECT_EQ(proto.src().port(), 2);
  EXPECT_EQ(proto.dst().ip_address(), "5.6.7.8");
  EXPECT_EQ(proto.dst().port(), 3);
}

}  // namespace