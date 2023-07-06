#include "include/proto_utils.h"
#include "include/rx_rule_client.h"

#include <absl/functional/bind_front.h>
#include <absl/log/check.h>
#include <absl/log/log.h>
#include <absl/status/status.h>
#include <absl/status/statusor.h>
#include <absl/strings/str_format.h>
#include <absl/time/time.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iterator>
#include <sys/un.h>
#include <unistd.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>

#include "include/flow_steer_ntuple.h"
#include "include/gpu_rxq_configuration_factory.h"
#include "include/nic_configurator_interface.h"
#include "include/rx_rule_manager.h"
#include "include/socket_helper.h"
#include "proto/unix_socket_message.pb.h"
#include "proto/unix_socket_proto.pb.h"
#include "gmock/gmock.h"
#include <google/protobuf/text_format.h>

namespace {
using tcpdirect::FlowSteerNtuple;
using tcpdirect::FlowSteerNtupleProto;
using tcpdirect::SocketAddress;

TEST(ProtoUtilsTest, ConvertProtoToStructSuccess) {
  FlowSteerNtupleProto ntuple_proto;
  ntuple_proto.set_flow_type(1);
  auto *src = ntuple_proto.mutable_src();
  src->set_ip_address("1.2.3.4");
  src->set_port(2);
  auto *dst = ntuple_proto.mutable_dst();
  dst->set_ip_address("5.6.7.8");
  dst->set_port(3);
  auto proto_struct = tcpdirect::ConvertProtoToStruct(ntuple_proto);

  EXPECT_EQ(proto_struct.flow_type, 1);
  EXPECT_EQ(
      tcpdirect::AddressToStr((union SocketAddress *)&proto_struct.src_sin),
      "1.2.3.4");
  EXPECT_EQ(
      tcpdirect::GetAddressPort((union SocketAddress *)&proto_struct.src_sin),
      2);
  EXPECT_EQ(
      tcpdirect::AddressToStr((union SocketAddress *)&proto_struct.dst_sin),
      "5.6.7.8");
  EXPECT_EQ(
      tcpdirect::GetAddressPort((union SocketAddress *)&proto_struct.dst_sin),
      3);
}

TEST(ProtoUtilsTest, ConvertStructToProtoSuccess) {
  FlowSteerNtuple flow_steer_ntuple;
  flow_steer_ntuple.flow_type = 1;
  flow_steer_ntuple.src_sin = tcpdirect::AddressFromStr("1.2.3.4").sin;
  flow_steer_ntuple.dst_sin = tcpdirect::AddressFromStr("5.6.7.8").sin;
  tcpdirect::SetAddressPort((union SocketAddress *)&flow_steer_ntuple.src_sin,
                            2);
  tcpdirect::SetAddressPort((union SocketAddress *)&flow_steer_ntuple.dst_sin,
                            3);
  auto proto = tcpdirect::ConvertStructToProto(flow_steer_ntuple);

  EXPECT_EQ(proto.flow_type(), 1);
  EXPECT_EQ(proto.src().ip_address(), "1.2.3.4");
  EXPECT_EQ(proto.src().port(), 2);
  EXPECT_EQ(proto.dst().ip_address(), "5.6.7.8");
  EXPECT_EQ(proto.dst().port(), 3);
}

} // namespace