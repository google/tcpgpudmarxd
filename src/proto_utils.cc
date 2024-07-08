/*
 Copyright 2024 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

#include "include/proto_utils.h"

#include <absl/log/check.h>
#include <absl/log/log.h>
#include <arpa/inet.h>
#include <linux/ethtool.h>
#include <sys/socket.h>

#include <string>

#include "include/flow_steer_ntuple.h"
#include "include/socket_helper.h"
#include "proto/unix_socket_proto.pb.h"

namespace gpudirect_tcpxd {

struct FlowSteerNtuple ConvertProtoToStruct(
    const FlowSteerNtupleProto& ntuple_proto) {
  struct FlowSteerNtuple ntuple;
  ntuple.flow_type = ntuple_proto.flow_type();

  union SocketAddress src_socket_address =
      AddressFromStr(ntuple_proto.src().ip_address());
  SetAddressPort(&src_socket_address, ntuple_proto.src().port());

  union SocketAddress dst_socket_address =
      AddressFromStr(ntuple_proto.dst().ip_address());
  SetAddressPort(&dst_socket_address, ntuple_proto.dst().port());

  CHECK_EQ(src_socket_address.sa.sa_family, dst_socket_address.sa.sa_family);
  if (src_socket_address.sa.sa_family == AF_INET) {
    ntuple.src_sin = src_socket_address.sin;
    ntuple.dst_sin = dst_socket_address.sin;
  } else {
    ntuple.src_sin6 = src_socket_address.sin6;
    ntuple.dst_sin6 = dst_socket_address.sin6;
  }

  return ntuple;
}
FlowSteerNtupleProto ConvertStructToProto(
    const struct FlowSteerNtuple& ntuple_struct) {
  FlowSteerNtupleProto ntuple_proto;
  ntuple_proto.set_flow_type(ntuple_struct.flow_type);

  auto* proto_src = ntuple_proto.mutable_src();
  auto* proto_dst = ntuple_proto.mutable_dst();

  const auto* src_address =
      ntuple_struct.flow_type == TCP_V4_FLOW
          ? (const union SocketAddress*)&ntuple_struct.src_sin
          : (const union SocketAddress*)&ntuple_struct.src_sin6;
  const auto* dst_address =
      ntuple_struct.flow_type == TCP_V4_FLOW
          ? (const union SocketAddress*)&ntuple_struct.dst_sin
          : (const union SocketAddress*)&ntuple_struct.dst_sin6;

  proto_src->set_ip_address(AddressToStr(src_address));
  proto_src->set_port(GetAddressPort(src_address));

  proto_dst->set_ip_address(AddressToStr(dst_address));
  proto_dst->set_port(GetAddressPort(dst_address));

  return ntuple_proto;
}
}  // namespace gpudirect_tcpxd
