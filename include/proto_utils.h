#ifndef EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_PROTO_UTILS_H_
#define EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_PROTO_UTILS_H_

#include "include/flow_steer_ntuple.h"
#include "proto/unix_socket_proto.pb.h"

namespace tcpdirect {
struct FlowSteerNtuple ConvertProtoToStruct(
    const FlowSteerNtupleProto& ntuple_proto);
FlowSteerNtupleProto ConvertStructToProto(
    const struct FlowSteerNtuple& ntuple_struct);

}  // namespace tcpdirect

#endif  // EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_PROTO_UTILS_H_
