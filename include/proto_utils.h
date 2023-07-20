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

#ifndef EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_PROTO_UTILS_H_
#define EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_PROTO_UTILS_H_

#include "include/flow_steer_ntuple.h"
#include "proto/unix_socket_proto.pb.h"

namespace gpudirect_tcpxd {
struct FlowSteerNtuple ConvertProtoToStruct(
    const FlowSteerNtupleProto& ntuple_proto);
FlowSteerNtupleProto ConvertStructToProto(
    const struct FlowSteerNtuple& ntuple_struct);

}  // namespace gpudirect_tcpxd

#endif  // EXPERIMENTAL_USERS_CHECHENGLIN_TCPGPUDMAD_INCLUDE_PROTO_UTILS_H_
