# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(FindProtobuf)
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIR})

file(DOWNLOAD https://raw.githubusercontent.com/googleapis/googleapis/master/google/rpc/code.proto ${CMAKE_CURRENT_BINARY_DIR}/code.proto)
file(DOWNLOAD https://raw.githubusercontent.com/googleapis/googleapis/master/google/rpc/status.proto ${CMAKE_CURRENT_BINARY_DIR}/status.proto)

protobuf_generate_cpp(PROTO_SRC PROTO_HEADER
    ${CMAKE_CURRENT_BINARY_DIR}/code.proto
    ${CMAKE_CURRENT_BINARY_DIR}/status.proto
)
add_library(google_status_proto ${PROTO_HEADER} ${PROTO_SRC})
