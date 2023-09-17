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
find_package(gRPC REQUIRED)

set(PROTO_FILES
	rx_manager_telemetry.proto
)

add_library(tcpgpudmarxd_telemetry_proto ${PROTO_FILES})
target_link_libraries(tcpgpudmarxd_telemetry_proto
    PUBLIC
        protobuf::libprotobuf
        gRPC::grpc
        gRPC::grpc++

)
target_include_directories(tcpgpudmarxd_telemetry_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

get_target_property(grpc_cpp_plugin_location gRPC::grpc_cpp_plugin LOCATION)
protobuf_generate(TARGET tcpgpudmarxd_telemetry_proto LANGUAGE cpp)
protobuf_generate(TARGET tcpgpudmarxd_telemetry_proto LANGUAGE grpc GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc PLUGIN "protoc-gen-grpc=${grpc_cpp_plugin_location}")