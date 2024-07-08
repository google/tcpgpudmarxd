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

ADD_SUBDIRECTORY(proto)

# telemetry base
add_library(telemetry_interface STATIC telemetry_interface.cc)
target_include_directories(
    telemetry_interface
    PUBLIC
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}> $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/third_party>
	$<INSTALL_INTERFACE:${TCPGPUDMARXD_INCLUDE_DIR}>
)
target_link_libraries(telemetry_interface PUBLIC grpc++)
target_link_libraries(telemetry_interface PRIVATE tcpgpudmarxd_telemetry_proto)
target_link_libraries(telemetry_interface PUBLIC ${Protobuf_LIBRARIES})
target_link_libraries(telemetry_interface PUBLIC absl::absl_check absl::flags absl::flags_parse absl::log absl::str_format absl::debugging absl::debugging_internal absl::failure_signal_handler)
add_dependencies(telemetry_interface tcpgpudmarxd_telemetry_proto)

# rx rule client telemetry
add_library(rx_rule_client_telemetry STATIC rx_rule_client_telemetry.cc)
target_include_directories(
    rx_rule_client_telemetry
    PUBLIC
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}> $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/third_party>
	$<INSTALL_INTERFACE:${TCPGPUDMARXD_INCLUDE_DIR}>
)
target_link_libraries(rx_rule_client_telemetry PUBLIC grpc++)
target_link_libraries(rx_rule_client_telemetry PRIVATE tcpgpudmarxd_telemetry_proto)
target_link_libraries(rx_rule_client_telemetry PUBLIC ${Protobuf_LIBRARIES})
target_link_libraries(rx_rule_client_telemetry PUBLIC telemetry_interface)
add_dependencies(rx_rule_client_telemetry tcpgpudmarxd_telemetry_proto)



# rx rule manager telemetry
add_library(rx_rule_manager_telemetry STATIC rx_rule_manager_telemetry.cc)
target_include_directories(rx_rule_manager_telemetry PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_link_libraries(rx_rule_manager_telemetry PUBLIC grpc++)
target_link_libraries(rx_rule_manager_telemetry PUBLIC tcpgpudmarxd_telemetry_proto)
target_link_libraries(rx_rule_manager_telemetry PUBLIC ${Protobuf_LIBRARIES})
target_link_libraries(rx_rule_manager_telemetry PUBLIC telemetry_interface)


# gpu mem importer telemetry
add_library(gpu_mem_importer_telemetry STATIC gpu_mem_importer_telemetry.cc)
target_include_directories(
    gpu_mem_importer_telemetry
    PUBLIC
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}> $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/third_party>
	$<INSTALL_INTERFACE:${TCPGPUDMARXD_INCLUDE_DIR}>
)
target_link_libraries(gpu_mem_importer_telemetry PUBLIC grpc++)
target_link_libraries(gpu_mem_importer_telemetry PUBLIC tcpgpudmarxd_telemetry_proto)
target_link_libraries(gpu_mem_importer_telemetry PUBLIC ${Protobuf_LIBRARIES})
target_link_libraries(gpu_mem_importer_telemetry PUBLIC telemetry_interface)
add_dependencies(gpu_mem_importer_telemetry tcpgpudmarxd_telemetry_proto)

# gpu mem exporter telemetry
add_library(gpu_mem_exporter_telemetry STATIC gpu_mem_exporter_telemetry.cc)
target_include_directories(gpu_mem_exporter_telemetry PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_link_libraries(gpu_mem_exporter_telemetry PUBLIC grpc++)
target_link_libraries(gpu_mem_exporter_telemetry PUBLIC tcpgpudmarxd_telemetry_proto)
target_link_libraries(gpu_mem_exporter_telemetry PUBLIC ${Protobuf_LIBRARIES})
target_link_libraries(gpu_mem_exporter_telemetry PUBLIC telemetry_interface)

set(client_telemetry_libs
    rx_rule_client_telemetry
    gpu_mem_importer_telemetry
    tcpgpudmarxd_telemetry_proto
    telemetry_interface
)

install(TARGETS ${client_telemetry_libs} EXPORT gpudirect_tcpxd DESTINATION "${TCPGPUDMARXD_LIB_DIR}")