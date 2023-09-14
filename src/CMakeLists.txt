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

# gpu rxq configuration

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(absl REQUIRED)
find_package(Protobuf REQUIRED)

set(GPU_RXQ_CONFIG_SRC
        a3_gpu_rxq_configurator.cu
        a3vm_gpu_rxq_configurator.cc
        gpu_rxq_configuration_factory.cc
        monstertruck_gpu_rxq_configurator.cc
        predvt_gpu_rxq_configurator.cc
        pci_helpers.cc
)

add_library(gpu_rxq_configurator STATIC ${GPU_RXQ_CONFIG_SRC})
target_include_directories(gpu_rxq_configurator PRIVATE ${CMAKE_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_include_directories(gpu_rxq_configurator PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(gpu_rxq_configurator PRIVATE CUDA::cuda_driver)
target_link_libraries(gpu_rxq_configurator PRIVATE absl::absl_check absl::log absl::flags absl::status absl::str_format)
target_link_libraries(gpu_rxq_configurator PUBLIC tcpgpudmarxd_proto)
target_link_libraries(gpu_rxq_configurator PUBLIC ${Protobuf_LIBRARIES})
set_target_properties(gpu_rxq_configurator PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

# NIC configurator

set(NIC_CONFIGURATOR_SRC
        dummy_ethtool_nic_configurator.cc
        ethtool_no_headersplit_nic_configurator.cc
        ethtool_nic_configurator.cc
        nic_configurator_factory.cc
)

add_library(nic_configurator STATIC ${NIC_CONFIGURATOR_SRC})
target_include_directories(nic_configurator PRIVATE ${CMAKE_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_include_directories(nic_configurator PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(nic_configurator PRIVATE absl::absl_check absl::log absl::status absl::str_format)
target_link_libraries(nic_configurator PUBLIC tcpgpudmarxd_proto)
target_link_libraries(nic_configurator PUBLIC ${Protobuf_LIBRARIES})

# gpu page exporter interface

add_library(gpu_page_exporter_interface STATIC
        gpu_page_exporter_interface.cc)
target_include_directories(gpu_page_exporter_interface PUBLIC ${CMAKE_SOURCE_DIR})
target_include_directories(gpu_page_exporter_interface PRIVATE ${CMAKE_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_link_libraries(gpu_page_exporter_interface PRIVATE absl::log absl::status absl::str_format)
target_link_libraries(gpu_page_exporter_interface PUBLIC tcpgpudmarxd_proto)
target_link_libraries(gpu_page_exporter_interface PUBLIC ${Protobuf_LIBRARIES})

# unix socket lib

set(UNIX_SOCKET_CLIENT_LIB_SRC
        unix_socket_client.cc
        unix_socket_connection.cc
)

set(UNIX_SOCKET_SERVER_LIB_SRC
        unix_socket_connection.cc
        unix_socket_server.cc
)

add_library(unix_socket_server_lib STATIC ${UNIX_SOCKET_SERVER_LIB_SRC})
target_include_directories(unix_socket_server_lib PUBLIC ${CMAKE_SOURCE_DIR} )
target_include_directories(unix_socket_server_lib PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR}/third_party ${Protobuf_INCLUDE_DIRS})
target_link_libraries(unix_socket_server_lib PRIVATE absl::flat_hash_map absl::log absl::status absl::statusor absl::str_format)
target_link_libraries(unix_socket_server_lib PUBLIC ${Protobuf_LIBRARIES})
target_link_libraries(unix_socket_server_lib PUBLIC tcpgpudmarxd_proto)

add_library(unix_socket_client_lib STATIC ${UNIX_SOCKET_CLIENT_LIB_SRC})
target_include_directories(unix_socket_client_lib PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> $<INSTALL_INTERFACE:${TCPGPUDMARXD_INCLUDE_DIR}>)
target_include_directories(unix_socket_client_lib PRIVATE $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}> $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/third_party> $<BUILD_INTERFACE:${Protobuf_INCLUDE_DIRS}>)
target_link_libraries(unix_socket_client_lib PRIVATE absl::flat_hash_map absl::log absl::status absl::statusor absl::str_format)
target_link_libraries(unix_socket_client_lib PUBLIC ${Protobuf_LIBRARIES})
add_dependencies(unix_socket_client_lib tcpgpudmarxd_proto)


# socket helper

add_library(socket_helper STATIC socket_helper.cc)
target_include_directories(socket_helper PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>)
target_link_libraries(socket_helper PRIVATE absl::absl_check absl::log)

# proto utils

add_library(proto_utils STATIC proto_utils.cc)
target_include_directories(proto_utils PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>)
target_include_directories(proto_utils PRIVATE $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}> $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/third_party> $<BUILD_INTERFACE:${Protobuf_INCLUDE_DIRS}>)
target_link_libraries(proto_utils PRIVATE absl::absl_check absl::log)
target_link_libraries(proto_utils PUBLIC socket_helper)
target_link_libraries(proto_utils PUBLIC ${Protobuf_LIBRARIES})
add_dependencies(proto_utils tcpgpudmarxd_proto)

# rx rule client
add_library(rx_rule_client STATIC rx_rule_client.cc)
target_include_directories(
        rx_rule_client
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${TCPGPUDMARXD_INCLUDE_DIR}>
)
target_include_directories(rx_rule_client PRIVATE $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}> $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/third_party>  ${Protobuf_INCLUDE_DIRS})
target_link_libraries(rx_rule_client PRIVATE absl::log absl::status absl::str_format)
target_link_libraries(rx_rule_client PUBLIC proto_utils)
target_link_libraries(rx_rule_client PUBLIC rx_rule_client_telemetry)
target_link_libraries(rx_rule_client PUBLIC ${Protobuf_LIBRARIES})
add_dependencies(rx_rule_client tcpgpudmarxd_proto)

# rx rule manager
add_library(rx_rule_manager STATIC rx_rule_manager.cc)
target_include_directories(rx_rule_manager PUBLIC ${CMAKE_SOURCE_DIR})
target_include_directories(rx_rule_manager PRIVATE ${CMAKE_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_link_libraries(rx_rule_manager PRIVATE absl::log absl::status absl::statusor absl::str_format)
target_link_libraries(rx_rule_manager PUBLIC tcpgpudmarxd_proto)
target_link_libraries(rx_rule_manager PUBLIC proto_utils)
target_link_libraries(rx_rule_manager PUBLIC ${Protobuf_LIBRARIES})
target_link_libraries(rx_rule_manager PUBLIC GTest::gtest_main)


set(client_headers
        ${CMAKE_SOURCE_DIR}/include/rx_rule_client.h
        ${CMAKE_SOURCE_DIR}/include/proto_utils.h
        ${CMAKE_SOURCE_DIR}/include/socket_helper.h

        ${CMAKE_SOURCE_DIR}/include/flow_steer_ntuple.h
        ${CMAKE_SOURCE_DIR}/include/unix_socket_client.h
        ${CMAKE_SOURCE_DIR}/include/unix_socket_connection.h
)
install(FILES ${client_headers} DESTINATION "${TCPGPUDMARXD_INCLUDE_DIR}")

set(client_libs
        rx_rule_client
        unix_socket_client_lib
        proto_utils
        socket_helper
)
install(TARGETS ${client_libs} EXPORT gpudirect_tcpxd DESTINATION "${TCPGPUDMARXD_LIB_DIR}")