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

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
	unix_socket_test
	unix_socket_server_client_test.cc
)
target_include_directories(unix_socket_test PUBLIC ${CMAKE_SOURCE_DIR})
target_include_directories(unix_socket_test PRIVATE ${CMAKE_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_link_libraries(unix_socket_test PUBLIC ${Protobuf_LIBRARIES})
target_link_libraries(unix_socket_test PRIVATE absl::log absl::status absl::statusor absl::str_format)
target_link_libraries(unix_socket_test PUBLIC tcpgpudmarxd_proto)
target_link_libraries(unix_socket_test PUBLIC google_status_proto)
target_link_libraries(unix_socket_test PUBLIC unix_socket_client_lib unix_socket_server_lib)
target_link_libraries(unix_socket_test PUBLIC GTest::gtest_main)

add_executable(
    rx_rule_manager_test
    rx_rule_manager_test.cc
)
target_include_directories(rx_rule_manager_test PUBLIC ${CMAKE_SOURCE_DIR})
target_include_directories(rx_rule_manager_test PRIVATE ${CMAKE_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_link_libraries(rx_rule_manager_test PUBLIC rx_rule_manager)
target_link_libraries(rx_rule_manager_test PUBLIC ${Protobuf_LIBRARIES})
target_link_libraries(rx_rule_manager_test PRIVATE absl::log absl::status absl::statusor absl::str_format)
target_link_libraries(rx_rule_manager_test PUBLIC tcpgpudmarxd_proto)
target_link_libraries(rx_rule_manager_test PUBLIC google_status_proto)
target_link_libraries(rx_rule_manager_test PUBLIC unix_socket_client_lib unix_socket_server_lib)
target_link_libraries(rx_rule_manager_test PUBLIC GTest::gtest_main)
target_link_libraries(rx_rule_manager_test PUBLIC GTest::gmock_main)

add_executable(
    rx_rule_client_test
    rx_rule_client_test.cc
)
target_include_directories(rx_rule_client_test PUBLIC ${CMAKE_SOURCE_DIR})
target_include_directories(rx_rule_client_test PRIVATE ${CMAKE_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_link_libraries(rx_rule_client_test PUBLIC rx_rule_client)
target_link_libraries(rx_rule_client_test PUBLIC ${Protobuf_LIBRARIES})
target_link_libraries(rx_rule_client_test PRIVATE absl::log absl::status absl::statusor absl::str_format)
target_link_libraries(rx_rule_client_test PUBLIC tcpgpudmarxd_proto)
target_link_libraries(rx_rule_client_test PUBLIC google_status_proto)
target_link_libraries(rx_rule_client_test PUBLIC unix_socket_client_lib unix_socket_server_lib)
target_link_libraries(rx_rule_client_test PUBLIC GTest::gtest_main)
target_link_libraries(rx_rule_client_test PUBLIC GTest::gmock_main)

add_executable(
    gpu_rxq_configuration_factory_test
    gpu_rxq_configuration_factory_test.cc
)
target_include_directories(gpu_rxq_configuration_factory_test PUBLIC ${CMAKE_SOURCE_DIR})
target_include_directories(gpu_rxq_configuration_factory_test PRIVATE ${CMAKE_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_link_libraries(gpu_rxq_configuration_factory_test PUBLIC gpu_rxq_configurator)
target_link_libraries(gpu_rxq_configuration_factory_test PUBLIC ${Protobuf_LIBRARIES})
target_link_libraries(gpu_rxq_configuration_factory_test PRIVATE absl::log absl::status absl::statusor absl::str_format)
target_link_libraries(gpu_rxq_configuration_factory_test PUBLIC tcpgpudmarxd_proto)
target_link_libraries(gpu_rxq_configuration_factory_test PUBLIC google_status_proto)
target_link_libraries(gpu_rxq_configuration_factory_test PUBLIC unix_socket_client_lib unix_socket_server_lib)
target_link_libraries(gpu_rxq_configuration_factory_test PUBLIC GTest::gtest_main)
target_link_libraries(gpu_rxq_configuration_factory_test PUBLIC GTest::gmock_main)

add_executable(
    proto_utils_test
    proto_utils_test.cc
)
target_include_directories(proto_utils_test PUBLIC ${CMAKE_SOURCE_DIR})
target_include_directories(proto_utils_test PRIVATE ${CMAKE_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_link_libraries(proto_utils_test PUBLIC proto_utils)
target_link_libraries(proto_utils_test PUBLIC ${Protobuf_LIBRARIES}  )
target_link_libraries(proto_utils_test PRIVATE absl::log absl::status absl::statusor absl::str_format)
target_link_libraries(proto_utils_test PUBLIC tcpgpudmarxd_proto)
target_link_libraries(proto_utils_test PUBLIC google_status_proto)
target_link_libraries(proto_utils_test PUBLIC unix_socket_client_lib unix_socket_server_lib)
target_link_libraries(proto_utils_test PUBLIC GTest::gtest_main)
target_link_libraries(proto_utils_test PUBLIC GTest::gmock_main)

add_executable(
    socket_helper_test
    socket_helper_test.cc
)
target_include_directories(socket_helper_test PUBLIC ${CMAKE_SOURCE_DIR})
target_include_directories(socket_helper_test PRIVATE ${CMAKE_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_link_libraries(socket_helper_test PRIVATE absl::log absl::status absl::statusor absl::str_format)
target_link_libraries(socket_helper_test PUBLIC tcpgpudmarxd_proto)
target_link_libraries(socket_helper_test PUBLIC ${Protobuf_LIBRARIES}  )
target_link_libraries(socket_helper_test PUBLIC socket_helper)
target_link_libraries(socket_helper_test PUBLIC unix_socket_client_lib unix_socket_server_lib)
target_link_libraries(socket_helper_test PUBLIC GTest::gtest_main)
target_link_libraries(socket_helper_test PUBLIC GTest::gmock_main)

include(GoogleTest)
gtest_discover_tests(unix_socket_test)
gtest_discover_tests(rx_rule_manager_test)
gtest_discover_tests(rx_rule_client_test)
gtest_discover_tests(gpu_rxq_configuration_factory_test)
gtest_discover_tests(proto_utils_test)
gtest_discover_tests(socket_helper_test)




