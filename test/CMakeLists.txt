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
target_link_libraries(unix_socket_test PRIVATE absl::log absl::status absl::statusor absl::str_format)
target_link_libraries(unix_socket_test PUBLIC tcpgpudmarxd_proto)
target_link_libraries(unix_socket_test PUBLIC ${Protobuf_LIBRARIES})
target_link_libraries(unix_socket_test PUBLIC unix_socket_lib)
target_link_libraries(unix_socket_test PUBLIC GTest::gtest_main)

add_executable(
    rx_rule_manager_test
    rx_rule_manager_test.cc
)
target_include_directories(rx_rule_manager_test PUBLIC ${CMAKE_SOURCE_DIR})
target_include_directories(rx_rule_manager_test PRIVATE ${CMAKE_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_link_libraries(rx_rule_manager_test PRIVATE absl::log absl::status absl::statusor absl::str_format)
target_link_libraries(rx_rule_manager_test PUBLIC tcpgpudmarxd_proto)
target_link_libraries(rx_rule_manager_test PUBLIC ${Protobuf_LIBRARIES})
target_link_libraries(rx_rule_manager_test PUBLIC rx_rule_manager)
target_link_libraries(rx_rule_manager_test PUBLIC unix_socket_lib)
target_link_libraries(rx_rule_manager_test PUBLIC GTest::gtest_main)
target_link_libraries(rx_rule_manager_test PUBLIC GTest::gmock_main)

add_executable(
    rx_rule_client_test
    rx_rule_client_test.cc
)
target_include_directories(rx_rule_client_test PUBLIC ${CMAKE_SOURCE_DIR})
target_include_directories(rx_rule_client_test PRIVATE ${CMAKE_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
target_link_libraries(rx_rule_client_test PRIVATE absl::log absl::status absl::statusor absl::str_format)
target_link_libraries(rx_rule_client_test PUBLIC tcpgpudmarxd_proto)
target_link_libraries(rx_rule_client_test PUBLIC ${Protobuf_LIBRARIES})
target_link_libraries(rx_rule_client_test PUBLIC rx_rule_client)
target_link_libraries(rx_rule_client_test PUBLIC unix_socket_lib)
target_link_libraries(rx_rule_client_test PUBLIC GTest::gtest_main)
target_link_libraries(rx_rule_client_test PUBLIC GTest::gmock_main)

include(GoogleTest)
gtest_discover_tests(unix_socket_test)
gtest_discover_tests(rx_rule_manager_test)
gtest_discover_tests(rx_rule_client_test)

