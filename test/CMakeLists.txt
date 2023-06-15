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

include(GoogleTest)
gtest_discover_tests(unix_socket_test)
