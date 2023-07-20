find_package(absl REQUIRED)
find_package(Protobuf REQUIRED)

add_library(benchmark_common STATIC benchmark_common.cc)
target_include_directories(benchmark_common PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(benchmark_common PRIVATE absl::log absl::status absl::strings absl::str_format)

SET(TCP_EVENT_HANDLER_SRC
  tcp_receive_event_handler.cc
  tcp_receive_tcp_direct_event_handler.cc
  tcp_send_event_handler.cc
  tcp_send_tcp_direct_event_handler.cc
)

add_library(tcp_event_handlers STATIC ${TCP_EVENT_HANDLER_SRC})
target_include_directories(tcp_event_handlers PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(tcp_event_handlers PRIVATE absl::log absl::absl_check absl::status absl::strings absl::str_format)
target_link_libraries(tcp_event_handlers PRIVATE validation)
